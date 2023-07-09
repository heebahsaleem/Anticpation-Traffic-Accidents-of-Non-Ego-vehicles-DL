import numpy as np
import cv2
import os
import gi
import time
from threading import Thread
import sys
import matplotlib.pyplot as plt
import argparse
import yaml
import glob
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

import torch

from mmdet.apis import init_detector, inference_detector
from lib.sort import *
from lib.models import FOL
from lib.models import AP

from lib.tracking import xy_limit, xy2cxcy
from lib.utils import delta_bbox


def parse_args(a=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config',
                        dest='config_file',
                        default='configs/infer_test.yaml',
                        # type=argparse.FileType(mode='r'),
                        help='The yaml configuration file')
    if a == None:
        args, unprocessed_args = parser.parse_known_args()
    else:
        args, unprocessed_args = parser.parse_known_args(args=a)

    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.load(f))

    args = parser.parse_args(unprocessed_args)

    return args


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, rtsp_server='rtsp://192.168.0.159:20001/stream1'):

        self.mode = 'stream'
        self.capture = None
        self.stopFlag = False
        self.args = args

        # sources = [sources]

        # n = len(s)
        self.img = None
        cap = cv2.VideoCapture(rtsp_server)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) % 100
        _, self.img = cap.read()  # guarantee first frame
        self.capture = cap
        print(' success (%gx%g at %.2f FPS).' % (w, h, fps))


        thread = Thread(target=self.update)
        thread.start()
        print('')  # newline



    def update(self):
        # Read next stream frame in a daemon thread
        cap = self.capture
        while cap.isOpened() and not self.stopFlag:
            cap.grab()
            _, self.img = cap.retrieve()
            """plt.figure()
            plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
            plt.show()"""
            time.sleep(0.1)  # wait time


    def stop(self):
        self.stopFlag = True
        print("stop!")


class Inference:
    def __init__(self, args):

        self.args = args
        self.det_targets = [1, 2, 3, 4, 6, 8]
        self.batch_infer = self.args.batch_infer
        self.repeat = self.args.repeat
        self.var = vars(args)
        if 'from_folder' in self.var.keys():
            self.img_folder = self.args.from_folder
        else:
            self.img_folder = None
            self.streamer = LoadStreams(rtsp_server=self.args.rtsp)

        if not self.img_folder is None:
            self.img_dirs = glob.glob(os.path.join(self.img_folder, '*.{}'.format(self.args.img_format)))
            self.img_dirs.sort()

        try:
            self.infer_save = self.args.infer_save
        except:
            self.infer_save = None

        try:
            if self.args.gui_show == 1:
                self.gui_show = True
                plt.figure()
            else:
                self.gui_show = False
        except:
            self.gui_show = False

        ### Results drawing
        self.font = 2
        self.txt_scale = 1
        self.colors = [(0,0,255), (255,0,255), (255,0,0)]

        ### Load object detector
        self.OD_model = init_detector(self.args.detect_config,
                                      self.args.detect_weights,
                                      device=self.args.device)


        ### Load Tracker
        self.Tracker = Sort()

        ### Load FOL and AP models
        self.FOL_model = FOL(self.args).to(self.args.device)
        self.FOL_model.load_state_dict(torch.load(self.args.best_fol_model))
        self.FOL_model.eval()

        self.best_ap_model_path = os.path.join(self.args.checkpoint,
                                               self.args.trial_name,
                                               self.args.best_harmonic_model)
        self.AP_model = AP(self.args).to(self.args.device)
        self.AP_model.load_state_dict(torch.load(self.best_ap_model_path))
        self.AP_model.eval()

        self.frame_num = 0
        self.frame = None
        self.detected = None
        self.tracked = {}
        self.hidden = {}
        self.preds = {}
        self.cnt = 0
        self.to_show = None

        try:
            self.num_memory = self.args.mem_limit
        except:
            self.num_memory = 10

        self.stop = False

        self.thread = Thread(target=self.inferLoop())
        self.thread.start()


    def drawResult(self, show=True, save_folder=None):
        img = self.frame.copy()
        if self.detected is not None:
            for det in self.detected:
                x1, y1, x2, y2 = det[:4].astype(int)
                conf = det[-1] * 100
                img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
                txt = 'D: %.1f' % conf
                cv2.putText(img, txt, (x2-200, y2+40), self.font, self.txt_scale, (0,255,0))

        rank = []
        if len(self.preds.keys()) > 0:
            for ID in self.preds.keys():
                p = np.array(self.preds[ID])
                if len(p.shape) > 1:
                    match = np.where(p[:,0] == self.frame_num)[0]
                    if len(match) > 0:
                        score = p[match[0]][1] * 100
                    else:
                        score = 0
                else:
                    if p[0] == self.frame_num:
                        score = p[1] * 100
                rank.append([ID, score])
            rank = np.array(rank)
        if len(rank) > 0:
            idx = np.argsort(rank[:,1])
            rank = rank[idx][::-1]

        if len(self.tracked.keys()) > 0:
            for ID in self.tracked.keys():
                if len(rank) > 0:
                    r = np.where(rank[:,0] == ID)[0]
                    if len(r) > 0:
                        if r[0] < 3:
                            color = self.colors[r[0]]
                            score = rank[r[0]][1]
                        else:
                            color = (0,255,0)
                            score = rank[r[0]][1]
                    else:
                        score = 0
                        color = (0,255,255)
                else:
                    score = 0
                    color = (0,255,255)

                tracked = np.array(self.tracked[ID])
                match = np.where(tracked[:,0] == self.frame_num)[0]
                if len(match) > 0:
                    bbox = tracked[match[0]][1:]
                    x1, y1, x2, y2 = bbox.astype(int)
                    cx = int(np.abs(x2+x1)/2)
                    cy = int(np.abs(y2+y1)/2)
                    img = cv2.circle(img, (cx, cy), 10, color, -1)
                    txt = 'ID: %d' % ID
                    cv2.putText(img, txt, (cx-100,cy-60), self.font, self.txt_scale, color)
                    txt = 'A: %.1f' % score
                    cv2.putText(img, txt, (cx-100, cy-30), self.font, self.txt_scale, color)
        self.to_show = img
        if show:
            #plt.figure()
            #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.show()
            cv2.imshow("Results", img.astype(np.uint8))
            k = cv2.waitKey(10)
            if k == 27:
                cv2.destroyAllWindows()
                self.stop = True
            #print(img)
        if save_folder is not None:
            name = os.path.join(save_folder, '%06d.png' % self.frame_num)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            cv2.imwrite(name, img)


    def getCurretFrame(self):

        if not self.img_folder is None:
            self.frame = cv2.imread(self.img_dirs[self.cnt])
            #print(self.frame.shape)
            self.cnt += 1
            if self.cnt >= len(self.img_dirs)-1:
                self.cnt = 0
                if self.args.repeat == 0:
                    self.stop = True
        else:
            self.frame = self.streamer.img

    def detection(self):
        self.detected = None
        detected = inference_detector(self.OD_model, self.frame)
        result = None
        if 'mask_rcnn' in self.args.detect_config:
            res = detected[0]
        else:
            res = detected

        targets = None
        for tid in self.det_targets:
            temp = res[tid]
            if targets is None:
                targets = temp
            else:
                targets = np.vstack([targets, temp])

        idxs = np.where(targets[:,4] >= self.args.detect_thresh)[0]

        if len(idxs) > 0:
            for idx in idxs:
                x1, y1, x2, y2, conf = targets[idx]
                #temp = np.array([x1,y1,x2,y2,conf])
                temp = [x1, y1, x2, y2, conf]
                if result is None:
                    result = [temp]
                else:
                    #result = np.vstack([result, temp])
                    result.append(temp)

            self.detected = np.array(result)

    def tracking(self):
        dets = self.detected
        if self.cnt == 0:
            self.Tracker = Sort()

        if dets is not None:
            res = self.Tracker.update(dets)

            if len(res) > 0:
                for d in res:
                    [x1, y1, x2, y2, ID] = d
                    ID = int(ID)
                    x1, y1 = xy_limit(x1, y1)
                    x2, y2 = xy_limit(x2, y2)

                    if not ID in self.tracked.keys():
                        self.tracked[ID] = [[self.frame_num, x1, y1, x2, y2]]
                    else:
                        self.tracked[ID].append([self.frame_num, x1, y1, x2, y2])
                        if len(self.tracked[ID]) > self.num_memory:
                            self.tracked[ID].pop(0)

            else:
                for ID in self.tracked.keys():
                    self.tracked[ID].append(self.tracked[ID][-1])
                    if len(self.tracked[ID]) > self.num_memory:
                        self.tracked[ID].pop(0)

        else:
            for ID in self.tracked.keys():
                self.tracked[ID].append(self.tracked[ID][-1])
                if len(self.tracked[ID]) > self.num_memory:
                    self.tracked[ID].pop(0)


    def predOnBatch(self):
        num_id = len(self.tracked.keys())
        length = np.ceil(num_id / self.batch_infer).astype(int)

        list_id = list(self.tracked.keys())
        id_split = []
        for i in range(length):
            if length == 1:
                id_split.append(list_id)
            else:
                if i + 1 == length:
                    id_split.append(list_id[i*self.batch_infer:])
                else:
                    id_split.append(list_id[i*self.batch_infer:(i+1)*self.batch_infer])

        for ids in id_split:
            bboxes = torch.zeros(len(ids), 4).to(self.args.device)
            dms = torch.zeros(len(ids), 4).to(self.args.device)
            box_hs = torch.zeros(len(ids), self.args.box_enc_size).to(self.args.device)
            dm_hs = torch.zeros(len(ids), self.args.dm_enc_size).to(self.args.device)

            for i in range(len(ids)):
                ID = ids[i]
                bbox, dm, box_h, dm_h = self.getHidden(ID, outs=True)
                bboxes[i] = bbox
                dms[i] = dm
                box_hs[i] = box_h
                dm_hs[i] = dm_h

            preds, box_hs, dm_hs, dec_hs = self.FOL_model.predict(bboxes, dms,
                                                                  box_hs,
                                                                  dm_hs)

            dec_hs_in = torch.unsqueeze(dec_hs, 1)
            scores = self.AP_model.forward(dec_hs_in)
            del dec_hs_in

            for i in range(len(ids)):
                ID = ids[i]

                if not ID in self.preds.keys():
                    self.preds[ID] = [[self.frame_num, scores.detach().cpu().numpy()[i][0][0]]]
                else:
                    self.preds[ID].append([self.frame_num, scores.detach().cpu().numpy()[i][0][0]])

                self.hidden[ID] = {}
                self.hidden[ID]['bbox'] = box_hs[i]
                self.hidden[ID]['dm'] = dm_hs[i]
                self.hidden[ID]['decoder'] = dec_hs[i]


    def getHidden(self, ID, outs=False):
        tracked = self.tracked[ID]

        # xxyy
        bbox = np.array(tracked[-1])
        # [frame_num, x1, y1, x2, y2] -> [x1, y1, x2, y2]
        bbox = bbox[1:]
        # cx, cy, w, h and normalization
        x1, y1, x2, y2 = bbox
        bbox = xy2cxcy(x1, y1, x2, y2, norm=True, args=self.args, ret_np=True)

        if len(tracked) == 1:
            dm = np.array([0, 0, 0, 0])
        elif len(tracked) > 1:
            pre_bbox = np.array(tracked[-2])
            pre_bbox = pre_bbox[1:]
            x1, y1, x2, y2 = pre_bbox
            pre_bbox = xy2cxcy(x1, y1, x2, y2, norm=True, args=self.args, ret_np=True)

            for_dm = np.vstack([pre_bbox, bbox])
            dm = np.array(delta_bbox(for_dm)[-1])

        # Hidden state initializing
        if ID in self.hidden.keys() and self.cnt != 0:
            box_h = self.hidden[ID]['bbox']
            dm_h = self.hidden[ID]['dm']

        else:
            # Initial hidden state : zeros
            box_h = torch.zeros(1, self.args.box_enc_size).to(self.args.device)
            dm_h = torch.zeros(1, self.args.dm_enc_size).to(self.args.device)

        bbox = np.expand_dims(bbox, axis=0)
        dm = np.expand_dims(dm, axis=0)

        bbox = torch.Tensor(bbox).to(self.args.device)
        dm = torch.Tensor(dm).to(self.args.device)

        if outs:
            return bbox, dm, box_h, dm_h

        else:
            pred, box_h, dm_h, dec_h = self.FOL_model.predict(bbox, dm,
                                                              box_h,
                                                              dm_h)

            self.hidden[ID] = {}
            self.hidden[ID]['bbox'] = box_h
            self.hidden[ID]['dm'] = dm_h
            self.hidden[ID]['decoder'] = dec_h



    def prediction(self, ID):
        # 1,10,1024
        dec_h = self.hidden[ID]['decoder']
        # 1,1,10,1024
        dec_h = torch.unsqueeze(dec_h, 0)
        score = self.AP_model.forward(dec_h)[0]
        if not ID in self.preds.keys():
            self.preds[ID] = [[self.frame_num, score.detach().cpu().numpy()[0][0]]]
        else:
            self.preds[ID].append([self.frame_num, score.detach().cpu().numpy()[0][0]])
        #print(self.frame_num, ID, score.cpu())

    def inferLoop(self):
        while self.stop == False:
            if self.cnt == 0:
                self.frame_num = 0
                self.frame = None
                self.detected = None
                self.tracked = {}
                self.hidden = {}
                self.preds = {}

            pre =time.time()
            self.getCurretFrame()
            self.detection()
            self.tracking()
            if self.args.batch_infer > 1:
                self.predOnBatch()
            else:
                for ID in self.tracked.keys():
                    self.getHidden(ID)
                    self.prediction(ID)
            self.drawResult(show=self.gui_show, save_folder=self.infer_save)
            if len(self.tracked.keys()) > 0:
                intv = time.time() - pre
                fps = 1.0/intv
                print("FPS: %.1f" % fps, ', Num IDs: ', len(self.tracked.keys()))

            self.frame_num += 1

if __name__ == '__main__':
    #os.system("python lib/rtsp_cv2.py")
    args = parse_args()
    run = Inference(args)

