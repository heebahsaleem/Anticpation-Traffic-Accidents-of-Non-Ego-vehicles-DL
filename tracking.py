from .sort import *
from PIL import Image, ImageDraw, ImageFont
import pickle
import os
from glob import glob
import numpy as np
import argparse
import pandas as pd


def xy_limit(x1, y1):
    if x1 < 0:
        x1 = 0
    if x1 > 1280:
        x1 = 1280
    if y1 < 0:
        y1 = 0
    if y1 > 720:
        y1 = 720
    return x1, y1


def xy2cxcy(x1, y1, x2, y2, norm=False, args=None, ret_np=False):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    if norm and args is not None:
        cx /= args.W
        cy /= args.H
        w /= args.W
        h /= args.H
    if ret_np:
        return np.array([cx, cy, w, h])
    else:
        return cx, cy, w, h


def vis_sort_result(image_path, save_path, frame, trackers):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    # fnt =  ImageFont.truetype('Arial' ,30)
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 30)

    for d in trackers:
        [x1, y1, x2, y2, id] = d
        x1, y1 = xy_limit(x1, y1)
        x2, y2 = xy_limit(x2, y2)
        draw.rectangle(((x1, y1), (x2, y2)), outline=(255, 255, 0), width=2)
        draw.text((x2, y1), str(int(id)), font=fnt, fill=(0, 0, 255))
    image.save(save_path)


def ObjectTracking(args):

    det_root = os.path.join(args.data_root, args.detected)
    frame_root = os.path.join(args.data_root, args.frames)
    save_root = os.path.join(args.data_root, args.tracked)


    existing = glob(os.path.join(save_root, '**/*.pkl'), recursive=True)
    if args.overwrite == 'FALSE' and len(existing) > 0:
        print('Files already exist in save root (Overwrite FALSE) {}'.format(save_root))
        return 0


    sub_folder = glob(os.path.join(det_root, '*/'))
    sub_flag = False
    if len(sub_folder) != 0:
        sub_flag = True
        dataset_list = []
        for sub in sub_folder:
            txts = glob(os.path.join(sub, '*.txt'))
            dataset_list.extend(txts)
    else:
        dataset_list = glob(os.path.join(det_root, '*.txt'))
        dataset_list.sort()

    cnt = 0

    label = pd.read_excel(os.path.join(args.data_root, args.annotation), engine='openpyxl')
    vid_labs = np.array(label)[:, 0]

    for dataset in dataset_list:
        cnt += 1
        vid_name = os.path.split(dataset)[-1].replace('.txt', '')

        # img_list = os.listdir(frame_folder + vid_name +"/images/")
        if sub_flag:
            sub_f = dataset.split('/')[-2]
            img_folder = os.path.join(frame_root, sub_f, vid_name)
        else:
            img_folder = os.path.join(frame_root, vid_name)
        img_list = glob(os.path.join(img_folder, '*.{}'.format(args.img_format)))
        img_list.sort()

        # Accident frame
        af = None
        # Collided object
        cid = None
        vi = np.where(vid_labs == vid_name)[0]
        if len(vi) > 0:
            af = int(np.nan_to_num(label.iloc[vi[0]]['acc_frame_id']))
            #cid = int(np.nan_to_num(label.iloc[vi[0]]['acc_obj_id']))
            if af > 0:
                # Do not use post-collision frames
                img_list = img_list[:af + 1]

        if sub_flag:
            save_folder = os.path.join(save_root, sub_f, vid_name)
        else:
            save_folder = os.path.join(save_root, vid_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        vis_folder = os.path.join(save_folder, "visualization")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        # create instance of SORT
        mot_tracker = Sort()

        total_frames = 0
        object_id_list = []
        listall = []
        frameid_list = []
        # get detections information
        seq_dets = np.loadtxt(dataset, delimiter=",")
        if len(seq_dets) == 0:
            print(dataset, " passed")
            continue
        if seq_dets.shape == (10,):
            seq_dets = seq_dets.reshape((1, 10))

        for frame in range(len(img_list)):
            image_path = img_list[frame]
            save_img_path = os.path.join(vis_folder, "{:06}.jpg".format(frame))

            if seq_dets[seq_dets[:, 0] == frame] == []:
                # Nothing Detection
                print("Detecting nothing")

            else:
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                trackers = mot_tracker.update(dets)
                for d in trackers:
                    [x1, y1, x2, y2, id] = d
                    id = int(id)
                    x1, y1 = xy_limit(x1, y1)
                    x2, y2 = xy_limit(x2, y2)
                    cx, cy, w, h = xy2cxcy(x1, y1, x2, y2)

                    if not id in object_id_list:
                        # this tracking id is first time.
                        object_id_list.append(id)
                        frameid_list.append([frame])
                        listall.append([[cx, cy, w, h]])

                    else:
                        index = object_id_list.index(id)
                        temp = listall.pop(index)
                        temp.insert(len(temp), [cx, cy, w, h])
                        listall.insert(index, temp)

                        temp2 = frameid_list.pop(object_id_list.index(id))
                        temp2.append(frame)
                        frameid_list.insert(object_id_list.index(id), temp2)

                vis_sort_result(image_path, save_img_path, frame, trackers)

        mot_tracker.reset_count()

        print(cnt, img_folder, len(img_list), len(object_id_list))

        for k in range(len(object_id_list)):
            # result_file = result_path + "/test_"+str(nth_dataset)+"_obj_"+str(object_id_list[k])+".pkl"
            result_file = os.path.join(save_folder, "{}_{}.pkl".format(vid_name, object_id_list[k]))
            result = {'bbox': np.asarray(listall[k]), 'frame_id': np.asarray(frameid_list[k])}

            # Normalize
            bbox = result['bbox']
            for i in range(len(bbox)):
                temp = bbox[i]
                temp[0] = temp[0] / float(args.W)
                temp[1] = temp[1] / float(args.H)
                temp[2] = temp[2] / float(args.W)
                temp[3] = temp[3] / float(args.H)

                bbox[i] = temp
            result['bbox'] = bbox

            """if object_id_list[k] == cid:
                result['accident_frame_id'] = float(cid)
            else:
                result['accident_frame_id'] = 1000.0"""


            with open(result_file, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)