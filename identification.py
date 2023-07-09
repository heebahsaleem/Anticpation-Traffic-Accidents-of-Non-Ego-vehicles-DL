import json
import os
import numpy as np
from glob import glob
import pickle
import cv2
import matplotlib.pyplot as plt
import shutil
import pandas as pd


def cxcy(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return np.array([cx, cy, w, h])


def xxyy(bbox):
    cx, cy, w, h = bbox

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    return np.array([x1, y1, x2, y2])

def reverse(bbox, args):
    cx, cy, w, h = bbox
    cx = cx * args.W
    cy = cy * args.H
    w = w * args.W
    h = h * args.H

    bbox[0] = cx
    bbox[1] = cy
    bbox[2] = w
    bbox[3] = h

    return bbox



def calIOU(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def findCollided(args, vid_path, anno_root, iou=0.5, drawing=False,
                 dist_th=50,
                 show=False):
    vid_name = vid_path.split('/')[-2]
    pkls = glob(os.path.join(vid_path, '*.pkl'))
    #imgs = glob(os.path.join(frame_root, vid_name, '*.jpg'))
    #imgs = glob(os.path.join(vid_path.replace(args.tracked, args.frames), '*.{}'.format(args.img_format)))
    imgs = glob(os.path.join('/HDD/accident_anticipation/Data/DoTA_Ego_20220204/Frames/train',
                             '{}/*.{}'.format(vid_name, args.img_format)))
    imgs.sort()
    last_frame = len(imgs) - 1
    anno = json.load(open(os.path.join(anno_root, vid_name + '.json'), 'r'))
    ano_start = anno['anomaly_start']

    frame_ids = []
    col_bboxes = []
    for i in range(ano_start, len(imgs)):
        if len(anno['labels'][i]['objects']) == 0:
            continue
        frame_ids.append(i)
        bbox = anno['labels'][i]['objects'][0]['bbox']
        col_bboxes.append(bbox)
    col_bboxes = np.array(col_bboxes)
    frame_ids = np.array(frame_ids)

    det_bboxes = {}
    for i in range(len(pkls)):
        path = pkls[i]
        obj = int(os.path.split(path)[1].replace('.pkl', '').split('_')[-1])
        data = pickle.load(open(path, 'rb'))
        ids = data['frame_id']

        temp = {}

        for j in range(len(frame_ids)):
            idx = np.where(ids == frame_ids[j])[0]
            if len(idx) == 0:
                continue
            temp[ids[idx[0]]] = reverse(data['bbox'][idx[0]], args)

        if len(temp.keys()) > 0:
            det_bboxes[obj] = temp

    # From DeepSORT cx, cy, w, h
    # From anno x1 , y1, x2, y2

    # Calculate dist
    Dist = {}
    for obj in det_bboxes.keys():
        fb = det_bboxes[obj]

        for k in fb.keys():
            kid = np.where(frame_ids == k)[0][0]
            ref = cxcy(col_bboxes[kid])
            obs = fb[k]

            dist = np.linalg.norm(ref[:2] - obs[:2])

            if obj in Dist.keys():
                Dist[obj].append(dist)
            else:
                Dist[obj] = [dist]

    # Calculate IOU
    IOU = {}
    for obj in det_bboxes.keys():
        fb = det_bboxes[obj]

        for k in fb.keys():
            kid = np.where(frame_ids == k)[0][0]
            ref = col_bboxes[kid]
            obs = xxyy(fb[k])

            cal_IOU = calIOU(ref, obs)

            if obj in IOU.keys():
                IOU[obj].append(cal_IOU)
            else:
                IOU[obj] = [cal_IOU]

    # Mean Dist for obj
    mDist = []
    DOBJ = []
    for obj in Dist.keys():
        mean = np.mean(np.array(Dist[obj]))
        DOBJ.append(obj)
        mDist.append(mean)
    mDist = np.array(mDist)

    # Mean IOU for obj
    mIOU = []
    OBJ = []
    for obj in IOU.keys():
        mean = np.mean(np.unique(np.array(IOU[obj])))
        # mIOU[obj] = mean
        OBJ.append(obj)
        mIOU.append(mean)

    mIOU = np.array(mIOU)


    if len(mDist) > 0 and len(mIOU) > 0:
        flag = True
        tDist = mDist
        while flag:
            if len(mDist) == 0:
                flag = False
                continue
            ii = np.argmin(tDist)
            idist = tDist[ii]
            odist = DOBJ[ii]
            iiou = np.where(np.array(OBJ) == odist)[0][0]
            oIOU = mIOU[iiou]
            if oIOU >= iou and idist <= dist_th:
                Est = odist
                EDI = [mDist[ii], oIOU]
                flag = False
            else:
                tDist = np.delete(tDist, ii)
                if len(tDist) == 0:
                    flag = False
                    return [vid_name, [Dist, IOU]]


    else:
        return [vid_name, [Dist, IOU]]


    # Draw
    if drawing:
        Ebbox = det_bboxes[Est]
        last = list(Ebbox.keys())[-1]

        Edraw = xxyy(Ebbox[last]).astype(int)
        l_idx = np.where(frame_ids == last)[0][0]
        Rdraw = col_bboxes[l_idx].astype(int)

        img = cv2.imread(imgs[last])

        img = cv2.rectangle(img, Edraw[:2], Edraw[2:], (0, 0, 255), 3)
        img = cv2.rectangle(img, Rdraw[:2], Rdraw[2:], (0, 255, 0), 5)

        if show:
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()

    return [vid_name, Est, EDI, img]



def bad_collect(vid_name, save_to, track_root, num=20):
    imgs = glob(os.path.join(track_root, '**/{}/*.jpg'.format(vid_name)), recursive=True)
    imgs.sort()
    imgs = imgs[-num:]

    for path in imgs:
        name = os.path.split(path)[1]
        folder = os.path.join(save_to, vid_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        full = os.path.join(folder, name)
        shutil.copy(path, full)


def getColObj(args):

    anno_root = args.dota_original_annotations
    tracked = os.path.join(args.data_root, args.tracked)
    frames = os.path.join(args.data_root, args.frames)
    af_anno = pd.read_excel(os.path.join(args.data_root, args.annotation), engine='openpyxl')

    vids = glob(os.path.join(tracked, '*/'))
    sub_flag = False
    temp = []
    if len(vids) <= 3:
        sub_flag = True
        for sub in vids:
            t_vid = glob(os.path.join(sub, '*/'))
            temp.extend(t_vid)
        vids = temp
    vids.sort()

    vid_anno = np.array(af_anno)[:,0]
    temp = []
    for path in vids:
        vn = path.split('/')[-2]
        if vn in vid_anno:
            temp.append(path)
    vids = temp

    done = []
    for i in range(len(vids)):
        res = findCollided(args, vids[i], anno_root,
                           drawing=True,
                           show=False)
        done.append(res)

    good = []
    bad = []

    for res in done:
        if len(res) >= 3:
            good.append(res)
        else:
            bad.append(res)

    print("Identified videos: {}, Missed videos: {}".format(len(good), len(bad)))
    all_list = np.array(good)
    af_anno_vids = np.array(af_anno)[:,0]

    new_anno = []
    header = ['Name', 'acc_frame_id', 'acc_obj_id', 'detection_error', 'acc_class_error']

    for i in range(len(all_list)):
        v = all_list[i][0]
        idx = np.where(af_anno_vids == v)[0]

        if len(idx) > 0:
            line = af_anno.iloc[idx[0]]
            name = line[0]
            af = line[1]
            obj = all_list[i][1]
            de = 0
            ace = np.nan_to_num(line[4])

            new_anno.append([name, af, obj, de, ace])

    new_anno = pd.DataFrame(new_anno, columns=header)
    new_anno.to_excel(os.path.join(args.data_root, args.updated_anno), engine='openpyxl', index=False)


def assigning(args):
    new_anno = pd.read_excel(os.path.join(args.data_root, args.updated_anno), engine='openpyxl')
    anno_names = np.array(new_anno)[:,0]
    all_tracked = glob(os.path.join(args.data_root, args.tracked, '**/*.pkl'), recursive=True)
    all_tracked.sort()


    for_normals = {}
    for_acc = []
    for pkl in all_tracked:
        name = os.path.split(pkl)[1].replace('.pkl', '')
        temp = name.split('_')
        vid_name = ''
        for i in range(len(temp)-1):
            vid_name += temp[i] + '_'
        vid_name = vid_name[:-1]
        pkl_id = int(temp[-1])
        idx = np.where(anno_names == vid_name)[0]

        for_normals[vid_name] = []
        if len(idx) == 0:
            data = pickle.load(open(pkl, 'rb'))
            data['accident_frame_id'] = 1000.0
            with open(pkl, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for_normals[vid_name].append(pkl)
            if len(args.tracked_avails) > 0:
                new_path = pkl.replace(args.tracked, args.tracked_avails)
                new_folder = os.path.split(new_path)[0]
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                with open(new_path, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        else:
            line = new_anno.iloc[idx[0]]
            oid = int(np.nan_to_num(line['acc_obj_id']))
            af = np.nan_to_num(line['acc_frame_id'])
            if not vid_name in for_acc:
                for_acc.append(vid_name)
            if pkl_id == oid:
                data = pickle.load(open(pkl, 'rb'))
                data['accident_frame_id'] = float(af)
                with open(pkl, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                if len(args.tracked_avails) > 0:
                    new_path = pkl.replace(args.tracked, args.tracked_avails)
                    new_folder = os.path.split(new_path)[0]
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    with open(new_path, 'wb') as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                data = pickle.load(open(pkl, 'rb'))
                data['accident_frame_id'] = 1000.0
                with open(pkl, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
                if len(args.tracked_avails) > 0:
                    new_path = pkl.replace(args.tracked, args.tracked_avails)
                    new_folder = os.path.split(new_path)[0]
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    with open(new_path, 'wb') as f:
                        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    if len(args.tracked_avails) > 0:
        num = len(for_acc)
        norms = np.array(list(for_normals.keys()))
        select = np.random.choice(norms, num)

        for key in select:
            pkls = for_normals[key]
            for path in pkls:
                new_path = path.replace(args.tracked, args.tracked_avails)
                new_folder = os.path.split(new_path)[0]
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                shutil.copy(path, new_path)

    print("Available - Acc: {}, Norm: {}, Total_Norm: {}".format(num, len(select), len(for_normals.keys())))