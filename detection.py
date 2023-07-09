from mmdet.apis import init_detector, inference_detector
import cv2
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def ObjectDetection(args):

    frame_root = os.path.join(args.data_root, args.frames)
    save_root = os.path.join(args.data_root, args.detected)
    existing = glob(os.path.join(save_root, '**/*.txt'), recursive=True)
    if args.overwrite == 'FALSE' and len(existing) > 0:
        print('Files already exist in save root (Overwrite FALSE) {}'.format(save_root))
        return 0
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    videos = glob(os.path.join(frame_root, '*/'))
    videos.sort()
    temp = []
    sub_flag = False
    if len(videos) <= 3:
        for sub in videos:
            t_vids = glob(os.path.join(sub, '*/'))
            temp.extend(t_vids)
        videos = temp
        sub_flag = True

    if len(videos) == 0:
        print("Cannot find frame folders under ", frame_root)
        return 0

    model = init_detector(args.detect_config, args.detect_weights, device=args.device)
    times = []
    for vid_folder in videos:
        img_dirs = glob(os.path.join(vid_folder, '*.{}'.format(args.img_format)))
        img_dirs.sort()
        vid_name = vid_folder.split('/')[-2]
        print("Detect {}".format(vid_name))
        if sub_flag:
            save_name = os.path.join(save_root, vid_folder.split('/')[-3], vid_name+'.txt')
        else:
            save_name = os.path.join(save_root, vid_name+'.txt')


        """for img in img_dirs:
            out = inference_detector(model, img)
            results.append(out)"""
        outs = []
        pre = time.time()
        for img in img_dirs:
            res = inference_detector(model, img)
            outs.append(res)
        dt = time.time() - pre
        times.append([dt, len(outs), dt/len(outs)])


        # MSCOCO  person(1), bicycle(2), car(3), motorcycle(4), bus(6), truck(8)
        results = None
        target_id = [1, 2, 3, 4, 6, 8]
        for i in range(len(outs)):
            n_frame = i
            if 'mask_rcnn' in args.detect_config:
                res = outs[i][0]
            else:
                res = outs[i]

            targets = None
            for tid in target_id:
                temp = res[tid]
                if targets is None:
                    targets = temp
                else:
                    targets = np.vstack([targets, temp])

            idxs = np.where(targets[:,4] >= args.detect_thresh)[0]
            img = cv2.imread(img_dirs[i])

            if len(idxs) > 0:
                for idx in idxs:
                    x1, y1, x2, y2, conf = targets[idx]
                    temp = np.array([n_frame, -1, x1, y1, x2, y2, conf, -1, -1, -1])
                    if results is None:
                        results = temp
                    else:
                        results = np.vstack([results, temp])

        if results is not None:
            if len(results) > 0:
                save_folder = os.path.split(save_name)[0]
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                np.savetxt(save_name, results, fmt='%.3f', delimiter=',')
            else:
                print('Detection Skip ', vid_name)
        else:
            print('Detection Skip ', vid_name)

    times = np.array(times)
    T = np.sum(times[:,0])
    N = np.sum(times[:,1])
    M = T/N

    time_data = {'Mean': M, 'All': times}
    name = os.path.join(save_folder, 'TIME.pkl')
    with open(name, 'wb') as f:
        pickle.dump(time_data, f, pickle.HIGHEST_PROTOCOL)

    del model
