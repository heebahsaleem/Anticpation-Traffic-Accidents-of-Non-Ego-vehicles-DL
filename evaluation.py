# AP :  Accident Prediction
# Evaluation Accident Anticiaption
# Input : pickle file (information - bounding box, differential motion, frame id

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import pandas as pd
from glob import glob
import torch

from .models import FOL
from .models import AP
from .utils import calculate_AP_ATTC, plot_PR_curve, plot_RT_curve

def precondition(vid_name, label):
    labeling_array = label
    index = np.where(labeling_array[:, 0] == vid_name)[0][0]
    frame_end = labeling_array[index, 1]
    return int(frame_end)


def cal_single_obj_risk_score(ap_result, frame_end):
    if len(ap_result) == 0:
        risk_score = np.zeros((frame_end + 1,))
    else:
        risk_score = np.zeros((frame_end + 1,))
        obj_frame_id = ap_result[:, 1]
        obj_frame_id = obj_frame_id.astype(np.int64)
        obj_risk_score = ap_result[:, 0]
        for index, id in enumerate(obj_frame_id):
            risk_score[id] = obj_risk_score[index]
    return risk_score


def cal_overall_risk_score(objs_ap_results, frame_end):
    if len(objs_ap_results) == 0:
        overall_risk_score = np.zeros((frame_end + 1,))
    else:
        num_objs = len(objs_ap_results)
        objs_risk_scores = np.zeros((num_objs, frame_end + 1))
        # print("objs_risk_scores:", objs_risk_scores.shape)

        for k in range(num_objs):
            obj_ap_result = objs_ap_results[k]
            obj_frame_id = obj_ap_result[:, 1]
            obj_frame_id = obj_frame_id.astype(np.int64)
            obj_risk_score = obj_ap_result[:, 0]

            for index, id in enumerate(obj_frame_id):
                objs_risk_scores[k, id] = obj_risk_score[index]

        overall_risk_score = np.max(objs_risk_scores, axis=0)  # select max risk scores among objects
    return overall_risk_score


def vis_obj_risk_score_graph(ap_result, frame_end, vis_save_for_vid, pkl_name):
    # visualization object's predicition risk scores and ground truth scores graph.
    # save the graph.
    risk_score = np.zeros((frame_end + 1,))
    obj_frame_id = ap_result[:, 1]
    obj_frame_id = obj_frame_id.astype(np.int64)
    obj_risk_score = ap_result[:, 0]
    for index, id in enumerate(obj_frame_id):
        risk_score[id] = obj_risk_score[index]

    time = np.array(range(frame_end + 1))
    plot_name = os.path.join(vis_save_for_vid, pkl_name + '.png')

    plt.clf()
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, frame_end)
    plt.ylabel('Risk Score')
    plt.xlabel('Time')
    plt.plot(time, risk_score, 'r-')
    plt.savefig(plot_name)


def vis_risk_score_graph(overall_risk_score, frame_end, vis_save_for_vid, plot_title):
    # visualization predicition risk scores and ground truth scores graph.
    # save the graph.
    plot_name = os.path.join(vis_save_for_vid, 'Videowise_risk_score.png')
    time = np.array(range(frame_end + 1))
    plt.clf()
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, frame_end)
    plt.ylabel('Risk Score')
    plt.xlabel('Frame')
    plt.title(plot_title, loc='center')
    plt.plot(time, overall_risk_score, 'r-')
    plt.savefig(plot_name)


def accident_prediction(args, pkl_data, fol_model, ap_model, acc_frame_id):
    # use AP model.
    # risk scores.
    device = args.device
    pred_risk_scores = []  # list
    frame_id = pkl_data['frame_id']
    limit_index = np.where(frame_id <= acc_frame_id)[0][-1]

    frame_id = frame_id[0:limit_index + 1]
    bbox = pkl_data['bbox'][0:limit_index + 1]
    dm = delta_bbox(bbox)

    total_length = frame_id.shape[0]
    box_h = torch.zeros(1, args.box_enc_size).to(device)
    dm_h = torch.zeros(1, args.dm_enc_size).to(device)

    for i in range(total_length):
        input_box = torch.from_numpy(bbox[i]).float().to(device)
        input_dm = torch.from_numpy(dm[i]).float().to(device)
        input_box = input_box.view(1, input_box.size()[0])  # [1,4]
        input_dm = input_dm.view(1, input_dm.size()[0])  # [1,4]

        risk_score, box_h, dm_h = ap_model.predict(input_box, input_dm, box_h, dm_h, fol_model)
        risk_score = risk_score.cpu().detach().numpy()  # tensor to numpy array [1,1]
        risk_score = risk_score[0][0]
        pred_risk_scores.append(risk_score)

    pred_risk_scores = np.array(pred_risk_scores)
    return pred_risk_scores, frame_id


def delta_bbox(input_bbox):
    l = input_bbox.shape[0]

    result_bbox = np.zeros((l, 4))

    for j in range(1, l):
        cx1, cy1, w1, h1 = input_bbox[j - 1]
        cx2, cy2, w2, h2 = input_bbox[j]

        x1_1 = cx1 - (w1 / 2)
        x1_2 = cx1 + (w1 / 2)
        y1_1 = cy1 - (h1 / 2)
        y1_2 = cy1 + (h1 / 2)

        x2_1 = cx2 - (w2 / 2)
        x2_2 = cx2 + (w2 / 2)
        y2_1 = cy2 - (h2 / 2)
        y2_2 = cy2 + (h2 / 2)

        if j == 1:
            result_bbox[0] = [0, 0, 0, 0]
            result_bbox[1] = [x2_1 - x1_1, x2_2 - x1_2, y2_1 - y1_1, y2_2 - y1_2]
        else:
            result_bbox[j] = [x2_1 - x1_1, x2_2 - x1_2, y2_1 - y1_1, y2_2 - y1_2]
    if (l == 1):
        result_bbox[0] = [0, 0, 0, 0]

    return result_bbox

def evaluate(args):

    device = args.device
    fol_model = FOL(args).to(device)
    fol_model.load_state_dict(torch.load(args.best_fol_model))

    best_ap_model_path = os.path.join(args.checkpoint, args.trial_name, args.best_harmonic_model)
    AP_model = AP(args).to(device)
    AP_model.load_state_dict(torch.load(best_ap_model_path))
    AP_model.eval()

    test_data_inTracked_path = os.path.join(args.data_root, args.tracked_avails, args.test_root)
    save_path = os.path.join(args.data_root, args.evaluation_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vis_save_path = os.path.join(save_path, "visualization/")
    if not os.path.exists(vis_save_path):
        os.makedirs(vis_save_path)


    anno_file = os.path.join(args.data_root, args.annotation)
    if os.path.exists(os.path.join(args.data_root, args.updated_anno)):
        anno_file = os.path.join(args.data_root, args.updated_anno)

    if len(args.test_anno) > 0:
        anno_file = os.path.join(args.data_root, args.test_anno)

    df = pd.read_excel(anno_file, engine='openpyxl')
    labeling_array = np.array(df)
    positive_data_array = labeling_array[
                              np.where(np.logical_and(labeling_array[:, 4] != 'o',
                                                      labeling_array[:, 3] != 'o'))][:, 0]
    all_obj_risk_score = {}
    all_dataset_risk_score = {}

    #test_dataset_list = os.listdir(test_data_inTracked_path)
    test_vid_list = glob(os.path.join(test_data_inTracked_path, '*/'))
    cnt = 0
    for vid_path in test_vid_list:
        vid_name = vid_path.split('/')[-2]
        print('Progress: {}/{}'.format(cnt+1,len(test_vid_list)), ' Evaluating on ', vid_name)
        cnt += 1
        vis_save_for_vid = os.path.join(vis_save_path, vid_name)
        if not os.path.exists(vis_save_for_vid):
            os.makedirs((vis_save_for_vid))

        if vid_name in positive_data_array:
            frame_end = precondition(vid_name, labeling_array)
            plot_title= "Accident Video"
        else:
            plot_title = "Non-accident Video"
            if "Normal" in vid_name:
                dota_anno_file = os.path.join(args.dota_original_annotations,
                                              vid_name.replace('Normal_', '') + '.json')
            else:
                dota_anno_file = os.path.join(args.dota_original_annotations,
                                               vid_name + '.json')
            with open(dota_anno_file) as json_file:
                json_data = json.load(json_file)
                frame_end = json_data["anomaly_start"] - 1
            if frame_end == 0:
                continue

        pkls = glob(os.path.join(test_data_inTracked_path, vid_name, '*.pkl'))

        objs_ap_results = []
        for pkl in pkls:
            name = os.path.split(pkl)[1].replace('.pkl', '')
            pkl_data = pickle.load(open(pkl, 'rb'))
            frame_id = pkl_data['frame_id']
            if frame_id[0] >= frame_end:
                continue

            pred_risk_scores, frame_id = accident_prediction(args, pkl_data, fol_model, AP_model, frame_end)
            pred_risk_scores = np.reshape(pred_risk_scores, (-1, 1))
            frame_id = np.reshape(frame_id, (-1, 1))

            ap_result = np.concatenate((pred_risk_scores, frame_id), axis=1)
            vis_obj_risk_score_graph(ap_result, frame_end, vis_save_for_vid, name)
            single_obj_risk_score = cal_single_obj_risk_score(ap_result, frame_end)

            all_obj_risk_score[name] = single_obj_risk_score
            objs_ap_results.append(ap_result)

        # calculate video-clip's overall prediction risk score
        overall_risk_score = cal_overall_risk_score(objs_ap_results, frame_end)
        vis_risk_score_graph(overall_risk_score, frame_end, vis_save_for_vid, plot_title)
        all_dataset_risk_score[vid_name] = overall_risk_score

    # save pickle file
    result_file = os.path.join(save_path, 'all_pred.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(all_dataset_risk_score, f, pickle.HIGHEST_PROTOCOL)

    obj_result_file = os.path.join(save_path, 'all_obj_pred.pkl')
    with open(obj_result_file, 'wb') as f:
        pickle.dump(all_obj_risk_score, f, pickle.HIGHEST_PROTOCOL)


    # Calculate AP, ATTC
    # Objectwise
    all_pred = all_obj_risk_score
    num_data = len(all_pred.keys())
    positive_data_obj_list = []

    for data_name in positive_data_array:
        positive_obj = labeling_array[np.where(labeling_array[:, 0] == data_name)][0][2]
        if isinstance(positive_obj, str):
            obj_list = positive_obj.split(",")
            for obj in obj_list:
                positive_data_obj_list.append(data_name + "_" + obj)
        else:
            positive_data_obj_list.append(data_name + "_" + str(positive_obj))

    all_labels = np.zeros(num_data)

    for i in range(len(all_labels)):
        if list(all_pred.keys())[i] in positive_data_obj_list:
            all_labels[i] = 1.0

    APs, Time, Precision, Recall = calculate_AP_ATTC(all_pred, all_labels, num_data, obj=True)
    PR_list = pd.DataFrame(np.transpose([Recall, Precision, Time]), columns=['Recall', 'Precision', 'Time'])
    PR_list.to_excel(os.path.join(save_path, 'Objectwise_PR_list.xlsx'), engine='openpyxl')
    plot_RT_curve(Time, Recall, os.path.join(save_path, 'Objectwise_RT_curve.png'))
    plot_PR_curve(Precision, Recall, os.path.join(save_path, 'Objectwise_PR_curve.png'))
    Time_not_nan = np.nan_to_num(Time, copy=True, nan=0.0, posinf=None, neginf=None)
    ATTC = np.mean(Time_not_nan) / args.FPS

    txt_line = 'Objectwise AP: {}%, ATTC: {} s\n'.format(APs*100, ATTC)

    #####################################
    # Videowise
    all_pred = all_dataset_risk_score
    num_data = len(all_pred.keys())
    all_labels = np.zeros(num_data)

    for i in range(len(all_labels)):
        if list(all_pred.keys())[i] in positive_data_array:
            all_labels[i] = 1.0

    APs, Time, Precision, Recall = calculate_AP_ATTC(all_pred, all_labels, num_data, obj=False)
    PR_list = pd.DataFrame(np.transpose([Recall, Precision, Time]), columns=['Recall', 'Precision', 'Time'])
    PR_list.to_excel(os.path.join(save_path, 'Videowise_PR_list.xlsx'), engine='openpyxl')
    plot_RT_curve(Time, Recall, os.path.join(save_path, 'Videowise_RT_curve.png'))
    plot_PR_curve(Precision, Recall, os.path.join(save_path, 'Videowise_PR_curve.png'))
    Time_not_nan = np.nan_to_num(Time, copy=True, nan=0.0, posinf=None, neginf=None)
    ATTC = np.mean(Time_not_nan) / args.FPS

    txt_line += 'Videowise AP: {}%, ATTC: {} s\n'.format(APs*100, ATTC)

    final = os.path.join(save_path, 'Final_results.txt')
    with open(final, 'w') as f:
        f.write(txt_line)
