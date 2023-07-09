import os
import numpy as np
from glob import glob
import pickle as pkl
import torch
from torch.utils import data
import matplotlib.pyplot as plt

class dataloader(data.Dataset):
    # NO padding
    def __init__(self, args, phase):
        '''
        Target : Ego-Involved accdient
        Traffic Accident datasets. EX) DoTA
        Contains bbox, flow and frame_id, accident_frame_id

        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        if len(self.args.tracked_avails) > 0:
            self.data_root = os.path.join(self.args.data_root, self.args.tracked_avails, phase)
        else:
            self.data_root = os.path.join(self.args.data_root, self.args.tracked, phase)
        self.sessions = glob(os.path.join(self.data_root, '**/*.pkl'), recursive=True)
        self.device = args.device
        self.all_inputs = []
        for session in self.sessions:
            # for each object in dataset, we split to several trainig samples
            data = pkl.load(open(session, 'rb'))
            input_bbox = data['bbox']
            input_flow = self.delta_bbox(input_bbox)  # data['bbox']
            # input_flow = data['expend']
            input_ego_motion = np.zeros((input_bbox.shape[0], 3), dtype=float)  # [yaw, x, z]

            acc_frame_id = data['accident_frame_id']  # when the accident began to occur

            # if session.split('\\')[-1].split('0')[0] == 'n':
            #    acc_frame_id = 1000
            # if session.split('\\')[-1].split('0')[0] == 'p':
            #    acc_frame_id = 40

            if acc_frame_id == 1000:
                # negative case object
                target_risk_score = np.array([0.])
            else:
                # positive case object
                target_risk_score = np.array([1.])

            self.all_inputs.append([input_bbox, input_flow, input_ego_motion, target_risk_score])

    def delta_bbox(self, input_bbox):

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

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, index): 
        input_bbox, input_flow, input_ego_motion, target_risk_score = self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(self.device)
        input_flow = torch.FloatTensor(input_flow).to(self.device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(self.device)
        target_risk_score = torch.FloatTensor(target_risk_score).to(self.device)

        return input_bbox, input_flow, input_ego_motion, target_risk_score, self.sessions[index].split('\\')[-1]


class dataloaderHidden(data.Dataset): #Non-Ego
    # load fol's output
    # ['hidden state'] , ['target_risk_score']
    def __init__(self, args, phase=None):
        '''
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        self.device = args.device
        if phase is not None:
            self.hidden_root = os.path.join(self.args.data_root, self.args.hiddens, phase)
            self.sessions = glob(os.path.join(self.hidden_root, '*.pkl'))
        else:
            self.sessions = glob(os.path.join(self.args.data_root, '**/*.pkl'), recursive=True)
        # self.max_length = 170
        self.sessions.sort()

        self.all_inputs = []
        for session in self.sessions:
            data = pkl.load(open(session, 'rb'))
            input_all_dec_h = data['hidden_state']
            target_risk_score = data['target_risk_score']
            length = input_all_dec_h.shape[0]
            except_pred = np.ones((length), dtype=float) #to prevent from shape erro, np.ones is used to fill it ones and match the size.

            self.all_inputs.append([input_all_dec_h, target_risk_score, except_pred])

    def __len__(self):
        return len(self.all_inputs)

    def __getitem__(self, index): #gives 1024, change to 2048.
        input_all_dec_h, target_risk_score, except_pred = self.all_inputs[index]
            input_all_dec_h = torch.FloatTensor(input_all_dec_h).to(self.device)
        target_risk_score = torch.FloatTensor(target_risk_score).to(self.device)
        except_pred = torch.FloatTensor(except_pred).to(self.device)

        return input_all_dec_h, target_risk_score, except_pred


def calculate_AP_ATTC(all_pred, all_labels, num_data, obj=True):
    temp_shape = 0
    all_pred_flatten = []
    clips_list = list(all_pred.keys())

    for i in range(num_data):
        key_value = list(all_pred.keys())[i]
        temp_shape += len(all_pred[key_value])
        all_pred_flatten.extend(all_pred[key_value])

    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0

    if obj:
        print(len(sorted(all_pred_flatten)))
        all_pred_flatten = np.round(all_pred_flatten, 4)
        all_pred_flatten = set(all_pred_flatten)
        print(len(sorted(all_pred_flatten)))

    for threshold in sorted(all_pred_flatten):
        if cnt % 1000 == 0:
            print(cnt)
        Tp = 0.0
        Tp_Fp = 0.0  # TP + FP
        Tp_Tn = 0.0
        frame_to_acc = 0.0
        counter = 0.0
        for i in range(len(clips_list)):
            tp = np.where(all_pred[clips_list[i]] * all_labels[i] >= threshold)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                frame_to_acc += len(all_pred[clips_list[i]]) - tp[0][0]
                counter = counter + 1
            Tp_Fp += float(len(np.where(all_pred[clips_list[i]] >= threshold)[0]) > 0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp / Tp_Fp
        if np.sum(all_labels) == 0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp / np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (frame_to_acc / counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]
    new_Recall = new_Recall[:-1]
    new_Precision = new_Precision[:-1]
    new_Time = new_Time[:-1]

    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    return AP, new_Time, new_Precision, new_Recall


def plot_PR_curve(new_Precision, new_Recall, save_full):
    plt.clf()
    plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(save_full)

def plot_RT_curve(new_Time, new_Recall, save_full):
    plt.clf()
    plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
    plt.xlabel('Recall')
    plt.ylabel('Frame')
    plt.ylim([0, 100])
    plt.xlim([0.0, 1.0])
    plt.title('Recall-mean_time' )
    plt.savefig(save_full)


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