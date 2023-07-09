import torch
import numpy as np
from tqdm import tqdm
from pdb import set_trace
from timeit import default_timer as timer
from datetime import timedelta

###### Heebah Start ########
def EL(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC):
    '''
        pred_risk_score : [batch_size, max_length, 1]
        target_risk_score : [batch_size, 1]
        except_pred = [batch size, max_length]
    '''

    total_loss = 0.0

    binary_ce = torch.nn.BCELoss()
    batch_size = pred_risk_score.shape[0]
    max_length = pred_risk_score.shape[1]
    hyper_parameter = 0.5  # lambda, gamma

    # print(">> Traget risk_score:", target_risk_score)

    for i in range(batch_size):  # batchsize

        for j in range(max_length):
            # positive loss
            
            #d_time = (max_length - j - 1) / 10.0
            #term = d_time - before_ATTC - hyper_parameter
            #penalty_weight = np.exp(-np.max([0, term]))
            #pos_loss = penalty_weight * binary_ce(pred_risk_score[i, j], target_risk_score[i])
            
            # positive loss:
            d_time = (max_length-j-1) / 10.0 #max_length:total frames; j:current frames.  
            #term = d_time - before_ATTC - hyper_parameter
            penalty_weight = np.exp(-np.max([0, d_time]))
            pos_loss = penalty_weight*binary_ce(pred_risk_score[i, j], target_risk_score[i])
            
            # negative loss:
            neg_loss = binary_ce(pred_risk_score[i, j], target_risk_score[i])
            # neg_loss = binary_ce(pred_risk_score[i,j], target_risk_score[i,j])

            temp_loss = pos_loss * target_risk_score[i] + neg_loss * (1 - target_risk_score[i])

            # except loss: 
            total_loss += temp_loss * except_pred[i, j]

    # batch_loss = total_loss / batch_size

    return total_loss

###### Heebah End ########

def AdaLEA(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC): 
    '''
        pred_risk_score : [batch_size, max_length, 1]
        target_risk_score : [batch_size, 1]
        except_pred = [batch size, max_length]
    '''

    total_loss = 0.0

    binary_ce = torch.nn.BCELoss()
    batch_size = pred_risk_score.shape[0]
    max_length = pred_risk_score.shape[1]
    hyper_parameter = 0.5  # lambda, gamma

    # print(">> Traget risk_score:", target_risk_score)

    for i in range(batch_size):  # batchsize

        for j in range(max_length):
            # positive loss
            d_time = (max_length - j - 1) / 10.0
            term = d_time - before_ATTC - hyper_parameter
            penalty_weight = np.exp(-np.max([0, term]))
            pos_loss = penalty_weight * binary_ce(pred_risk_score[i, j], target_risk_score[i])
            # negative loss
            neg_loss = binary_ce(pred_risk_score[i, j], target_risk_score[i])
            # neg_loss = binary_ce(pred_risk_score[i,j], target_risk_score[i,j])

            temp_loss = pos_loss * target_risk_score[i] + neg_loss * (1 - target_risk_score[i])

            # except loss
            total_loss += temp_loss * except_pred[i, j]

    # batch_loss = total_loss / batch_size

    return total_loss

def cal_ATTC(all_pred, all_labels, max=None):
    # calculate mean TTC(time to collision) in Batch data
    # all_label : [data_num, 1]
    # all_pred : dictionary

    temp_shape = 0
    all_pred_flatten = []
    clips_list = list(all_pred.keys())
    num_clips = len(clips_list)

    cnt = 0
    AP = 0.0

    for i in range(num_clips):
        key_value = list(all_pred.keys())[i]
        temp_shape += len(all_pred[key_value])
        all_pred_flatten.extend(all_pred[key_value])

    min_threshold = sorted(all_pred_flatten)[0]
    if max is None:
        max_threshold = sorted(all_pred_flatten)[-1]
    else:
        max_threshold = 1.0

    thresholds = np.arange(min_threshold, max_threshold, 0.0005)
    thresholds = np.append(thresholds, max_threshold)
    thresholds = np.delete(thresholds, 0)
    # print("num of thresholds :", len(thresholds))

    #Heebah --Start--
    True_pos = np.zeros((len(thresholds))) #True Positive
    Falsepos = np.zeros((len(thresholds))) #True pos False pos
    Falseneg = np.zeros((len(thresholds))) #True neg False neg
    True_neg = np.zeros((len(thresholds))) #True Negative
    #Heebah --End--
    
    Precision = np.zeros((len(thresholds)))
    Recall = np.zeros((len(thresholds)))
    Time = np.zeros((len(thresholds)))
    
    for threshold in thresholds:
        Tp = 0.0
        Tp_Fp = 0.0  # TP + FP
        Tp_Tn = 0.0
        Tn_Fn = 0.0 #Heebah
        Tn = 0.0 #Heebah
        frame_to_acc = 0.0
        counter = 0.0
        for i in range(len(clips_list)):
            tp = np.where(all_pred[clips_list[i]] * all_labels[i] >= threshold)
            Tp += float(len(tp[0]) > 0)
            
            tn = np.where((all_pred[clips_list[i]] * (1.0-all_labels[i]) < threshold) & (all_pred[clips_list[i]] * (1.0-all_labels[i]) > 0.0)) #Heebah
            Tn +=  float(len(tn[0]) > 0) #Heebah    
            if float(len(tp[0]) > 0) > 0:
                frame_to_acc += len(all_pred[clips_list[i]]) - tp[0][0]
                counter = counter + 1
            Tp_Fp += float(len(np.where(all_pred[clips_list[i]] >= threshold)[0]) > 0)
            Tn_Fn += float(len(np.where(all_pred[clips_list[i]] < threshold)[0]) > 0) #Heebah
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
            #Heebah --Start--
            True_pos[cnt] = np.nan
            Falsepos[cnt] = np.nan
            Falseneg[cnt] = np.nan
            True_neg[cnt] = np.nan
            #Heebah --End--
             
        else:
            Precision[cnt] = Tp / Tp_Fp
            #Heebah --Start--
            True_pos[cnt] = Tp
            Falsepos[cnt] = Tp_Fp - Tp
            Falseneg[cnt] = Tn_Fn - Tn
            True_neg[cnt] = Tn
            #Heebah --End--
            
        if np.sum(all_labels) == 0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp / np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (frame_to_acc / counter)
        cnt += 1

    #set_trace()
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    
    True_neg = True_neg[new_index]
    Falseneg = Falseneg[new_index]
    Falsepos = Falsepos[new_index]
    True_pos = True_pos[new_index]

    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    newTp = np.zeros(len(rep_index))
    newFp = np.zeros(len(rep_index))
    newTn = np.zeros(len(rep_index))
    newFn = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

        newTp[i] = np.max(True_pos[rep_index[i]:rep_index[i + 1]])
        newFp[i] = np.max(Falsepos[rep_index[i]:rep_index[i + 1]])
        newTn[i] = np.max(True_neg[rep_index[i]:rep_index[i + 1]])
        newFn[i] = np.max(Falseneg[rep_index[i]:rep_index[i + 1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    #set_trace()

    new_Recall = new_Recall[~np.isnan(new_Time)]
    new_Precision = new_Precision[~np.isnan(new_Time)]
    new_Time = new_Time[~np.isnan(new_Time)]

    # print("###########################")
    # print("all label:", all_labels)
    # print("new_recall:", new_Recall)
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    ATTC = np.mean(new_Time) / 10.0

    if AP > 1.:
        new_Precision[0] = new_Precision[1]
        if new_Recall[0] != 0:
            AP += new_Precision[0] * (new_Recall[0] - 0)
        for i in range(1, len(new_Precision)):
            AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    if np.isnan(ATTC):
        print("ATTC is nan value !!!")
        print("Check new_Time")
        print(new_Time)

    return AP, ATTC

###### Farhan - Heebah Start For Testing ########
def cal_ATTC_t5(all_pred, all_labels, max=None):
    # calculate mean TTC(time to collision) in Batch data
    # all_label : [data_num, 1]
    # all_pred : dictionary

    temp_shape = 0
    all_pred_flatten = []
    clips_list = list(all_pred.keys())
    num_clips = len(clips_list)

    cnt = 0
    AP = 0.0

    # for i in range(num_clips):
    #     key_value = list(all_pred.keys())[i]
    #     temp_shape += len(all_pred[key_value])
    #     all_pred_flatten.extend(all_pred[key_value])

    # min_threshold = sorted(all_pred_flatten)[0]
    # if max is None:
    #     max_threshold = sorted(all_pred_flatten)[-1]
    # else:
    #     max_threshold = 1.0

    # thresholds = np.arange(min_threshold, max_threshold, 0.0005)
    # thresholds = np.append(thresholds, max_threshold)
    # thresholds = np.delete(thresholds, 0)
    # print("num of thresholds :", len(thresholds))

    #Heebah --Start--
    #True_pos = np.zeros((len(thresholds))) #True Positive
    #Truepos_Falsepos = np.zeros((len(thresholds))) #True pos False pos
    #Trueneg_Falseneg = np.zeros((len(thresholds))) #True neg False neg
    #True_neg = np.zeros((len(thresholds))) #True Negative
    #Heebah --End--
    
    #Precision = np.zeros((len(thresholds)))
    #Recall = np.zeros((len(thresholds)))
    #Time = np.zeros((len(thresholds)))
    
    threshold = 0.5

    #for threshold in thresholds:
    Tp = 0.0
    Tp_Fp = 0.0  # TP + FP
    Tp_Tn = 0.0
    Tn_Fn = 0.0 #Heebah
    Tn = 0.0 #Heebah
    frame_to_acc = 0.0
    counter = 0.0
    for i in range(len(clips_list)):
        tp = np.where(all_pred[clips_list[i]] * all_labels[i] >= threshold)
        Tp += float(len(tp[0]) > 0)
        
        tn = np.where((all_pred[clips_list[i]] * (1.0-all_labels[i]) < threshold) & (all_pred[clips_list[i]] * (1.0-all_labels[i])>0.0)) #Heebah
        Tn +=  float(len(tn[0]) > 0) #Heebah
        if float(len(tp[0]) > 0) > 0:
            frame_to_acc += len(all_pred[clips_list[i]]) - tp[0][0]
            counter = counter + 1
        Tp_Fp += float(len(np.where(all_pred[clips_list[i]] >= threshold)[0]) > 0)
        Tn_Fn += float(len(np.where(all_pred[clips_list[i]] < threshold)[0]) > 0) #Heebah
        set_trace()
    print('total clips --- ', len(clips_list))
    print('TP --- ', Tp)
    print('TN --- ', Tn)
    print('FP --- ', Tp_Fp - Tp)
    print('FN --- ', Tn_Fn - Tn)

    AP = Tp/Tp_Fp


    # if Tp_Fp == 0:
    #     Precision[cnt] = np.nan
    #     #Heebah --Start--
    #     True_pos[cnt] = np.nan
    #     Truepos_Falsepos[cnt] = np.nan
    #     Trueneg_Falseneg[cnt] = np.nan
    #     True_neg[cnt] = np.nan
    #     #Heebah --End--
            
    # else:
    #     Precision[cnt] = Tp / Tp_Fp
    #     #Heebah --Start--
    #     True_pos[cnt] = Tp
    #     Truepos_Falsepos[cnt] = Tp_Fp
    #     Trueneg_Falseneg[cnt] = Tn_Fn
    #     True_neg[cnt] = Tn
    #     #Heebah --End--
        
    # if np.sum(all_labels) == 0:
    #     Recall[cnt] = np.nan
    # else:
    #     Recall[cnt] = Tp / np.sum(all_labels)
    # if counter == 0:
    #     Time[cnt] = np.nan
    # else:
    #     Time[cnt] = (frame_to_acc / counter)
    # cnt += 1

    # new_index = np.argsort(Recall)
    # Precision = Precision[new_index]
    # Recall = Recall[new_index]
    # Time = Time[new_index]
    # _, rep_index = np.unique(Recall, return_index=1)
    # new_Time = np.zeros(len(rep_index))
    # new_Precision = np.zeros(len(rep_index))
    # for i in range(len(rep_index) - 1):
    #     new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
    #     new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

    # new_Time[-1] = Time[rep_index[-1]]
    # new_Precision[-1] = Precision[rep_index[-1]]
    # new_Recall = Recall[rep_index]
    # new_Time = new_Time[~np.isnan(new_Precision)]
    # new_Recall = new_Recall[~np.isnan(new_Precision)]
    # new_Precision = new_Precision[~np.isnan(new_Precision)]

    # new_Recall = new_Recall[~np.isnan(new_Time)]
    # new_Precision = new_Precision[~np.isnan(new_Time)]
    # new_Time = new_Time[~np.isnan(new_Time)]

    # # print("###########################")
    # # print("all label:", all_labels)
    # # print("new_recall:", new_Recall)
    # if new_Recall[0] != 0:
    #     AP += new_Precision[0] * (new_Recall[0] - 0)
    # for i in range(1, len(new_Precision)):
    #     AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    # ATTC = np.mean(new_Time) / 10.0

    # if AP > 1.:
    #     new_Precision[0] = new_Precision[1]
    #     if new_Recall[0] != 0:
    #         AP += new_Precision[0] * (new_Recall[0] - 0)
    #     for i in range(1, len(new_Precision)):
    #         AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    # if np.isnan(ATTC):
    #     print("ATTC is nan value !!!")
    #     print("Check new_Time")
    #     print(new_Time)

    return AP
###### Farhan - Heebah End For Testing ########


#Heebah -- Start--
#def confusionmatrix():
    temp_shape = 0
    all_pred_flatten = []
    clips_list = list(all_pred.keys())
    num_clips = len(clips_list)

    cnt = 0
    AP = 0.0

    for i in range(num_clips):
        key_value = list(all_pred.keys())[i]
        temp_shape += len(all_pred[key_value])
        all_pred_flatten.extend(all_pred[key_value])

    min_threshold = sorted(all_pred_flatten)[0]
    if max is None:
        max_threshold = sorted(all_pred_flatten)[-1]
    else:
        max_threshold = 1.0

    thresholds = np.arange(min_threshold, max_threshold, 0.0005)
    thresholds = np.append(thresholds, max_threshold)
    thresholds = np.delete(thresholds, 0)
    # print("num of thresholds :", len(thresholds))

    True_pos = np.zeros((len(thresholds))) #True Positive
    Truepos_Falsepos = np.zeros((len(thresholds))) #True pos False pos
    Trueneg_Falseneg = np.zeros((len(thresholds))) #True neg False neg
    True_neg = np.zeros((len(thresholds))) #True Negative
    
    Precision = np.zeros((len(thresholds)))
    Recall = np.zeros((len(thresholds)))
    Time = np.zeros((len(thresholds)))
    
    for threshold in thresholds:
        Tp = 0.0
        Tp_Fp = 0.0  # TP + FP
        Tp_Tn = 0.0
        Tn_Fn = 0.0
        Tn = 0.0
        frame_to_acc = 0.0
        counter = 0.0
        for i in range(len(clips_list)):
            tp = np.where(all_pred[clips_list[i]] * all_labels[i] >= threshold)
            Tp += float(len(tp[0]) > 0)
            tn = np.where(all_pred[clips_list[i]] * all_labels[i] < threshold)
            Tn +=  float(len(tn[0]) > 0)
            if float(len(tp[0]) > 0) > 0:
                frame_to_acc += len(all_pred[clips_list[i]]) - tp[0][0]
                counter = counter + 1
            Tp_Fp += float(len(np.where(all_pred[clips_list[i]] >= threshold)[0]) > 0)
            Tn_Fn += float(len(np.where(all_pred[clips_list[i]] < threshold)[0]) > 0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
            True_pos[cnt] = np.nan
            Truepos_Falsepos[cnt] = np.nan
            Trueneg_Falseneg[cnt] = np.nan
            True_neg[cnt] = np.nan
             
        else:
            Precision[cnt] = Tp / Tp_Fp
            True_pos[cnt] = Tp
            Truepos_Falsepos[cnt] = Tp_Fp
            Trueneg_Falseneg[cnt] = Tn_Fn
            True_neg[cnt] = Tn
            
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
    
    True_pos = True_pos[new_index]
    Truepos_Falsepos = Truepos_Falsepos[new_index]
    Trueneg_Falseneg = Trueneg_Falseneg[new_index]
    True_neg = True_neg[new_index] 
    
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])
      
    True_pos = np.zeros(len(rep_index))
    Truepos_Falsepos = np.zeros(len(rep_index))
    Trueneg_Falseneg = np.zeros(len(rep_index))
    True_neg = np.zeros(len(rep_index)) 

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    new_Recall = new_Recall[~np.isnan(new_Time)]
    new_Precision = new_Precision[~np.isnan(new_Time)]
    new_Time = new_Time[~np.isnan(new_Time)]





    return True_pos, Truepos_Falsepos, Trueneg_Falseneg, True_neg

#Heebah --End--

def train_loss_cal(epoch, ap_model, optimizer, train_gen, before_ATTC, verbose=True, trainable=True):
    ap_model.train()
    print(" Train ...")
    # fol_model.eval() #use pre-trained fol model ( fol model already training !!! )

    total_train_loss = 0.
    all_pred = {}
    all_labels = np.zeros(len(train_gen))

    loader = tqdm(train_gen, total=len(train_gen))

    # print("-- len(train_gen.dataset) :", len(train_gen.dataset)) = the number of dataset

    n = 0
    time = [] 
    with torch.set_grad_enabled(trainable):
        for batch_idx, data in enumerate(loader):
            #start = timer() #Inference Time start
            
            input_all_dec_h, target_risk_score, except_pred = data

            max_length = input_all_dec_h.shape[1]
            batch_size = input_all_dec_h.shape[0]

            pred_risk_score = ap_model(input_all_dec_h)
            # print("pred_risk_score:", pred_risk_score.shape) # [batch_size, max_length, 1]

            risk_score_pred_loss = AdaLEA(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC)
            #risk_score_pred_loss = EL(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC) #Added for experiment : Heebah

            batch_loss = risk_score_pred_loss / max_length
            loss = batch_loss / batch_size

            total_train_loss += loss.item()
            if trainable:
                optimizer.zero_grad()  # avoid gradient accumulate from loss.backward()
                loss.backward()
                optimizer.step()

            for i in range(batch_size):
                nth = n * batch_size + i
                array_pred_risk_score_i = pred_risk_score[i].cpu().detach().numpy()
                all_pred[nth] = array_pred_risk_score_i.reshape((-1,))
                all_labels[nth] = target_risk_score[i].cpu().detach().numpy()

            n += 1
            #end = timer() #Inference Time end
            #print(time.append(timedelta(seconds=end-start)))
    # print("check n:", n) # 6100?
    avg_risk_score_pred_loss = total_train_loss / n

    # if verbose:
    #    print('\nTrain set: Average loss: {:.4f},\n'.format(avg_risk_score_pred_loss))

    # -------------------------s--------------------------
    # Calculate ATTC & AP on Train Dataset
    # print("length of dict-all_pred:", len(all_pred))
    # print("all_labels:", all_labels.shape)
    AP, ATTC = cal_ATTC(all_pred, all_labels)
    #print("Sum inference time :", np.sum(time) )
    #print("Avearge inference time :", np.mean(time) ) 
    print("-- AP on Train dataset :", AP) 
    print("-- ATTC on Train dataset :", ATTC)

    return avg_risk_score_pred_loss, AP, ATTC


def val_loss_cal(epoch, ap_model, val_gen, before_ATTC, verbose=True):
    print(" Validation ...")
    ap_model.eval()
    # fol_model.eval()
    total_val_loss = 0
    all_pred = {}
    all_labels = np.zeros(len(val_gen))

    loader = tqdm(val_gen, total=len(val_gen))

    n_v = 0
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):

            input_all_dec_h, target_risk_score, except_pred = data
            max_length = input_all_dec_h.shape[1]
            batch_size = input_all_dec_h.shape[0]

            # run forward
            pred_risk_score = ap_model(input_all_dec_h)

            # compute loss
            risk_score_pred_loss = AdaLEA(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC)
            #risk_score_pred_loss = EL(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC) #Added for experiment : Heebah
            batch_loss = risk_score_pred_loss / max_length
            loss = batch_loss / batch_size

            total_val_loss += loss.item()

            for i in range(batch_size):
                nth = n_v * batch_size + i
                array_pred_risk_score_i = pred_risk_score[i].cpu().detach().numpy()
                all_pred[nth] = array_pred_risk_score_i.reshape((-1,))
                all_labels[nth] = target_risk_score[i].cpu().detach().numpy()

            n_v += 1

    avg_val_loss = total_val_loss / n_v

    # if verbose:
    #    print('\nVal set: Average loss: {:.6f},\n'.format(avg_val_loss))

    # ---------------------------------------------------
    # Calculate ATTC & AP on Train Dataset
    # print("length of dict-all_pred:", len(all_pred))
    # print("all_labels:", all_labels.shape)
    AP, ATTC = cal_ATTC(all_pred, all_labels)
    print("-- AP on Validation dataset :", AP)
    print("-- ATTC on Validation dataset :", ATTC)

    return avg_val_loss, AP, ATTC

#Farhan-Heebah --Start-- For Testing
def test_results(ap_model, val_gen, before_ATTC=0, verbose=True):
    print(" Testing ...")
    ap_model.eval()
    # fol_model.eval()
    total_val_loss = 0
    all_pred = {}
    all_labels = np.zeros(len(val_gen))

    loader = tqdm(val_gen, total=len(val_gen))

    n_v = 0
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):

            input_all_dec_h, target_risk_score, except_pred = data
            max_length = input_all_dec_h.shape[1]
            batch_size = input_all_dec_h.shape[0]

            # run forward
            pred_risk_score = ap_model(input_all_dec_h)

            # compute loss
            #risk_score_pred_loss = AdaLEA(pred_risk_score, target_risk_score, epoch, except_pred, before_ATTC)
            risk_score_pred_loss = EL(pred_risk_score, target_risk_score, 1, except_pred, before_ATTC)
            batch_loss = risk_score_pred_loss / max_length
            loss = batch_loss / batch_size

            total_val_loss += loss.item()

            for i in range(batch_size):
                nth = n_v * batch_size + i
                array_pred_risk_score_i = pred_risk_score[i].cpu().detach().numpy()
                all_pred[nth] = array_pred_risk_score_i.reshape((-1,))
                all_labels[nth] = target_risk_score[i].cpu().detach().numpy()

            n_v += 1

    avg_val_loss = total_val_loss / n_v

    # if verbose:
    #    print('\nVal set: Average loss: {:.6f},\n'.format(avg_val_loss))

    # ---------------------------------------------------
    # Calculate ATTC & AP on Train Dataset
    # print("length of dict-all_pred:", len(all_pred))
    # print("all_labels:", all_labels.shape)
    AP, ATTC = cal_ATTC(all_pred, all_labels)
    print("-- AP on Validation dataset :", AP)
    print("-- ATTC on Validation dataset :", ATTC)

    return  AP, ATTC
#Farhan-Heebah --End--