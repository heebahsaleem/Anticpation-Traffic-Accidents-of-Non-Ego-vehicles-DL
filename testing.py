#Farhan-Heebah --Start--
#Start
# In this dataloader_split file, we are trying to split the data into train and test with the proportion of 
# 70% as train and 30 % as test;
# The folder structure have a root folder and further it is divided into - Frames, Detected, Tracked, Tracked_Avails;
# Frames -  This folder includes the raw images that needs to be passed when running the project from scratch;
# VideoNames as Foldername -> Raw images/Frames of the videos with 10fps;
# Detected - This folder includes the Bounding Box coordinates , x, y, cx, cy in the form of text file with the name given
# as video_name.txt format;
# Tracked - This folder includes the visualization folder and pickle files; visualization folder includes the frames with 
# Tracked_avails - This folder includes the pickel file only in the format - video_name.pkl
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
import math
import os
import numpy as np
from glob import glob
import pickle5 as pickle 

import torch
import pandas as pd
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations
import itertools


class dataloader_split(data.Dataset):
    
    def __init__(self, args, foldertype, phase):       
        self.foldertype = foldertype
        self.args = args
        self.phase=phase
        self.tlst=[]
        self.all_inputs=[]
# Here, we are using the foldertype - [tracked, tracked_avails, detected, frames]

        if self.foldertype == self.args.tracked_avails:
# Here, we are calling the tracked_avails. Now, the data_root path will be created via path mentioned in .yaml file 
# Then the sessions will save the list of all the files in the recursive manner i.e., all the files stored in directories
# subdirectories will be saved in the list named as sessions.
# now the __tracked_avails__() here returns the dictionary that is passed in the dataframe

                    if len(self.args.tracked_avails) > 0:
                        self.data_root = os.path.join(self.args.data_root, self.args.tracked_avails)
                        self.sessions = glob(os.path.join(self.data_root, '**/*.*'), recursive = True)
                        self.dataframe = pd.DataFrame(self.__returndict__())
                        self.tracked_avails_list = self.__combination__()    
                        self.pickle_dataframe_filtered = self.creating_dataframe()
                        

                    
        elif self.foldertype == self.args.tracked:
# same as above    
# This one is for the tracked folder 

                    if len(self.args.tracked) > 0:
                        self.data_root = os.path.join(self.args.data_root, self.args.tracked)
                        self.sessions = glob(os.path.join(self.data_root, '**/*.*'), recursive = True)
                        self.dataframe = pd.DataFrame(self.__returndict__())
 
                    
        elif self.foldertype == self.args.frames:
# same as above             
# This one is for the frames folder             
            
                    if len(self.args.frames) > 0:
                        self.data_root = os.path.join(self.args.data_root, self.args.frames)
                        self.sessions = glob(os.path.join(self.data_root, '**/*.*'), recursive = True)
                        self.dataframe = pd.DataFrame(self.__returndict__())

                        
        else:
# same as above             
# This one is for the detected folder             
            
                    if len(self.args.detected) > 0:
                        self.data_root = os.path.join(self.args.data_root, self.args.detected)
                        self.sessions = glob(os.path.join(self.data_root, '**/*.*'), recursive = True)
                        self.dataframe = pd.DataFrame(self.__returndict__())

                        
        self.__split__()
        
        return None
                

    def __flush_dataframe__(self):
        del self.dataframe
        del self.train_df
        del self.test_df
        return None

    def __len__(self):
        return len(self.all_inputs)
    
# To return the values as test, train dataframe and list which is actually a combination of pickle files     
# Here the values are collected at random basis after shuffling 
    def __getitem__(self, index):
        
        input_all_dec_h, target_risk_score, except_pred = self.all_inputs[index]
        input_all_dec_h = torch.FloatTensor(input_all_dec_h).to(self.args.device)
        target_risk_score = torch.FloatTensor(target_risk_score).to(self.args.device)
        #target_risk_score= target_risk_score.view(-1,1)
        except_pred = torch.FloatTensor(except_pred).to(self.args.device)
        #target_risk_score= torch.squeeze(target_risk_score)

        return input_all_dec_h, target_risk_score, except_pred
        
        
# Here all the files data are stored in the form of dictionary as per the requirement.
# Example if there is tracked avails pickle file then its video name, file path and file type are stored in the dictionary
# Then it is called in the init () to initialise the dataframe 


    def __returndict__(self):
        lst = []
        file_type = []
        if self.foldertype == self.args.tracked_avails:
            for sessions in self.sessions:
                if sessions.endswith('.pkl'):
                    file_type.append('pkl')
                lst.append(sessions.split('\\')[-2])        # Video names due to the folders as stated here 

        elif self.foldertype == self.args.tracked:
             for sessions in self.sessions:
                if sessions.endswith('.jpg') or sessions.endswith('.jpeg') or sessions.endswith('.png'): 
                    file_type.append('Images')
                    lst.append(sessions.split('\\')[-3])

                elif sessions.endswith('.pkl'):
                    file_type.append('pkl')
                    lst.append(sessions.split('\\')[-2])       # Video names due to the folders as stated here 
                    
        elif self.foldertype == self.args.frames:
            for sessions in self.sessions:
                if sessions.endswith('.jpg') or sessions.endswith('.jpeg') or sessions.endswith('.png'):
                    file_type.append("image") 
                lst.append(sessions.split('\\')[-2])
        
        else:
            for sessions in self.sessions:
                if sessions.endswith(".txt"):
                    file_type.append('Text')
                lst.append(sessions.split('.')[-2])
                
        dict = {'File_Path' : self.sessions,
                'Video_Name': lst,
                'File Type' : file_type
               }
        #print("return dict")
        return (dict)

    def fiveframes(self):
        #excel1=pd.read_excel("C:/Users/Heebah Saleem/Desktop/df1_common_frames.xlsx")
        #self.train_df= self.train_df[(self.train_df['accident_frame_common_frames']!=1000)& (self.train_df['accident_frame_common_frames']!=-1)]
        #Col: Flag. Flag = 0 means >=5 else Flag =1 means <5
        #1. dic: count and flag 
        #2 count: >4 flag =1 else 0

        flag = []
        count = []
        for i in range(len(self.train_df['common_frames'])):
                        list = str(self.train_df['common_frames'].iloc[i]).split(',')
                        #print(list)
                        count.append(len(list))
                        flag.append(1 if len(list)>4 else 0)
        #self.train_df['count']=count
        self.train_df['flag']=flag
        self.train_df=self.train_df.where(self.train_df['flag']==1)
        #print(len(self.train_df))
        self.train_df=self.train_df.dropna()
        #print(len(self.train_df))
        #print("chal gaya")
        return 0

    def __split__(self):

# This function will split the dataframe['Video_Name'] column into train, test
# As, self.dataframe.Video_Name.unique() : represents the selecting the unique values from a column in the dataframe 
# Reason : As dataframe is a form of table therefore, the mapping will be 1 to many. Thus, at column Video_Name if there 
# is a video name as abc then it will be repeating multiple times with respect to the fellow file associated to it in the 
# file_name column
# Therefore, when we do the train_test_split, it will fail to split the videonames by 70 : 30 percentage uniquely due to 
# the redundancy in the column. 
# Here, train and test stores the list of videonames in the 70:30 proportion
        if args.tracked_avails:
                train, test = train_test_split(self.pickle_dataframe_filtered.video_names.unique(), test_size = 0.3)
                if self.phase == 'Train':
                    self.train_df = self.pickle_dataframe_filtered.where(self.pickle_dataframe_filtered['video_names'].isin(train))
                    self.train_df = self.train_df.dropna()
                    #print(len(self.train_df))
                    #print(type(self.train_df['accident_frame_common_frames'].iloc[0]))
                    
                    
                    #self.train_df.to_excel(r'C:\DoTA_Avail_220726\self_train_df.xlsx', index = False)
                    for i in range(len(self.train_df['accident_frame_common_frames'])):
                            k = self.train_df['accident_frame_common_frames'].iloc[i]
                            t=str(k).split('.')[0]
                            t=t.replace("[","")
                            self.train_df['accident_frame_common_frames'].iloc[i]= int(t)
                    
                    #620positive and 700 negative data
                    positive_df=self.train_df[(self.train_df['accident_frame_common_frames']!=1000)& (self.train_df['accident_frame_common_frames']!=-1)]
                    #positive_df.to_excel(r'C:\DoTA_Avail_220726\positive_train_df.xlsx', index = False)
                    negative_df=self.train_df[(self.train_df['accident_frame_common_frames']==1000) | (self.train_df['accident_frame_common_frames']==-1)]
                    negative_df.groupby('video_names').nth(0)
                    #negative_df.to_excel(r'C:\DoTA_Avail_220726\negative_train_df.xlsx', index = False)
                    #then append
                    #print("positive df",len(positive_df))
                    #print("negative df",len(negative_df))
                    self.train_df = positive_df.append(negative_df)
                    #print("Before 5 frames", len(self.train_df))
                    self.fiveframes()
                    #print("After 5 frames", len(self.train_df))
                    
                    
                   
                    
                    
                    self.train_df = self.train_df.sample(frac=1)
                    #print(len(self.train_df))
                    for i in range(0,len(self.train_df)):
                        z, target_risk_score= self.extraction(self.train_df['values_names'].iloc[i][0], self.train_df['values_names'].iloc[i][1], self.train_df['common_frames'].iloc[i], self.train_df['video_names'].iloc[i])                      
                        #print(type(z),type(target_risk_score))
                        except_pred = np.ones(z.shape[0], dtype = float)
                        self.all_inputs.append([z, target_risk_score,except_pred])

            
                    
                else:
                    self.test_df  = self.pickle_dataframe_filtered.where(self.pickle_dataframe_filtered['video_names'].isin(test))
                    self.test_df  = self.test_df.dropna()
                    
                    #620positive and 700 negative data
                    positive_df=self.test_df[(self.test_df['accident_frame_common_frames']!=1000)& (self.test_df['accident_frame_common_frames']!=-1)]
                    negative_df=self.test_df[(self.test_df['accident_frame_common_frames']==1000) | (self.test_df['accident_frame_common_frames']==-1)].head(700)
                    #then append
                    self.test_df = positive_df.append(negative_df)
                    
                    self.test_df = self.test_df.sample(frac=1)
                    for i in range(0,len(self.test_df)):
                        z, target_risk_score= self.extraction(self.test_df['values_names'].iloc[i][0], self.test_df['values_names'].iloc[i][1], self.test_df['common_frames'].iloc[i], self.test_df['video_names'].iloc[i])
                        except_pred = np.ones(z.shape[0], dtype = float)
                        self.all_inputs.append([z, target_risk_score,except_pred])

        else:
            train, test = train_test_split(self.dataframe.Video_Name.unique(), test_size = 0.3)
# here the dataframe is filtered with by comparing all the values stored in the train. Thus only those values will be kept
# that are stored in train list and others will be dropped using .dropna()
        
        
            self.train_df = self.dataframe.where(self.dataframe['Video_Name'].isin(train))
            self.train_df = self.train_df.dropna()
            self.test_df  = self.dataframe.where(self.dataframe['Video_Name'].isin(test))
            self.test_df  = self.test_df.dropna()
            
 # same is implemented for the test_df as well 

        return None


# Here, the combination of pickle files into groups are created 
    def __combination__(self):    
        #print("combination")
        h={} 
        p={} 
        b = 2 
        lst = [] 
        tlst = []
        vlst = []
        for subdir, dirs, files in os.walk(os.path.join(self.args.data_root, self.args.tracked_avails)): 
            for file in files: 
                key, value = os.path.basename(subdir), file 
                h.setdefault(key, []).append(value) 
        #print("combination")
        for i in h.keys(): 
            c = list(itertools.combinations(h[i], b))
            for t in c:
                tlst.append(t)
                vlst.append(i)
        
        p = {
            'video_names': vlst,
            'file_names' : tlst}         
        #print('Tracked Avails - Combinations')
        #print("combination")
        return (p)
    
    
    
    def creating_dataframe(self):
        p = self.tracked_avails_list
    
# creating another list of video_names, object1, object2 for the dictionary:
    
    
    
        obj1 =[]
        obj2 =[]
    
        for files in p['file_names']:
            k = files
            obj1.append(k[0].split('_')[2].split('.pkl')[0])
            obj2.append(k[1].split('_')[2].split('.pkl')[0])


        dict  = {"video_names": p['video_names'],
                    "obj1": obj1,
                    "obj2": obj2,
                    "values_names": p['file_names']}

        
# Here all the object id and video names are stored in the dataframe 


        pickle_dataframe = pd.DataFrame(dict)
        pickle_dataframe['obj1'] = pickle_dataframe['obj1'].astype('int32')
        pickle_dataframe['obj2'] = pickle_dataframe['obj2'].astype('int32')
        klst = []
        flst = []
        for i in range(0, len(pickle_dataframe)):
            klst.append(self.return_common_frames(pickle_dataframe['values_names'][i][0], pickle_dataframe['values_names'][i][1], pickle_dataframe['video_names'][i], args.tracked_avails ,args.data_root))
            flst.append(self.return_common_frames_accident_frame(pickle_dataframe['values_names'][i][0], pickle_dataframe['values_names'][i][1], pickle_dataframe['video_names'][i], args.tracked_avails ,args.data_root))

        pickle_dataframe['common_frames'] = klst
        pickle_dataframe['accident_frame_common_frames'] = flst
        #print('creating df')
        #return (self.annotation_file( os.path.join(r'E:\DoTA_Avail_220726\NEgo', "DoTA_Non_Ego_Done_220216_relabelled.xlsx"), pickle_dataframe))
        pickle_dataframe=pickle_dataframe[pickle_dataframe['common_frames'].notna()]
        return (pickle_dataframe)
    



# This fucntion will extract all the common frames_id if exists between the two object pickle files collected 
# from the tracked_avails
# This function is called in the creating_dataframe function in line no.46
# Returns a list of particular common frame id and stored in the pickle_dataframe['common_frames'] 
    def return_common_frames(self, filename_1, filename_2, video_name, folder_name, root_folder):
    
        lst = []
#   file = pickle_dataframe['values_names'][0][0]
#   part = os.path.join(args.data_root, args.tracked)
#   file  = os.path.join(part, pickle_dataframe['video_names'][0], file)
#   print(file)
    
        path_1 = os.path.join(root_folder, folder_name, video_name, filename_1)
        path_2 = os.path.join(root_folder, folder_name, video_name, filename_2)
    
        with open(path_1,'rb') as f:
            x = pickle.load(f)
        
        with open(path_2, 'rb') as f:
            y = pickle.load(f)
    
#   print("x:", x['frame_id'], "y:", y['frame_id'])
        for item in set(x['frame_id']).intersection(y['frame_id']):
            if item is not None:
                lst.append(item)
            else:
                continue
          
    
        if len(lst) < 1 :
             return None 
        else :
            return lst

        
        
    def return_common_frames_accident_frame(self, filename_1, filename_2, video_name, folder_name, root_folder):
    
        
        accident_frame_list=[]
#   file = pickle_dataframe['values_names'][0][0]
#   part = os.path.join(args.data_root, args.tracked)
#   file  = os.path.join(part, pickle_dataframe['video_names'][0], file)
#   print(file)
    
        path_1 = os.path.join(root_folder, folder_name, video_name, filename_1)
        path_2 = os.path.join(root_folder, folder_name, video_name, filename_2)
    
        with open(path_1,'rb') as f:
            x = pickle.load(f)
        
        with open(path_2, 'rb') as f:
            y = pickle.load(f)
    
#   print("x:", x['frame_id'], "y:", y['frame_id'])
        
        #set1 = list(int(x['accident_frame_id']))
        #set2 = list(int(y['accident_frame_id']))
        #print(set(set1).intersection(set2))
        #print(x['accident_frame_id'], y['accident_frame_id'])
        #for frame in set(x['accident_frame_id']).intersection(y['accident_frame_id']):
        if x['accident_frame_id'] == y['accident_frame_id'] :
            accident_frame_list.append(x['accident_frame_id'])

            #if frame is not None:
            #    accident_frame_list.append(frame)
            #else:
            #    continue        
    
        if len(accident_frame_list) < 1:
             return None 
        else :
            return accident_frame_list


# This file is manually labelled and used as a filter for the extraction of accidental groups 
# Thus the original dataframe underwent left join with the excel annotation file 
# Left Join - When file X is left joined with File Y, it only shows all those values that are in X and in (X && Y)
# if left join is done on X with respect to Y.
# Therefore, all the values that are stored in excel file will be filtered with the pickle dataframe rest all the 
# drop.na() is implemented to deleted Nan Values there 
# At the end it will return pickle_dataframe_filtered

    def annotation_file(self, csv_file_path, pickle_dataframe):
    
        excel_df = pd.read_excel(csv_file_path, sheet_name=1)
        filter =  (excel_df['acc_obj_id1'] != 0) & (excel_df['acc_obj_id2'] != 0) 
        excel_df = excel_df.where(filter).dropna()

        excel_df['acc_obj_id1'] = excel_df['acc_obj_id1'].astype(int)
        excel_df['acc_obj_id2'] = excel_df['acc_obj_id2'].astype(int)
    
        pickle_dataframe_filtered = pd.merge(pickle_dataframe, excel_df, left_on=['video_names','obj1', 'obj2'], right_on = ['Name','acc_obj_id1', 'acc_obj_id2'], how='left')
        pickle_dataframe_filtered = pickle_dataframe_filtered.dropna()

        #print('pickle_dataframe')
        return pickle_dataframe_filtered
   

    
    
# We need the common frame_ids, their location and need to extact the arrays-list of hidden states from that location 
# From here all the below written functions will be called in the __get_item__() because here we are collecting the 
# the list of all the hidden states generated in the system
    def extraction(self, file_name_1, file_name_2,  common_frames, video_names):
        #print('filename1:', file_name_1, 'filename2:', file_name_2)
            
#   Now we will create a path for the Hidden state 

        file_path_1 = os.path.join(self.args.data_root, self.args.hiddens, file_name_1)
        file_path_2 = os.path.join(self.args.data_root, args.hiddens, file_name_2)

#   File path generated 

# Now we will open the hidden state based pickle file that needs to be run 
        with open(file_path_1,'rb') as f:
            hidden_x = pickle.load(f)
            
            
        with open(file_path_2,'rb') as f:
            hidden_y = pickle.load(f)
#Target risk score:
        if (hidden_x['target_risk_score'][0] > 0.0 and hidden_y['target_risk_score'][0] > 0.0):
            target_risk_score = np.array([1.0])
        else:
            target_risk_score = np.array([0.0])
        #if (hidden_x['target_risk_score'] > hidden_y['target_risk_score']):
        #    target_risk_score = hidden_x['target_risk_score']
        #else:
        #    target_risk_score = hidden_y['target_risk_score']    
        

# Extracting the file path of tracked avails 
        file_path_tracked_avail_1 = os.path.join(self.args.data_root, self.args.tracked_avails, video_names,file_name_1)
        file_path_tracked_avail_2 = os.path.join(self.args.data_root, self.args.tracked_avails, video_names, file_name_2)


# First Tracked avail pickle file to be imported
        with open(file_path_tracked_avail_1,'rb') as f:
            file_path_tracked_avail_pickle_1 = pickle.load(f)


# Second tracked avail pickle file to be imported         
        with open(file_path_tracked_avail_2,'rb') as f:
            file_path_tracked_avail_pickle_2 = pickle.load(f)
        

        file_1_lst = []
        file_2_lst = []

# Position of common frames are extracted from the tracked avails pickle files
        for i in range(len(common_frames)):
            for j in file_path_tracked_avail_pickle_1['frame_id']:
                if common_frames[i] == j:
                    file_1_lst.append(list(file_path_tracked_avail_pickle_1['frame_id']).index(j))

        for i in range(len(common_frames)):
            for j in file_path_tracked_avail_pickle_2['frame_id']:
                if common_frames[i] == j:
                    file_2_lst.append(list(file_path_tracked_avail_pickle_2['frame_id']).index(j))
    
        return (self.appending_hidden_states( hidden_x, hidden_y, common_frames, file_1_lst, file_2_lst), target_risk_score)
    
     

# Appending the hidden states into the array and returning the appended array from the same 
# The attributes that have been taken into consideration are the x, y as the pickle files of hidden states,
# common frames from the data_frame and list1 and list 2 collected from the extraction function is passed here
# It returns the list of appended ndarray in the form of (x,10,2048) shape i.e., z 

    def appending_hidden_states(self, x, y, common_frames, lst_1, lst_2):
        #print(x['hidden_state'][0],lst_1,lst_2)
        #min_range = min(len(x['hidden_state']), len(y['hidden_state']))
        min_range=min(len(lst_1), len(lst_2))
        z = np.zeros([min_range, len(x['hidden_state'][0]), 2048], dtype = np.float64)

        for k in range(min_range):
            for i in range(0, len(x['hidden_state'][0])):
                z[k][i] = list(np.append(x['hidden_state'][lst_1[k]][i], y['hidden_state'][lst_2[k]][i]))
    
    
        return z    
    
############### TEST
import os
import numpy as np
import time

import torch
from torch import optim
from torch.utils import data
from torchsummaryX import summary

from loss import train_loss_cal, val_loss_cal, test_results
from models import AP

#from .utils import dataloaderHidden
import argparse
import yaml 
def parse_args(a=None):
    file_name = r"C://Users//user//Downloads//scene_agnostic_anticipation//scene_agnostic_anticipation-main//configs//od_yolox.yaml"
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config',
                        dest='config_file',
                        default=file_name,
                        # type=argparse.FileType(mode='r'),
                        help='The yaml configuration file')

    if a == None:
        args, unprocessed_args = parser.parse_known_args()
    else:
        args, unprocessed_args = parser.parse_known_args(args=a)

    if args.config_file:
        with open(args.config_file, 'r') as f:
#            try:
                parser.set_defaults(**yaml.safe_load(f))
#            except:
#                print('Exception_1')

        try:
                args = parser.parse_known_args(unprocessed_args)
        except:
                print('EXCEPTION_2')

    return args

from tensorboardX import SummaryWriter
import pandas as pd

args = parse_args()
args = args[0]

dataloader_params ={
            "batch_size": args.batch_size,
            "shuffle": args.shuffle
        }

val_set = dataloader_split(args, args.tracked_avails, 'Test')
#print(">> Number of validation samples:", val_set.__len__())
val_gen = data.DataLoader(val_set, **dataloader_params)

AP_model = AP(args).to(args.device)
AP_model.load_state_dict(torch.load(r'C:\Users\user\Downloads\scene_agnostic_anticipation\scene_agnostic_anticipation-main\lib\weights\best_AP_model_AP.pt'))

val_AP = test_results(AP_model, val_gen, before_ATTC=0, verbose=True)

#Farhan-Heebah --End--