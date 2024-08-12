from cProfile import label
import re
import cv2
import hydra
from scipy import rand
from sklearn.manifold import TSNE
import torch.utils.data as data
import torch
from collections import defaultdict
from torchvision import datasets, transforms
from PIL import Image
import json
import os
import pandas as pd
import zipfile
import numpy as np
import random

prefix = 'v_'
### todo file the config, complete
class VideoDataset(data.Dataset):
    def __init__(self,args):
        super(VideoDataset,self).__init__()
        self.args = args
        self.params = self.args[self.args.task]
        
        if self.args.dataset == 'ActivityNet':
            self.ac_classes = self.params.an_classes
            self.extract_path = self.args.data_paths.ActivityNet.extracted
            self.annotation = self.args.data_paths.ActivityNet.annotation
            self.action_name = self.args.data_paths.ActivityNet.action_name
            self.origianl_video = self.args.data_paths.ActivityNet.original
        elif self.args.dataset == 'THUMOS14':
            self.ac_classes = self.params.thu_classes
            self.extract_path = self.args.data_paths.THUMOS14.extracted
            self.annotation = self.args.data_paths.THUMOS14.annotation
            
        self.annot = json.load(open(self.annotation,'r'))
        self.video_dict = {}
        self.video_times = defaultdict(int)
        # get task specific data dict, for each class choose one query and 1 or 5 support videos
        self.get_fs_data(self.args.task,self.args.dataset)

        # self.class2idx = {c:i for i,c in enumerate(self.an_classes)}



        # 200
        # self.classes = list(self.video_dict.keys())
        # self.class2idx = {c:i for i,c in enumerate(self.classes)}

    def get_fs_data(self,task,dataset):
        if dataset == 'ActivityNet':
            self.classes = pd.read_csv(self.action_name)
            self.label2index = {label:index for index,label in enumerate(self.classes['action'])}
            video_names = os.listdir(self.extract_path)
            for video_name in video_names:
                video_name = video_name[:-4]
                #only run one times
                # #delete the video which is not in the annotation
                # if video_name not in self.annot['database'].keys():
                #     os.remove(os.path.join(self.extract_path,video_name+'.npy'))
                #     continue
                # #delete the video which is shorter than 10s
                # if self.annot['database'][video_name]['duration'] < 10:
                #     os.remove(os.path.join(self.extract_path,video_name+'.npy'))
                #     continue
                # #delete the video which is not 30 fps
                # cap = cv2.VideoCapture(os.path.join(self.origianl_video,prefix + video_name + '.mp4'))
                # fps = cap.get(cv2.CAP_PROP_FPS)
                # if fps < 29 or fps > 30:
                #     os.remove(os.path.join(self.extract_path,video_name+'.npy'))
                #     continue
                # cap.release()

                if self.annot['database'][video_name]['subset'] == task:
                #if self.annot['database'][video_name]['subset'] == task or self.annot['database'][video_name]['subset'] == 'validation':
                    video_class = self.annot['database'][video_name]['annotations'][0]['label']
                    if video_class not in self.video_dict.keys():
                        self.video_dict[video_class] = {}
                    self.video_dict[video_class][video_name] = self.annot['database'][video_name]
        elif dataset == 'THUMOS14':
            self.classes = self.ac_classes
            self.label2index = {label:index for index,label in enumerate(self.classes)}
            video_names = os.listdir(self.extract_path)
            for video_name in video_names:
                video_name = video_name[:-4]
                #if self.annot['database'][video_name]['subset'] == task:
                # don't check the subset
                video_class = self.annot['database'][video_name]['annotations'][0]['label']
                if video_class not in self.video_dict.keys():
                    self.video_dict[video_class] = {}
                self.video_dict[video_class][video_name] = self.annot['database'][video_name]


    def get_video(self,video_list):
        videos = []
        for video_name in video_list:
            video_path = os.path.join(self.extract_path,video_name+'.npy')
            feature = np.load(video_path)
            videos.append(feature)
        return videos
    
    def get_labels(self,video_list,video_class):
        labels = []
        for video_name in video_list:
            #labels.append(self.video_dict[video_class][video_name]['annotations'][0]['segment'])
            labels.append(self.video_dict[video_class][video_name]['annotations'])
        return labels

    #todo: may be have repeat class,seed infect the random?
    def __getitem__(self, index):
        # get class name
        videos = []
        labels = []
        # sample or choices
        video_classes = random.choices(self.ac_classes, k=self.args.way)

        for video_class in video_classes:
            video_list = random.choices(list(self.video_dict[video_class].keys()), k=self.args.shot + self.args.query_per_class)
            # print(video_list)
            # get query and support video,only one way
            if self.args.check_video_times:
                for i in video_list:
                    self.video_times[i] += 1

            videos = self.get_video(video_list)
            labels = self.get_labels(video_list,video_class)
            query_feature = np.array(videos[:self.args.query_per_class]).squeeze(0)
            class_label = self.label2index[video_classes[0]]
            segment_label = np.array(labels[:self.args.query_per_class]).squeeze(0)
            numbers_of_segment = len(segment_label)

            # multi instances
            segment_labels = []
            for i in segment_label:
                segment_labels.append(i['segment'])

            
        if self.args.shot == 1:
            support_feature = np.array(videos[self.args.query_per_class:]).squeeze(0)
            support_segment = np.array(labels[self.args.query_per_class:]).squeeze(0)
            support_feature = support_feature[:,int(support_segment[0]):int(support_segment[1])+1,:,:]
            # if support_feature.shape[1] < 10, repeat the feature 10 times
            if support_feature.shape[1] < 10:
                support_feature = np.repeat(support_feature,10,axis=1)

            # support_clip = []
            # for idx,support in enumerate(support_video):
            #     start = int(support_label[idx][0]*self.args.fps)
            #     end = int(support_label[idx][1]*self.args.fps)
            #     support_clip.append(support[start:end])
        else:
            support_feature = []
            for idx,support in enumerate(videos[self.args.query_per_class:]):
                start = int(labels[idx+self.args.query_per_class][0]['segment'][0])
                end = int(labels[idx+self.args.query_per_class][0]['segment'][1])
                support_feature.append(support[:,start:end,:,:])
            support_feature = np.concatenate(support_feature,axis=1)
            if support_feature.shape[1] < 10:
                support_feature = np.repeat(support_feature,10,axis=1)


        return {'vc':torch.tensor(class_label),'qf':torch.tensor(query_feature),'sf':torch.tensor(support_feature),'qsl':torch.tensor(segment_labels),'nos':torch.tensor(numbers_of_segment),'vt':self.video_times}

    def __len__(self):
        return self.args.dataset_len

if __name__ == '__main__':
    from torch import nn
    import matplotlib
    import math
    matplotlib.use('WebAgg')
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    #pca_50 = PCA(n_components=20)
    tsne = TSNE(n_components=2,perplexity=10, n_iter=300,n_jobs=-1,random_state=3407)
    import torch.utils.data as data

    @hydra.main(config_path="config", config_name="config",version_base=None)
    def main(cfg):
        a = data.DataLoader(VideoDataset(cfg),batch_size=1,shuffle=False)
        for i in a:
            print(i['vc'])
            print(i['qf'].shape)
            print(i['sf'].shape)
            print(i['qsl'])
            print(i['vt'])
    main()
