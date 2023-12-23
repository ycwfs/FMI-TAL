from cProfile import label
import re
import cv2
import hydra
from scipy import rand
from sklearn.manifold import TSNE
import torch.utils.data as data
import torch
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
        self.an_classes = self.params.an_classes
        if self.args.dataset == 'ActivityNet':
            self.extract_path = self.args.data_paths.ActivityNet.extracted
            self.annotation = self.args.data_paths.ActivityNet.annotation
            self.action_name = self.args.data_paths.ActivityNet.action_name
            self.origianl_video = self.args.data_paths.ActivityNet.original
        elif self.args.dataset == 'THUMOS14':
            self.extract_path = self.args.data_paths.THUMOS14.extracted
            self.annotation = self.args.data_paths.THUMOS14.annotation
            
        self.annot = json.load(open(self.annotation,'r'))
        self.video_dict = {}
        # get task specific data dict, for each class choose one query and 1 or 5 support videos
        self.get_fs_data(self.args.task)

        # self.class2idx = {c:i for i,c in enumerate(self.an_classes)}

        self.classes = pd.read_csv(self.action_name)
        self.label2index = {label:index for index,label in enumerate(self.classes['action'])}

        # 200
        # self.classes = list(self.video_dict.keys())
        # self.class2idx = {c:i for i,c in enumerate(self.classes)}

    def get_fs_data(self,task):
        video_names = os.listdir(self.extract_path)
        for video_name in video_names:
            video_name = video_name[:-4]
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
            labels.append(self.video_dict[video_class][video_name]['annotations'][0]['segment'])
        return labels

    #todo: may be have repeat class,seed infect the random?
    def __getitem__(self, index):
        # get class name
        videos = []
        labels = []
        video_classes = random.sample(self.an_classes, self.args.way)
        for video_class in video_classes:
            video_list = random.sample(self.video_dict[video_class].keys(), self.args.shot + self.args.query_per_class)
            # print(video_list)
            # get query and support video,only one way
            videos = self.get_video(video_list)
            labels = self.get_labels(video_list,video_class)
        if self.args.shot == 1:
            query_feature = np.array(videos[:self.args.query_per_class]).squeeze(0)
            support_feature = np.array(videos[self.args.query_per_class:]).squeeze(0)
            class_label = self.label2index[video_classes[0]]
            segment_label = np.array(labels[:self.args.query_per_class]).squeeze(0)
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
            
            return {'vc':torch.tensor(class_label),'qf':torch.tensor(query_feature),'sf':torch.tensor(support_feature),'qsl':torch.tensor(segment_label)}

    def __len__(self):
        return 10000

if __name__ == '__main__':
    from torch import nn
    import matplotlib
    import math
    matplotlib.use('WebAgg')
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    #pca_50 = PCA(n_components=20)
    tsne = TSNE(n_components=2,perplexity=10, n_iter=300,n_jobs=-1,random_state=3407)


    @hydra.main(config_path="config", config_name="config",version_base=None)
    def main(cfg):
        a = VideoDataset(cfg)
        for i in a:
            print(i['vc'])
            print(i['qf'][0].shape)
            print(i['sf'][0].shape)
            print(i['qsl'])

            # for j in i['sf']:
            #     print(j.shape)
        #     print(i['qv'][0].shape)
        #     print(i['ql'][0])
        #     video = i['qv'][0]
        #     # video = np.expand_dims(video.transpose(3,0,1,2),0)
        #     # video = c3d(torch.from_numpy(video))
        #     print(video.shape)
            # start = int(int(i['ql'][0][0])/30)
            # end = int(int(i['ql'][0][1])/30)


            # start = int(i['ql'][0][0])
            # end = int(i['ql'][0][1])
            # print(start,end)
            # label = np.zeros(i['qf'][0].shape[0])
            # label[start:end] = 1
            # #pca_result_50 = pca_50.fit_transform(i['qf'][0])
            # result = tsne.fit_transform((i['qf'][0]))
            # plt.scatter(result[:,0],result[:,1],c=label,s=5,cmap='Spectral')
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.savefig('featuretsne.png')

            # after = sp(video.reshape(video.shape[0],-1,3))
            # pca_result_50 = pca_50.fit_transform(after.reshape(after.shape[0],-1).detach().numpy())
            # result = tsne.fit_transform(pca_result_50)
            # plt.scatter(result[:,0],result[:,1],c=label,s=5,cmap='Spectral')
            # plt.gca().set_aspect('equal', 'datalim')
            # plt.savefig('after.png')
            break
    main()