#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import os.path
import json

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import cv2 as cv
import pickle
import sys
import glob
import tqdm
from deepface import DeepFace

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

save_path = sys.argv[1]
frames_path = sys.argv[2]

# res = np.loadtxt(save_path+'/short_id.txt', delimiter=' ',dtype=int)
res = np.loadtxt(save_path+'/track_id.txt', delimiter=' ', dtype=int)

#res = np.loadtxt('../../tracking/short_term_tracking/build/result.txt', delimiter=',')


if len(res.shape) == 1:
    res = np.expand_dims(res, axis=0)

box = {}

for n in range(res.shape[0]):
    if not (res[n][0] in box):
       box[res[n][0]] = []
    box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4]+res[n][2], res[n][5]+res[n][3]])
    #box[res[n][0]].append([res[n][1], res[n][2], res[n][3], res[n][4], res[n][5]])




flist = glob.glob(os.path.join(frames_path, '*.jpg'))
flist.sort()

print(len(flist))
# exit(0)

# fname = save_path+'/video.avi'
# feature = {}

# cap = cv.VideoCapture(fname)
# # # 获取视频的总帧数
# total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# print(f"Total frames: {total_frames}")

# # 释放视频对象
# cap.release()

# exit(0)

frame_num = 0

sys.stdout.write('\033[K')
feature = {}
for frame_num, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
    frame = cv.imread(fname)
    if frame is None:
        break
    print(frame_num, end = '\r')
    if frame_num in box:
        b = box[frame_num]
        for k in range(len(b)):
            pid = int(b[k][0])
            x1 = int(b[k][1])
            y1 = int(b[k][2])
            x2 = int(b[k][3])
            y2 = int(b[k][4])

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= frame.shape[1]:
                x2 = frame.shape[1]-1
            if y2 >= frame.shape[0]:
                y2 = frame.shape[0]-1 

            if x1 >= x2 or y1 >= y2:
                continue

            if pid not in feature:
                feature[pid] = []
            imcrop = frame[y1:y2,x1:x2,:]
            # imcrop = frame[x1:x2,y1:y2,:]

            # print(imcrop.shape)
            # exit(0)
            # if frame_num==218:
            #     cv.imwrite('face_'+str(frame_num)+'_'+str(k)+'.jpg', imcrop)
            #embeddings
            embedding_objs = DeepFace.represent(
            img_path = imcrop,
            model_name = models[0],
            enforce_detection=False,
            )
            output = embedding_objs[0]['embedding']
            # output = output.data.cpu().numpy().squeeze().tolist()

            feature[pid].append([frame_num, x1, y1, x2, y2] + output)

    frame_num = frame_num + 1       


with open('traj.pkl', 'wb') as f:     #以二进制读写方式打开，如果文件不存在，则创建新文件
      pickle.dump(feature, f, pickle.HIGHEST_PROTOCOL)
