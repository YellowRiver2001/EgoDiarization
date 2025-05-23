#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import cv2 as cv
import numpy as np
import pickle
import sys


save_path = sys.argv[1]

with open('result_group_fast.pkl', 'rb') as handle:
    feature = pickle.load(handle)


fname = save_path+'/video.avi'

cap = cv.VideoCapture(fname)
total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#print(fname, total)

sys.stdout.write('\033[K')

frame_num = 0
box = []
for frame_num in range(total):
    print(frame_num, end = '\r')
    for pid in feature:
        ff = np.array(feature[pid])
        if frame_num in ff[:,0]:
           index = int(np.where(ff[:,0] == frame_num)[0]) 
           b = feature[pid][index]
           x1 = int(b[1])
           y1 = int(b[2])
           x2 = int(b[3])
           y2 = int(b[4])
          
           box.append([frame_num, pid, x1, y1, x2, y2])
        
np.savetxt(save_path+'/global_id_new.txt', box, fmt='%d')  
