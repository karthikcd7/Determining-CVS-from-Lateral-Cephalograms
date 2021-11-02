# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:56:06 2021

@author: Karthik
"""

import cv2
import os
path = 'dataset/original images/'
save_path = 'dataset/'
for filename in os.listdir(path):
    image = cv2.imread(path+filename)
    resized = cv2.resize(image, (1800,2100), interpolation = cv2.INTER_AREA)    
    cv2.imwrite(save_path + 'resized/' + filename[:-4]+" resized.png", resized)
    cropped = resized[1200:2100, 0:650]
    cv2.imwrite(save_path + 'croppedimg/' + filename, cropped)


    
