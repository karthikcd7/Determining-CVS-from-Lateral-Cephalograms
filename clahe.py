# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:50:16 2021

@author: Karthik
"""

import cv2
import os
import numpy as np


path = 'dataset/croppedimg/'
import cv2
import numpy as np
save_path = 'dataset/clahe/'


for filename in os.listdir(path):

# Reading the image from the present directory
    image = cv2.imread(path+filename)
   
    # The initial processing of the image
    # image = cv2.medianBlur(image, 3)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
    # The declaration of CLAHE 
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image_bw) + 20
      
    # Showing all the three images
    cv2.imwrite(save_path+filename, final_img)

