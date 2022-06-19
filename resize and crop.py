# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:56:06 2021

@author: Karthik
"""

import cv2
import os
path = 'dataset/images/raw images/'
save_path = 'dataset/images/roi/'


for filename in os.listdir(path):
    imageread = cv2.imread(path+filename)
    image = imageread[1300:imageread.shape[0]-100, 20:620]
    imagedraw = cv2.selectROI(image)
    x = imagedraw[0]
    y = imagedraw[1]
    w = imagedraw[2]
    h = imagedraw[3]
    if h<10:
        continue
    canvas = image.copy()
    cx = x+w//2
    cy = y+h//2
    cr  = max(w,h)//2
    r = cr
    croped = image[cy-r:cy+r, cx-r:cx+r]
    if cx-r<0:
        croped = image[cy-r:cy+r, 0:cx+r+abs(cx-r)]
    resized = cv2.resize(croped, (500,500), interpolation = cv2.INTER_AREA)
    '''
    try:    
        #resized = cv2.resize(croped, (500,500), interpolation = cv2.INTER_AREA)
    except:
        #croped = image[cy-r:cy+r, 0:cx+r+abs(cx-r)]
        resized = cv2.resize(croped, (500,500), interpolation = cv2.INTER_AREA)
        pass
    '''
    cv2.imwrite(save_path + filename[:-4]+".png", resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    
'''   
filename = 'LC (517).jpg'
imageread = cv2.imread(path+filename)
image = imageread[1200:imageread.shape[0]-100, 20:620]
imagedraw = cv2.selectROI(image)
x = imagedraw[0]
y = imagedraw[1]
w = imagedraw[2]
h = imagedraw[3]
canvas = image.copy()
cx = x+w//2
cy = y+h//2
cr  = max(w,h)//2
r = cr
croped = image[cy-r:cy+r, cx-r:cx+r]
print(croped.shape)
try:    
    resized = cv2.resize(croped, (500,500), interpolation = cv2.INTER_AREA)
except:
    croped = image[cy-r:cy+r, 0:cx+r+abs(cx-r)]
    resized = cv2.resize(croped, (500,500), interpolation = cv2.INTER_AREA)
    pass

cv2.imwrite(save_path + filename[:-4]+".png", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()  
'''
'''
for filename in os.listdir(path):
    image = cv2.imread(path+filename)
    #resized = cv2.resize(image, (1800,2100), interpolation = cv2.INTER_AREA)
    #cv2.imwrite(save_path + 'resized/' + filename[:-4]+" resized.png", resized)
    cropped = image[1200:image.shape[2], 0:650]
    cv2.imshow(cropped)
    cv2.imwrite(save_path + filename, cropped)
'''
    
    