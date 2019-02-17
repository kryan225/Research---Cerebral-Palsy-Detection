# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:36:07 2019

Dense Optical Flow Tutorial: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

Calculate dense optical flow of video files and save the images in a new directory. The dense optical
flow can be maped over a folder containing multiple .avi files, resulting in multiple new directories containing
images of the dense optical flow calculations for every frame

@author: kryan
"""

import cv2 as cv
import numpy as np
import os
from os import listdir
from os.path import isfile, join



def capture(vidFile):
    '''
    Function that takes in a video file path and outputs the dense optical flow images for every frame. 
    The new images are exported into a new directory. Set vidFile to 0 for screen capture
    
    Press the 'esc' key to terminate
    '''
    cap = cv.VideoCapture(vidFile)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    x = 0;
    
    newDir = vidFile[:-4] + '_DIR/'
    os.mkdir(newDir)
    while(1):
        ret, frame2 = cap.read()
        
        #converts the image to a different color scheme - in this case changes to grey 
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        
        #flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0) - defaulat vals
        #cv.calcOpticalFlowFarneback (prev, next, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags)
        # - this fn calculates the optical flow for each previous pixel
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # The code below takes the frames and prepares them so they can be seen as an image
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        cv.imshow('frame2',bgr)
        
        #wait for termination key - esc
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        #elif k == ord('s'):
        
        #name1 = newDir + 'opticalfb' + str(x) +'.png'
        name2 = newDir + 'opticalhsv' + str(x) + '.png'
        #cv.imwrite(name1,frame2) - this is the original image
        cv.imwrite(name2,bgr) # write the dense optical flow image to the new directory
        x = x + 1
        prvs = next
    cap.release()
    cv.destroyAllWindows()
    
    
def batchCapture(path):
    '''
    This function takes in a directory path, that should contain multiple .avi files and will map the above
    function over every file. This will result in numerous new directories, each containing the dense optical flow
    images for a video in the specified path
    '''
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for video in files:
        if(video[-3:] == 'avi'):
            capture(video)
        else:
            print('ERROR - Video file: ' + video + ' not of type .avi. Could not calculate dense optical flow\n')