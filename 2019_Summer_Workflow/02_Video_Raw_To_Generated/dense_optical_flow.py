# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:36:07 2019

Dense Optical Flow Tutorial: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

Calculate dense optical flow of video files and save the images in a new directory. The dense optical
flow can be maped over a folder containing multiple .avi files, resulting in multiple new directories containing
images of the dense optical flow calculations for every frame

@author: kryan

pip install opencv-python-headless
"""

import pandas as pd
import cv2 as cv
import numpy as np
import os
import pymysql
from os import listdir
from os.path import isfile, join


        

def connect():
    '''
    Forms a connection to the amazon RDS server using my login credentials
    '''
    host = 'nicu-2019-03-05.c2lckhwrw1as.us-east-1.rds.amazonaws.com'
    port = 3306
    dbname = 'nicu'
    user = 'ryan'
    password = 'nicu_ryan'
    conn = pymysql.connect(host, user=user, port=port, passwd=password, db=dbname)
    return conn

def frameCapture(f1, f2, outFolder, name):
    '''
    Function that takes in two frames and outputs the dense optical flow image for them
    '''
    frame1 = cv.imread(f1)
    frame2 = cv.imread(f2)
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
        
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)

    name2 = outFolder + '/' + name + '.png'
    cv.imwrite(name2,bgr)
    cv.destroyAllWindows()
    
    

#reads a photo by taking in the filename
def read_file(filename):
    with open(filename, 'rb') as f:
        photo = f.read()
    return photo
    
    
    
    '''TO DO:
    
    CHECK TO MAKE SURE THAT DATA IS IN FOR CURRENT RAW_ID
    IF DATA ALREADY EXISTS FOR RAW_ID THEN DO NOTHING
    
    If given a video, parse into frames and then post to video_raw
    then do optical flow on it and update to video_generated
    '''
def sql_send(directory, con=connect()):
    '''
    Sends Optical Flow images from a directory to the amazon RDS
    
    CANT INSERT BLOBS - MUST UPDATE THEM INTO DATABASE
    '''
    curs = con.cursor()    
    
    for f in os.listdir(directory):
        currentID = int(f[:-4])
        print(currentID)
        fname = directory + '/' + f
        photo = read_file(fname)
        #sql command:
        insert_query = """INSERT INTO Video_Generated (raw_id) VALUES (%s)"""
        try:
            #insert the raw_id
            curs.execute(insert_query, (currentID))
            con.commit()
            print("success inserting raw_id")
        except:
            con.rollback()
            print("failure inserting raw_id")
            
            
        update_query = """UPDATE Video_Generated SET RGB_OpticalFlow = %s WHERE (raw_id = %s)"""
        try:
           # Execute the SQL command
           curs.execute(update_query, (photo, currentID))
           # Commit your changes in the database
           con.commit()
           print("success updating photo")
        except:
           # Rollback in case there is any error
           con.rollback() 
           print("failure updating photo")

            
    
'''

For the sql calls, write each RGB_frame to local using file.open('name', 'wb') and file.write(df[rgb_frame])
Then call frameCapture on the 2 frames
Then delete the file with os.remove('name')

'''
def sql_write(outFolder, name):
    '''
    Connects to Amazon RDS and creates dense optical flow images from Video_Raw
    Saves to local directory
    '''
    conn = connect()
    #djp3 editted below
    df = pd.read_sql('Select id, timestamp, RGB_frame from Video_Raw order by timestamp limit 10', con=conn)
    rawIds = pd.read_sql('Select raw_id from Video_Generated', con=conn)
    rawIds = rawIds['raw_id'].unique()
    written = []
    next = 'dumbyVar'
    for index, row in df.iterrows():
        idn = row['id']
        if(idn not in rawIds):
            written.append(idn)
            n = str(idn) + '.png'
            f = open(n, 'wb')
            f.write(row['RGB_frame'])
            f.close()
        
    for x in range(len(written) - 1):
        current = str(written[x]) + '.png'
        next = str(written[x + 1]) + '.png'
        nm = str(written[x + 1])# + '-' + str(df['id'][x + 1]) + '_OF'
        frameCapture(current, next, outFolder, nm)
        if os.path.exists(current):
            os.remove(current)
    if os.path.exists(next):
        os.remove(next)
    
    conn.close()
    return written
    
def capture(vidFile, outFolder, prefix):
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
    
    os.mkdir(outFolder)
    while(1):
        
        ret, frame2 = cap.read()
        if not ret:
            break
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
        
        #name1 = outFolder + '/' + 'opticalfb_' + str(x) +'.png'
        name2 = outFolder + '/' + prefix + str(x) + '.png'
        #cv.imwrite(name1,frame2) #- this is the original image
        cv.imwrite(name2,bgr) # write the dense optical flow image to the new directory
        x = x + 1
        prvs = next
    cap.release()
    cv.destroyAllWindows()
    
    
def parseLine(line):
    '''
    This function parses a line that has been read from a txt file and returns the strings between commas in a list
    '''
    ret = []
    lastC = -1
    
    #iterate through each character in the line
    for i in enumerate(line):     
        
        #if we've found a comma record and add the previous substring to the return list
        if(i[1] == ','):
            ret.append(line[lastC+1:i[0]])
            lastC = i[0]
            
    #we need to check if there is a \n at the end of the line
    if(line[-1:] == '\n'):
        end = -1
    else:
        end = len(line)
    ret.append(line[lastC+1:end])#avoid the \n character for a new line if it's there
    return ret
    
    
def batchCapture(textFile):
    '''
    This function takes in a txt file, which contains lines according to the format:
        path to a video file, desired output folder name, image prefix. 
    It then parses the lines and calculates,saves the dense optical flow images for each video in 
    the txt file. 
    '''
    f = open(textFile)
    lines = f.readlines()
    for line in enumerate(lines):
        parsed = parseLine(line[1])
        capture(parsed[0], parsed[1], parsed[2])
            
            
def main():
    sql_write("test","dummy")


main()
