# -*- coding: utf-8 -*-
"""
The goal here is to go through a set of RGB frames and calculate dense optical
flow image from them

Dense Optical Flow Tutorial: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

Calculate dense optical flow of video files and save the images in a new directory. The dense optical
flow can be maped over a folder containing multiple .avi files, resulting in multiple new directories containing
images of the dense optical flow calculations for every frame

@author: kryan,djp3

pip install opencv-python-headless
"""

import pandas as pd
import cv2 as cv
import numpy as np
import os
import pymysql
from os import listdir
from os.path import isfile, join

#Secrets shouldn't be in the repository
from secrets import credentials
# Of the form
#credentials = {
#        'db_host' : 'something.us-east-1.rds.amazonaws.com'
#        'db_port' : 3306
#        'db_name' : 'name',
#        'db_username' : 'something',
#        'db_password' : 'secret'
#        }
        
#Forms a connection to the database server using login credentials stored in credentials
def connect():
    db_host = credentials['db_host'];
    db_port = credentials['db_port'];
    db_name = credentials['db_name'];
    db_username = credentials['db_username']
    db_password = credentials['db_password']
    conn = pymysql.connect(db_host, user=db_username, port=db_port, passwd=db_password, db=db_name)
    return conn

'''
Function that takes in two frames and outputs the dense optical flow image for them
'''
def optical_flow_calculate(in_files, out_file, cache_path, base_flow):

    if len(in_files) < 2:
        raise Exception("Not enough files to calculate optical flow from: {}".format(in_files))

    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    frame1 = cv.imread(in_files[0])
    previous_frame = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    frame2 = cv.imread(in_files[1])
    next_frame = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)


    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    #documentation: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
    pyr_scale = 0.5
    levels = 3
    winsize = 8
    iterations = 3
    poly_n = 7
    poly_sigma = 1.5
    flags = cv.OPTFLOW_USE_INITIAL_FLOW | cv.OPTFLOW_FARNEBACK_GAUSSIAN

    if base_flow is None:
        base_flow = cv.calcOpticalFlowFarneback(prev=previous_frame,next=next_frame,flow=base_flow, pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)

    flow = cv.calcOpticalFlowFarneback(prev=previous_frame,next=next_frame,flow=base_flow, pyr_scale=pyr_scale, levels=levels, winsize=winsize, iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=flags)
    
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)

    cv.imwrite(cache_path+out_file,bgr)
    cv.destroyAllWindows()
    return base_flow




#Finds a selection of images indicated by a collection of recording_ids
#Store images in the cache_directory for processing
def process_rgb_frames(conn, recording_ids, cache_path):
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        
    cursor = conn.cursor()

    print("Collecting images for processing (~ = solution already cached, Σ = solution calculated, ⇣ = source image fetched from db, m = multiple source images in db, x = source image not in db)")

    for r_id in recording_ids:
        #Get the first relevant timestamp
        print("")
        print("Analyzing recording_id:",r_id,": ",end="")
        all_time_stamps = []
        firstTimeQuery='SELECT timestamp FROM Video_Raw WHERE (recording_id = %s) GROUP BY timestamp ORDER BY timestamp'
        try:
            cursor.execute(firstTimeQuery,(r_id))
            for row in cursor.fetchall():
                all_time_stamps.append(row[0])
        except Exception as e:
            print("Failure retrieving first timestamp",e)
            conn.rollback()
            raise e

        startTime=all_time_stamps.pop(0)
        capture=True
        base_flow=None
        timeWindowQuery = 'SELECT id, timestamp FROM Video_Raw WHERE (recording_id = %s) AND (timestamp BETWEEN %s AND date_add(%s, INTERVAL 1 SECOND)) ORDER BY timestamp LIMIT 2'          
        while (capture):
            try:
                cursor.execute(timeWindowQuery,(r_id, startTime, startTime))
                results=cursor.fetchall()
                if (len(results) > 1):
                    currentOutput=cache_path+str(results[1][0])+'.oflow.png'
                    print("Working on",currentOutput)
                    #if the output already exists in the cache, set a new start time 
                    if not os.path.exists(currentOutput):  
                        print("\tDoes not exist")
                        names=[]
                        timestamps=[]
                        
                        #collect the id and timestamps from the query data
                        for row in results: 
                            raw_id = row[0]
                            currentInput=cache_path+str(raw_id)+'.rgb.png'
                            names.append(currentInput)
                            timestamps.append(row[1])
                            
                            #if the rgb image is not in the cache, download it
                            if not os.path.exists(currentInput):    
                                print("\t\tSource image does not exist",currentInput)
                                #get the image from the database
                                cursor2 = conn.cursor()
                                rawQuery = 'SELECT RGB_frame FROM Video_Raw WHERE (id=%s)'
                                cursor2.execute(rawQuery, (raw_id))
                                print("\t\t\tThere are",cursor.rowcount,"images")
                                if cursor.rowcount == 0:
                                    print("x",end="",flush=True)
                                else:
                                    if cursor.rowcount > 1:
                                        print("m",end="",flush=True)
                                    for row2 in cursor2.fetchall():
                                        db_img = row2[0]
                                        if db_img is not None:
                                            img=cv.imdecode(np.asarray(bytearray(db_img),dtype=np.uint8),cv.IMREAD_ANYDEPTH)
                                            #Save the image
                                            cv.imwrite(currentInput,img)
                                            print("⇣",end="",flush=True)
                                            print("\t\t\t\tJust wrote",currentInput)
                                        else:
                                            print("\t\t\t\tJust couldn't write",currentInput)
                                            print("x",end="",flush=True)
                            else:
                                print("\t\tSource image exists",currentInput)
                        base_flow = optical_flow_calculate(names,currentOutput,cache_path,base_flow)
                        print("Σ",end="",flush=True)
                    else:
                        print("~",end="",flush=True)
                else:
                    base_flow=None

                if len(all_time_stamps) > 0:
                    startTime = all_time_stamps.pop(0)
                else:
                    capture=False
            except Exception as e:
                print("Something failed",e)
                raise e
    print("")


    
    
#Upload any processed images stored in the cache directory to the database
def store_processed_images(conn,cache_path):
    cursor = conn.cursor()    
    print("Storing the optical flow images (# = deleting multiple db entries, o = placeholder created in db, ⇡ = uploaded to db , ~ = db not updated b/c image already present with date)")
    print("")
    
    #Get all the ids that are currently present in the db with no update time
    null_update_ids = []
    check_query = "SELECT raw_id FROM Video_Generated WHERE RGB_Optical_Flow_Updated IS NULL order by raw_id"
    try:
        cursor.execute(check_query)
        for x in cursor.fetchall(): 
            null_update_ids.append(x[0])
    except Exception as e:
        print("Unable to select ids from Video_Generated with null update time",e)
        conn.rollback()
        raise e

    #Get all the ids that are currently present in the db with an update time
    not_null_update_ids = []
    check_query = "SELECT raw_id, RGB_Optical_Flow_Updated FROM Video_Generated WHERE RGB_Optical_Flow_Updated IS NOT NULL order by raw_id"
    try:
        cursor.execute(check_query)
        for x in cursor.fetchall(): 
            not_null_update_ids.append(x[0])
    except Exception as e:
        print("Unable to select ids from Video_Generated with not null update time",e)
        conn.rollback()
        raise e

    #Try and insert all the images from the cache that are of the pattern "*oflow.png"
    for f in sorted(os.listdir(cache_path)):
        if "oflow" in f:
            currentID = int(f[:-10])
            file_name = cache_path + f
            print("Storing:",currentID,"filename:",file_name)

            null_present = null_update_ids.count(currentID)
            not_null_present = not_null_update_ids.count(currentID)
            present = null_present + not_null_present

            #If there are multiple entries then remove them all
            if present > 1:
                delete_query = "DELETE FROM Video_Generated WHERE (raw_id = %s)"
                try:
                    cursor.execute(delete_query,(currentID))
                    conn.commit()
                    print(present,end="",flush=True)
                    null_present = 0
                    not_null_present = 0
                    present = null_present + not_null_present
                except Exception as e:
                    print("Unable to delete multiple ids from Video_Generated",delete_query,e)
                    conn.rollback()
                    raise e

            #Check to see if there is already an image in the database with a timestamp
            if not_null_present == 0:
                #Make sure there is an entry to put the photo in
                if present == 0:
                    insert_query = "INSERT INTO Video_Generated (raw_id) VALUES (%s)"
                    try:
                        #insert the raw_id
                        cursor.execute(insert_query, (currentID))
                        conn.commit()
                        null_present = 1
                        present = null_present + not_null_present
                        print("o",end="",flush=True)
                    except Exception as e:
                        print("Unable to make an entry for:",insert_query,file_name)
                        conn.rollback()
                        raise e

                #update the database with the processed frame and the current time
                update_query = "UPDATE Video_Generated SET RGB_Optical_Flow = %s, RGB_Optical_Flow_Updated = NOW() WHERE (raw_id = %s)"
                try:
                    with open(file_name, 'rb') as temp_f:
                        photo = temp_f.read()
                    cursor.execute(update_query, (photo, currentID))
                    conn.commit()
                    print("⇡",end="",flush=True)
                except Exception as e:
                    print("Unable to store scaled photo in the db",file_name)
                    conn.rollback() 
                    raise e
            else:
                print("~",end="",flush=True)
    print("")


            
    
def main():
    #sql_write("test","dummy")

    #open connection to database
    conn = connect()
    try:
        #identify the targets
        recording_ids = range(2,14)

        process_rgb_frames(conn,recording_ids,"cache/")
        store_processed_images(conn,"cache/")
    finally:
        conn.close()
    print("Don't forget to erase the cache files!");


main()
