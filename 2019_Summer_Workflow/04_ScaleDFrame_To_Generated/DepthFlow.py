# -*- coding: utf-8 -*-
"""
Dense Optical Flow Tutorial: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

Calculate depth flow of video files and save the images in a new directory. The depth
flow can be mapped over a folder containing multiple .avi files, resulting in multiple 
new directories containing images of the depth flow calculations for every frame

Modeled after dense_optical_flow.py by kryan

@author: nyoung
"""
from secret import credentials
import pandas as pd
import cv2 as cv
import numpy as np
import os
import pymysql

def connect():
    '''
    Forms a connection to the amazon RDS server using my login credentials
    '''
    db_host = credentials['db_host'];
    db_port = credentials['db_port'];
    db_name = credentials['db_name'];
    db_username = credentials['db_username']
    db_password = credentials['db_password']
    conn = pymysql.connect(db_host, user=db_username, port=db_port, passwd=db_password, db=db_name)
    return conn
     

def read_file(filename):
    '''
    reads a file from a local directory
    '''
    with open(filename, 'rb') as f:
        photo = f.read()
    return photo
    

def sql_send(directory, conn=connect()):
    '''
    Sends Depth Flow images from a directory to the amazon RDS
    '''
    curs = conn.cursor()    
    
    #query for all frames that do not have a processed update time
    null_update_ids = []
    check_query = "SELECT raw_id, D_Processed_Updated FROM Video_Generated WHERE D_Processed_Updated IS NULL order by raw_id"
    try:
        curs.execute(check_query)
        for x in curs.fetchall(): 
            null_update_ids.append(x[0])
    except Exception as e:
        print("Failure selecting null update time ids from Video_Generated",e)
        conn.rollback()
        raise e
        
    #query for all frames that do have a processed update time
    not_null_update_ids = []
    check_query = "SELECT raw_id, D_Processed_Updated FROM Video_Generated WHERE D_Processed_Updated IS NOT NULL order by raw_id"
    try:
        curs.execute(check_query)
        for x in curs.fetchall(): 
            not_null_update_ids.append(x[0])
    except Exception as e:
        print("Failure selecting update time ids from Video_Generated",e)
        conn.rollback()
        raise e
    
    
    for f in os.listdir(directory):
        
        #find processed depth frame images
        if 'processed' in f:
            currentID = int(f[:-14])
            print(currentID)
            fname = directory + '/' + f
            
            #find current state of the image in the database
            null_present = null_update_ids.count(currentID)
            not_null_present = not_null_update_ids.count(currentID)
            present = null_present + not_null_present
        
            #check for duplicate entries and delete any that exist
            if present > 1:
                delete_query = "DELETE FROM Video_Generated WHERE (raw_id = %s)"
                try:
                    curs.execute(delete_query,(currentID))
                    conn.commit()
                    null_present = 0
                    not_null_present = 0
                    present = null_present + not_null_present
                except Exception as e:
                    print("Failure deleting multiple ids from Video_Generated",e)
                    conn.rollback()
                    raise e
        
            #create an entry if there is not one present
            if not_null_present == 0:
                if present == 0:
                    insert_query = "INSERT INTO Video_Generated (raw_id) VALUES (%s)"
                    try:
                        curs.execute(insert_query, (currentID))
                        conn.commit()
                        null_present = 1
                        present = null_present + not_null_present
                    except Exception as e:
                        print('Failure inserting raw_id')
                        conn.rollback()
                        raise e
                
                #update the database with the scaled frame and the current time
                update_query = "UPDATE Video_Generated SET D_Processed = %s, D_Processed_Updated = NOW() WHERE (raw_id = %s)"
                
                try:
                    photo=read_file(fname)
                    curs.execute(update_query, (photo, currentID))
                    conn.commit()
                except Exception as e:
                    print("Failure updating scaled photo",e)
                    conn.rollback() 
                    raise e
    
   
def depthFrameCapture(images, outFolder):
    '''
    Takes in a list of frames and writes an optical flow image to the local directory outFolder
    '''
    #get all the frames into a list
    frameList=[]
    for image in images:
        frame=cv.imread(image,cv.IMREAD_ANYDEPTH)
        frameList.append(frame[...]) 

    #stack the frames into a 3-D array and compute standard deviation
    stacked=np.dstack(frameList)    
    generated=np.std(stacked, axis=2)   
    generated=generated.astype(np.uint16)
    
    #set the correct path for the depth frame and save the image to the local directory
    name=(images[len(images)-1]).replace('scale','processed') 
    print('Depth Frame captured for', name)
    cv.imwrite(name,generated) 
    return generated
    
def sql_write(outFolder, rec_ids, conn=connect()):
    '''
    Connects to Amazon RDS and creates depth flow images from Video_Raw
    Saves to local directory
    '''    
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
        
    curs=conn.cursor()
    for i in rec_ids:
        firstQuery='SELECT timestamp FROM Video_Raw WHERE (recording_id = %s) ORDER BY timestamp LIMIT 1'
        try:
            curs.execute(firstQuery, (str(i)))
            for row in curs.fetchall():
                startTime=row[0]
        except Exception as e:
            print("Failure retreaving first timestamp",e)
            conn.rollback()
            raise e
        capture=True
        timeQuery = 'SELECT id, timestamp FROM nicu.Video_Raw WHERE (recording_id = %s) AND (timestamp BETWEEN %s AND date_add(%s, INTERVAL 1 SECOND)) ORDER BY timestamp'          
        while (capture):
            try:
                curs.execute(timeQuery, (str(i), str(startTime), str(startTime)))
                results=curs.fetchall()
                currentOutput=outFolder+'/'+str(results[len(results)-1][0])+'.processed.png'
                print('Generating',currentOutput )
                if (len(results)>2):
                    #if the output already exists in the cache, set a new start time 
                    if not os.path.exists(currentOutput):  
                        names=[]
                        timestamps=[]
                        
                        #collect the id and timestamps of query data
                        for row1 in results: 
                            idn=row1[0]
                            
                            currentInput=outFolder+'/'+str(idn)+'.scale.png'
                            names.append(currentInput)
                            timestamps.append(row1[1])
                            
                                #if the scaled image is not in the cache, download it
                            if not os.path.exists(currentInput):    
                                print("downloading",str(currentInput))
                                depthQuery = 'SELECT D_Scaled FROM nicu.Video_Generated WHERE (raw_id=%s)'
                                curs.execute(depthQuery, (str(idn)))
                                for row2 in curs.fetchall():
                                    if (row2[0]!= None):
                                        img=cv.imdecode(np.asarray(bytearray(row2[0]),dtype=np.uint8),cv.IMREAD_ANYDEPTH)
                                        cv.imwrite(currentInput,img)
                                    else:
                                        print('ID',idn,'is missing D_Scaled')
                        depthFrameCapture(names, outFolder)
                    startTime=results[1][1]
                else:
                    allTimeQuery = 'SELECT timestamp FROM nicu.Video_Raw WHERE (recording_id = %s) AND (timestamp > %s) ORDER BY timestamp'
                    curs.execute(allTimeQuery, (str(i), str(startTime)))
                    results=curs.fetchall()
                    if (len(results) > 3):
                        startTime=results[1][0]
                    else:
                        capture=False
            except Exception as e:
                print("Something failed",e)
                raise e
    print('All file downloads complete')

    
def main(): 
    
    conn=connect()
    try:
        recordingIDs=range(2,14)
        sql_write("cache",recordingIDs, conn)
        sql_send("cache")
        
    finally:
        conn.close()
    
main()

