# -*- coding: utf-8 -*-
"""
Calculate a new kind of image called a "depth flow" from depth files generated
by the D in RGB-D cameras.
The original images are taken from a database, stored locally, processed, stored
locally, then put in the database.

In main you specify the complete set of recording_ids that you are processing 
over.  The code loops through all of those and calculates the depth flow images
from them.  It looks in a local cache directory, 'cache' for the images
before asking the database for them.  If it does need an image from the database
it stores it in the cache. Make sure existing files in the cache are good or you
will be processing bad data.

From those images we create an image in which each pixel is the standard
deviation of the same pixel in images within a window of 1 second in time.
A version of each processed image is also stored it in the cached
directory in the format <raw_id>.dflow.png.  If the file already exists, then it
is not recalculated or replaced. So make sure those are good.
Then the code goes through the cache directory and uploads any files that have
"dflow" in the name.  It does not update any entries in the database unless the
"D_Depth_Flow_Updated" field is NULL.
Clear the D_Depth_Flow_Updated field with a query like:

    UPDATE Video_Raw JOIN Video_Generated ON Video_Raw.id = Video_Generated.raw_id SET Video_Generated.D_Depth_Flow_Updated = NULL WHERE Video_Raw.recording_id >= 2;

@author: nyoung,djp3
"""
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

import cv2 as cv
import numpy as np
import os
import pymysql


#Forms a connection to the database server using login credentials stored in credentials
def connect():
    db_host = credentials['db_host'];
    db_port = credentials['db_port'];
    db_name = credentials['db_name'];
    db_username = credentials['db_username']
    db_password = credentials['db_password']
    conn = pymysql.connect(db_host, user=db_username, port=db_port, passwd=db_password, db=db_name)
    return conn

#Takes in a list of frames and writes depth flow image to the local
#directory cache_path
def depth_frame_calculate(images,currentOutput,cache_path):
    #get all the frames into a list
    frameList=[]
    for image in images:
        frame=cv.imread(image,cv.IMREAD_ANYDEPTH)
        try:
            frameList.append(frame[...]) 
        except Exception as e:
            print(e, image, currentOutput, cache_path)

    #stack the frames into a 3-D array and compute standard deviation
    stacked=np.dstack(frameList)    
    generated=np.std(stacked, axis=2)   
    generated=generated.astype(np.uint16)
    
    #set the correct path for the depth frame and save the image to the local directory
    cv.imwrite(currentOutput,generated) 
    
#Finds a selection of images indicated by a collection of recording_ids
#Store images in the cache_directory
def process_d_frames(conn, recording_ids, cache_path):
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        
    cursor = conn.cursor()
    print("Collecting images for processing (~ = solution already cached, Σ = solution calculated, ⇣ = source image fetched from db, x = source image not in db)")
    for r_id in recording_ids:
        #Get relevant timestamps
        print("")
        print("Analyzing recording_id:",r_id,": ",end="")
        all_time_stamps = []
        firstTimeQuery='SELECT timestamp FROM Video_Raw WHERE (recording_id = %s) GROUP BY timestamp ORDER BY timestamp ASC '
        try:
            cursor.execute(firstTimeQuery,(r_id))
            for row in cursor.fetchall():
                all_time_stamps.append(row[0])
        except Exception as e:
            print("Failure retrieving first timestamp",e)
            conn.rollback()
            raise e

        startTime = all_time_stamps.pop(0)
        print(startTime)
        raise Exception("Stop")
        capture=True
        timeWindowQuery = 'SELECT id, timestamp FROM Video_Raw WHERE (recording_id = %s) AND (timestamp BETWEEN date_sub(%s,INTERVAL 1 SECOND) AND %s) ORDER BY timestamp'          
        while (capture):
            try:
                cursor.execute(timeWindowQuery,(r_id, startTime, startTime))
                results=cursor.fetchall()
                if (len(results) > 2):
                    currentOutput=cache_path+str(results[len(results)-1][0])+'.dflow.png'
                    #print("Working on",currentOutput)
                    #if the output already exists in the cache, set a new start time 
                    if not os.path.exists(currentOutput):  
                        #print("\tDoes not exist")
                        names=[]
                        timestamps=[]
                        
                        #collect the id and timestamps from the query data
                        for row in results: 
                            raw_id = row[0]
                            currentInput=cache_path+str(raw_id)+'.scale.png'
                            names.append(currentInput)
                            timestamps.append(row[1])
                            
                            #if the scaled image is not in the cache, download it
                            if not os.path.exists(currentInput):    
                                #print("\t\tSource image does not exist",currentInput)
                                #get the image from the database
                                cursor2 = conn.cursor()
                                depthQuery = 'SELECT D_Scaled FROM Video_Generated WHERE (raw_id=%s)'
                                cursor2.execute(depthQuery, (raw_id))
                                #print("\t\t\tThere are",cursor.rowcount,"images")
                                if cursor.rowcount == 0:
                                    print("x",end="",flush=True)
                                else:
                                    for row2 in cursor2.fetchall():
                                        db_img = row2[0]
                                        if db_img is not None:
                                            img=cv.imdecode(np.asarray(bytearray(db_img),dtype=np.uint8),cv.IMREAD_ANYDEPTH)
                                            #Save the image
                                            cv.imwrite(currentInput,img)
                                            print("⇣",end="",flush=True)
                                            #print("\t\t\t\tJust wrote",currentInput)
                                        else:
                                            print("\t\t\t\tJust couldn't write",currentInput)
                                            #print("x",end="",flush=True)
                            #else:
                                #print("\t\tSource image does exist",currentInput)
                        depth_frame_calculate(names,currentOutput,cache_path)
                        print("Σ",end="",flush=True)
                    else:
                        print("~",end="",flush=True)

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
    print("Storing the depth flow images (# = deleting multiple db entries, o = placeholder created in db, ⇡ = uploaded to db , ~ = db not updated b/c image already present with date)")
    print("")
    
    #Get all the ids that are currently present in the db with no update time
    null_update_ids = []
    check_query = "SELECT raw_id FROM Video_Generated WHERE D_Depth_Flow_Updated IS NULL order by raw_id"
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
    check_query = "SELECT raw_id, D_Depth_Flow_Updated FROM Video_Generated WHERE D_Depth_Flow_Updated IS NOT NULL order by raw_id"
    try:
        cursor.execute(check_query)
        for x in cursor.fetchall(): 
            not_null_update_ids.append(x[0])
    except Exception as e:
        print("Unable to select ids from Video_Generated with not null update time",e)
        conn.rollback()
        raise e

    #Try and insert all the images from the cache that are of the pattern "*dflow.png"
    for f in sorted(os.listdir(cache_path)):
        if "dflow" in f:
            currentID = int(f[:-10])
            file_name = cache_path + f
            #print("Storing:",currentID,"filename:",file_name)

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
                update_query = "UPDATE Video_Generated SET D_Depth_Flow = %s, D_Depth_Flow_Updated = NOW() WHERE (raw_id = %s)"
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
    #open connection to database
    conn = connect()
    try:
        #identify the targets
        recording_ids = range(2,3)

        process_d_frames(conn,recording_ids,"cache/")
        store_processed_images(conn,"cache/")
    finally:
        conn.close()
    print("Don't forget to erase the cache files!");



main()
