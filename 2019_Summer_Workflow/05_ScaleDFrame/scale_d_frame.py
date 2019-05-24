# -*- coding: utf-8 -*-
"""
The goal here is to go through a collection of D_frames find the max and min
values and then rescale the same D_frames by the max and min and store it in the
D_Scaled field.  There are a few things that are happening that reduce repeated
computation.

In main you specify the complete set of recording_ids that you are scaling
over.  The code loops through all of those and finds the max and min values in
the depth images.  It looks in a local cache directory, 'cache' for the images
before asking the database for them.  If it does need an image from the database
it stores it in the cache. Make sure existing files in the cache are good or you
will be processing bad data.

From those images we create a scaled version of each image also storing it in the cached
directory in the format <raw_id>.scale.png.  If the file already exists, then it
is not recalculated or replaced. So make sure those are good.

Then the code goes through the cache directory and uploads any files that have
"scale" in the name.  It does not update any entries in the database unless the
"D_Scaled_Updated" field is NULL.
Clear the D_Scaled_Updated field with a query like:

    UPDATE Video_Raw JOIN Video_Generated ON Video_Raw.id = Video_Generated.raw_id SET Video_Generated.D_Scaled_Updated = NULL WHERE Video_Raw.recording_id >= 2;
@author djp3
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

import sys
import pandas as pd
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


#Find the min and max values in a selection of images indicated by a collection of recording_ids
#Store images in the cache_directory
def find_min_max(conn,recording_ids,cache_path):
    cursor = conn.cursor()
    my_max = -sys.maxsize-1
    my_min = sys.maxsize
    print("Calculating min and max of images (~ = image in local disk cache, ⇣ = image fetched from db, x = not image in db)")
    for r_id in recording_ids:
        #Find all the relevant depth images to analyze
        print("")
        print("Analyzing recording_id:",r_id,": ",end="")
        query = 'SELECT id from Video_Raw where (recording_id=%s) order by id'
        try:
            cursor.execute(query,(r_id))
            print(cursor.rowcount,"entries");
            for row in cursor.fetchall():
                img = None
                current = cache_path+str(row[0])+'.png'
                #check if it's already downloaded
                if os.path.exists(current):
                    print("~",end="",flush=True)
                    img = cv.imread(current,cv.IMREAD_ANYDEPTH)
                else:
                    #get the image from the database
                    cursor2 = conn.cursor()
                    query = 'SELECT D_frame from Video_Raw where (id=%s)'
                    cursor2.execute(query,(row[0]))
                    #Should only be 1 result
                    for row2 in cursor2.fetchall():
                        db_img = row2[0]
                        if db_img is not None:
                            img = cv.imdecode(np.asarray(bytearray(db_img),dtype=np.uint8),cv.IMREAD_ANYDEPTH)
                            #Save the image
                            cv.imwrite(current,img)
                            print("⇣",end="",flush=True)
                        else:
                            print("x",end="",flush=True)
                if img is not None:
                    new_max = np.amax(img)
                    if my_max < new_max :
                        my_max = new_max
                    new_min = np.amin(img)
                    if my_min > new_min :
                        my_min = new_min
        except Exception as e:
            print("Something failed",e)
            conn.rollback()
            raise e
    return (my_min,my_max)

#Scale the images in the database according to min/max and store them in the
#cache directory
def rescale_images(conn,min,max,recording_ids,cache_path):
    cursor = conn.cursor()
    #Keep track of post-scaling information
    my_max = -sys.maxsize-1
    my_min = sys.maxsize
    print("Rescaling the white/black levels of images (~ = solution already cached, Σ = solution calculated, ⇣ = original fetched from db, x = not image in db)")
    for r_id in recording_ids:
        print("")
        print("Rescaling recording_id:",r_id,": ",end="")
        query = 'SELECT id from Video_Raw where (recording_id=%s) order by id'
        try:
            cursor.execute(query,(r_id))
            print(cursor.rowcount,"entries");
            for row in cursor.fetchall():
                img = None
                current_ID = row[0]
                current_input = cache_path+str(current_ID)+'.png'
                current_output = cache_path+str(current_ID)+'.scale.png'
                #check if the output is already calculated
                if os.path.exists(current_output):
                    print("~",end="",flush=True)
                else:
                    #check if the input is already downloaded
                    if os.path.exists(current_input):
                        img = cv.imread(current_input,cv.IMREAD_ANYDEPTH)
                    else:
                        #get the image from the database
                        cursor2 = conn.cursor()
                        query = 'SELECT D_frame from Video_Raw where (id=%s)'
                        cursor2.execute(check_query,(current_ID))
                        #Should only be 1 result
                        for row2 in results.fetchall():
                            db_img = row2[0]
                            if db_img is not None:
                                img = cv.imdecode(np.asarray(bytearray(db_img),dtype=np.uint8),cv.IMREAD_ANYDEPTH)
                                #Save the cache image
                                cv.imwrite(current_input,img)
                                print("⇣",end="",flush=True)
                            else:
                                print("x",end="",flush=True)
                    if img is not None:
                        img -= [min]
                        scale = np.floor(65536./(max-min)).astype('uint16')
                        img *= [scale]
                        #Save the scaled image
                        cv.imwrite(current_output,img)
                        print("Σ",end="",flush=True)
                        #Keep track of post-scaling information
                        new_max = np.amax(img)
                        if my_max < new_max :
                            my_max = new_max
                        new_min = np.amin(img)
                        if my_min > new_min :
                            my_min = new_min
        except Exception as e:
            print("Something failed",e)
            conn.rollback()
            raise e
    print("")
    print("The post-scaled min is",my_min," and new max is",my_max)

#Upload any scaled images stored in the cache directory to the database
def store_scaled_images(conn,cache_path):
    cursor = conn.cursor()
    print("Storing the w/b corrected (# = deleting multiple db entries, o = placeholder created in db, ⇡ = uploaded to db , ~ = db not updated b/c image already present with date)")
    print("")

    #Get all the ids that are currently present in the db with no update time
    null_update_ids = []
    check_query = "SELECT raw_id, D_Scaled_Updated FROM Video_Generated WHERE D_Scaled_Updated IS NULL order by raw_id"
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
    check_query = "SELECT raw_id, D_Scaled_Updated FROM Video_Generated WHERE D_Scaled_Updated IS NOT NULL order by raw_id"
    try:
        cursor.execute(check_query)
        for x in cursor.fetchall(): 
            not_null_update_ids.append(x[0])
    except Exception as e:
        print("Unable to select ids from Video_Generated with not null update time",e)
        conn.rollback()
        raise e

    #Try and insert all the images from the cache that are of the pattern "*scale.png"
    for f in sorted(os.listdir(cache_path)):
        if "scale" in f:
            currentID = int(f[:-10])
            file_name = cache_path + f
            #print("Storing:",currentID,"filename:",file_name)

            null_present = null_update_ids.count(currentID)
            not_null_present = not_null_update_ids.count(currentID)
            present = null_present + not_null_present

            #print("\t",file_name,"is present with null",null_present,"is present with not null",not_null_present,"total is",present)

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


                update_query = "UPDATE Video_Generated SET D_Scaled = %s, D_Scaled_Updated = NOW() WHERE (raw_id = %s)"
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

            
def main():
    #open connection to database
    conn = connect()
    try:
        #identify the targets
        recording_ids = range(2,14)

        (min, max) = find_min_max(conn,recording_ids,"cache/")

        print("")
        print("The pre-scaled min is",min," and max is",max)

        rescale_images(conn,min,max,recording_ids,"cache/")

        store_scaled_images(conn,"cache/")
    finally:
        conn.close()
    print("Don't forget to erase the cache files!");



main()
                    
