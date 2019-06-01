# -*- coding: utf-8 -*-
"""
@author: djp3
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

#import cv2 as cv
#import numpy as np
#import os
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

def main():
    #open connection to database
    conn = connect()
    cursor = conn.cursor()    
    test_number = 0
    try:
        print("\tChecking for raw video images that don't have generated images and aren't labelled as test recordings: RGB or D (LIMIT 10)")
        test_number = test_number + 1
        check_query = 'SELECT p.* FROM Participant AS p JOIN Recording AS r ON p.id = r.participant_id JOIN Video_Raw AS v on r.id = v.recording_id LEFT JOIN Video_Generated AS g ON v.id = g.raw_id WHERE p.isTest = 0 AND g.id IS NULL LIMIT 10'
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed, there were",cursor.rowcount,"instances of ummatched image")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking for generated video images that don't have raw images as a source: RGB or D (LIMIT 10)")
        test_number = test_number + 1
        check_query = 'SELECT v.id FROM Video_Raw AS v RIGHT JOIN Video_Generated AS g ON v.id = g.raw_id WHERE v.id IS NULL LIMIT 10'
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed, there were",cursor.rowcount,"instances of ummatched image")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking if there are any downstream images without upstream images: RGB (LIMIT 10)")
        test_number = test_number + 1
        check_query = "SELECT v.id FROM Video_Raw AS v JOIN Video_Generated AS g ON v.id = g.raw_id WHERE v.RGB_frame is NULL AND g.RGB_Optical_Flow IS NOT NULL LIMIT 10"
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking if there are any downstream images without upstream images: D_Scaled (LIMIT 10)")
        test_number = test_number + 1
        check_query = "SELECT v.id FROM Video_Raw AS v JOIN Video_Generated AS g ON v.id = g.raw_id WHERE v.D_frame is NULL AND g.D_Scaled is NOT NULL LIMIT 10"
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking if there are any downstream images without upstream images: D_Depth_Flow (LIMIT 10)")
        test_number = test_number + 1
        check_query = "SELECT v.id FROM Video_Raw AS v JOIN Video_Generated AS g ON v.id = g.raw_id WHERE v.D_frame is NULL AND g.D_Depth_Flow is NOT NULL LIMIT 10"
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed")
            for row in cursor.fetchall():
                print(row)
        print()


        print("\tChecking if there are any downstream images without upstream images part: D_Scaled -> D_Depth_Flow (LIMIT 10)")
        test_number = test_number + 1
        check_query = "SELECT g.id FROM Video_Generated AS g WHERE g.D_Scaled IS NULL AND g.D_Depth_Flow IS NOT NULL LIMIT 10"
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed")
            for row in cursor.fetchall():
                print(row)
                
        print("\tChecking timing problems: Null timestamps(LIMIT 10)")
        test_number = test_number + 1
        check_query = 'SELECT r.*,v.timestamp,g.RGB_Optical_Flow_Updated,g.D_Scaled_Updated,g.D_Depth_Flow_Updated FROM Recording AS r JOIN Video_Raw AS v on r.id = v.recording_id JOIN Video_Generated AS g ON v.id = g.raw_id WHERE r.recording_start IS NULL OR v.timestamp IS NULL OR g.RGB_Optical_Flow_Updated IS NULL OR g.D_Scaled_Updated IS NULL OR g.D_Depth_Flow_Updated IS NULL LIMIT 10'
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed, there were",cursor.rowcount,"timing problems")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking timing problems: Timestamp order problems (LIMIT 10)")
        test_number = test_number + 1
        check_query = 'SELECT r.*,v.timestamp,g.RGB_Optical_Flow_Updated,g.D_Scaled_Updated,g.D_Depth_Flow_Updated FROM Recording AS r JOIN Video_Raw AS v on r.id = v.recording_id JOIN Video_Generated AS g ON v.id = g.raw_id WHERE r.recording_start > v.timestamp OR v.timestamp > g.RGB_Optical_Flow_Updated OR v.timestamp > g.D_Scaled_Updated OR v.timestamp > g.D_Depth_Flow_Updated OR g.D_Scaled_Updated > g.D_Depth_Flow_Updated LIMIT 10'
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed, there were",cursor.rowcount,"timing problems")
            for row in cursor.fetchall():
                print(row)
        print()

        print("\tChecking duplicates (LIMIT 10)")
        test_number = test_number + 1
        check_query = 'SELECT g.id, g.raw_id, COUNT(*) FROM Video_Generated AS g GROUP BY g.raw_id HAVING COUNT(*) > 1 LIMIT 10'
        cursor.execute(check_query)
        if cursor.rowcount == 0:
            print("\tTest",test_number,": Passed")
        else:
            print("\tTest",test_number,": Failed, there were",cursor.rowcount,"duplicate problems")
            #for row in cursor.fetchall():
                #print(row)
        print()


                
    finally:
        conn.close()



main()
