from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import json
import glob,os
import pylab
import cv2
#get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output
import skvideo as skv
import math
from IPython.display import HTML
import sort as sort
from scipy.spatial import distance

def nms(det_id,pos, posList,threshold):
        
    for det_id2,pos2 in enumerate(posList):

        if (det_id2 == det_id): 
            continue
        if (det_id2!=det_id):

            dis = distance.euclidean(pos2[0:2], pos[0:2])
            #print(dis)
            if (dis<threshold):

                if (pos2[2]>pos[2]):
                    return False

            #return False
    return True

def convert_detections_json_to_csv(dets_json,w,h,json_dets_file=None,dets_csv = None,use_nms=True):



    if (json_dets_file is None):
        json_dets_file = dets_json

    if (dets_csv is None):
        dets_csv= json_dets_file+'.dets.txt'


    with open(dets_json) as f:
        dets_json = json.load(f)

    keylist = list(dets_json.keys())
    out={}
    
    with open(dets_csv,'w') as out_file:
        for i in range(len(keylist)):
            frame = keylist[i]

            out[frame]=[]

            frameitem=dets_json[frame]
            parts=frameitem['parts']['2'] # 2==thorax

            for det in parts:
                thorax = (det[0],det[1],det[2])
                out[frame].append(thorax)

        for i in range(len(out)):
            frame = keylist[i]
            for det_id,pos in enumerate(out[frame]):
                if (use_nms):
                    if (nms(det_id,pos,out[frame],9.0) is not True):
                        continue

                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(int(frame),pos[0]-w/2,pos[1]-h/2,w,h),file = out_file)




    


    #         dets.append([int(frame),det_id,pos[0]-w/2,pos[1]-h/2,w,h, 1, -1,-1,-1])
    # seq_dets=np.array(dets)
    # return seq_dets



def do_tracking(dets_csv, tracks=None):

    if (dets_csv is None):
        dets_csv = dets_csv
    if (tracks is None):
        tracks = dets_csv+'.tracks.txt'

    seq_dets = np.loadtxt(dets_csv,delimiter=',') #load detections


    total_time = 0.0
    total_frames = 0
    trajectory=[]


    mot_tracker = sort.Sort() #create instance of the SORT tracker
    sort.KalmanBoxTracker.count=0
   

    with open(tracks,'w') as tracks_fid:
        for frame in range(int(seq_dets[:,0].max())+1):
             
            dets = seq_dets[seq_dets[:,0]==frame,2:7]
            dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            total_frames += 1

            start_time = time.time()
            trackers_states = mot_tracker.update(dets)
            cycle_time = time.time() - start_time
            total_time += cycle_time
            
            for d in trackers_states: 
                print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file = tracks_fid)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dets_json', default ='merged_C02_170621100000_fine_new.json', help='input detections file')

    parser.add_argument('--dets_csv', default='merged_C02_170621100000_fine_new.json.dets.txt', help='output name file, by default converted detections file name')
    
    parser.add_argument('--tracks', default='', help='output name file, by default track+detections file name')



    args = parser.parse_args()

    dets_json = args.dets_json
    
    

    path = args.tracks

    dets_csv = args.dets_csv

    output = args.tracks

    convert_detections_json_to_csv(dets_json,400,400,json_dets_file=None,dets_csv = None,use_nms = True)

    do_tracking(dets_csv, tracks= None)



    if output =='':

        output='Track_'+path

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
