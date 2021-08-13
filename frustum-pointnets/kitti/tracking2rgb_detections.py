#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:08:53 2020

@author: emec
"""
import os
import numpy as np
import sys
sys.path.append("../../faster_rcnn")
from lib.lstm_seq_data import get_dataset
import sys
sys.path.append('/frustum_framework/detection_2d/train/deteva/')
from hist_lib import read_ground_truth
import argparse
import IPython


def tracking2rgb_detections(tracking_path,rgb_det_path,drives):
    '''
    tracking_path: Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders
    rgb_det_path : Path to the rgb_detections folder that contain following: <drive_id>/rgb_detection.txt: 0001/rgb_detection.txt, 0011/rgb_detection.txt
    drive_ids : List of drive ids whose labels will be converted. [0,1,2,3]
    '''
    for drive in drives:
        # Create the drive folder
        drive_path = os.path.join(rgb_det_path,'{:04d}'.format(drive))
        if os.path.isdir(drive_path):
            print("Drive folder exists!")
        else:
            os.makedirs(drive_path)
        
        
        pred_list = read_ground_truth(os.path.join(tracking_path,'{:04d}'.format(drive),'track_labels','{:04d}.txt'.format(drive)),class_of_interest=['car', 'pedestrian','cyclist'])
        #IPython.embed()
        ## Get the dataset instance of a specific drive with functions to get objects
        
        img_addr_list = []
        cls_names = pred_list.keys()
        for cls_name in cls_names:
            img_addr_list += list(pred_list[cls_name].keys())
        
        img_addr_list = list(set(img_addr_list))
        img_addr_list.sort()

        objs_str = ''
        # Go through imgs to read objs of each
        for img_id in img_addr_list:
            for cls_name in cls_names:
                # objects as instances of tracking_object class
                try:
                    objs = pred_list[cls_name][img_id]
                except:
                    print('No object in drive {}, image {}, cls {} '.format(drive,img_id,cls_name))
                
                
                
                # Go through objects to generate string to be written into the file
                for i_obj,obj in enumerate(objs):
                    if cls_name == 'pedestrian':
                        obj_cls_id = 1
                    elif cls_name == 'car':
                        obj_cls_id = 2
                    elif cls_name == 'cyclist':
                        obj_cls_id = 3
                    else:
                        continue
    
                    obj_scr = np.random.uniform(low=0.8)
                    x1,y1,x2,y2 = obj.box2d

                    objs_str += '{:06d}.png {:04d} {} {} {:.2f} {} {} {} {}\n'.format(img_id, drive,\
                                                                                            obj.obj_id, \
                                                                                            obj_cls_id,\
                                                                                            obj_scr,\
                                                                                            x1,y1,x2,y2)
        # Open the file for each image and write labels
        with open(os.path.join(drive_path,'rgb_detection.txt'),'w') as file_name:
            file_name.write(objs_str)
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_path',help='Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders')
    parser.add_argument('--drive_ids', metavar='N', type=int, nargs='+',help='Drive ids whose labels will be converted. [0,1,2,3]')
    parser.add_argument('--rgb_det_path',default=None,help='Path to the rgb_detections folder that contain following: <drive_id>/rgb_detection.txt: 0001/rgb_detection.txt, 0011/rgb_detection.txt')    
    args = parser.parse_args()
    
    tracking2rgb_detections(args.tracking_path,args.rgb_det_path,args.drive_ids)
