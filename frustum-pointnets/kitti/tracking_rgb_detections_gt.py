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
import argparse

def tracking_rgb_detections_from_gt(tracking_path,rgb_det_path,drives):
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

        ## Get the dataset instance of a specific drive with functions to get objects
        drive_dataset = get_dataset(video_num=drive,main_path=tracking_path)

        # Get image addresses to see how many images to go through
        img_addrs = drive_dataset.image_addresses
        objs_str = ''
        # Go through imgs to read objs of each
        for img_addr in img_addrs:
            # int image index
            img_id = drive_dataset.get_id_from_addr(img_addr)[0]
            # objects as instances of tracking_object class
            objs = drive_dataset.get_objects(ind=img_id,filter_out_classes=[])

            # Go through objects to generate string to be written into the file

            for i_obj,obj in enumerate(objs):
                if obj.class_name == 'Pedestrian':
                    obj_cls_id = 1
                elif obj.class_name == 'Car':
                    obj_cls_id = 2
                elif obj.class_name == 'Cyclist':
                    obj_cls_id = 3
                else:
                    continue

                obj_scr = np.random.uniform(low=0.5)
                x1 = obj.x1 
                x2 = obj.x2
                y1 = obj.y1
                y2 = obj.y2
                objs_str += '{:06d}.png {:04d} {} {} {:.2f} {} {} {} {}\n'.format(img_id, drive,\
                                                                                        obj.track_id, \
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
    
    tracking_rgb_detections_from_gt(args.tracking_path,args.rgb_det_path,args.drive_ids)