#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:03:43 2020

@author: emec
"""

from kitti_object import kitti_tracking_object
import argparse
import IPython
import os
from shutil import copyfile

def reverse_drive(root_path,drive_ids,new_drive_ids):
    tracking_dataset = kitti_tracking_object(root_path,'training',drive_list=drive_ids)
    #IPython.embed()
    for d_id,n_id in zip(drive_ids,new_drive_ids):
        print('Drive ID {}-{}'.format(d_id,n_id))
        l_data = tracking_dataset.drive_length[d_id]
        
        drive_im_dir = os.path.join(tracking_dataset.image_dir,'{:04d}'.format(d_id))
        drive_lid_dir = os.path.join(tracking_dataset.lidar_dir,'{:04d}'.format(d_id))
        drive_calib_path = os.path.join(tracking_dataset.calib_dir,'{:04d}.txt'.format(d_id))
        drive_label_path = os.path.join(tracking_dataset.label_dir,'{:04d}.txt'.format(d_id))
        
        n_drive_im_dir = os.path.join(tracking_dataset.image_dir,'{:04d}'.format(n_id))
        n_drive_lid_dir = os.path.join(tracking_dataset.lidar_dir,'{:04d}'.format(n_id))
        n_drive_calib_path = os.path.join(tracking_dataset.calib_dir,'{:04d}.txt'.format(n_id))
        n_drive_label_path = os.path.join(tracking_dataset.label_dir,'{:04d}.txt'.format(n_id))
        
        try:
            os.makedirs(n_drive_im_dir)
            os.makedirs(n_drive_lid_dir)
        except:
            pass
        
        print('New directories: ')
        print('    Image: {}'.format(n_drive_im_dir))
        print('    LiDAR: {}'.format(n_drive_lid_dir))
        #os.makedirs(n_drive_calib_path)
        #os.makedirs(n_drive_label_path)
        
        copyfile(drive_calib_path,n_drive_calib_path)
        print('Calib file copied!')
        objects = {}
        for i_fr in range(l_data):
            n_i_fr = l_data-1-i_fr
            im = os.path.join(drive_im_dir,'{:06d}.png'.format(i_fr))
            lid = os.path.join(drive_lid_dir,'{:06d}.bin'.format(i_fr)) 
            objs =  tracking_dataset.get_label_objects(i_fr,d_id)
            for obj in objs:
                obj.frame = n_i_fr
            objects[n_i_fr] = objs
            
            n_im = os.path.join(n_drive_im_dir,'{:06d}.png'.format(n_i_fr))
            n_lid = os.path.join(n_drive_lid_dir,'{:06d}.bin'.format(n_i_fr))
            
            copyfile(im, n_im)
            print('        Drive ID {}-{}, Img {}-{} is copied!'.format(d_id,n_id,i_fr,n_i_fr))
            copyfile(lid, n_lid)
            print('        Drive ID {}-{}, LiDAR {}-{} is copied!'.format(d_id,n_id,i_fr,n_i_fr))
            
            
        with open(n_drive_label_path, 'w') as f:
            for n_i_fr in range(l_data):
                objs = objects[n_i_fr]
                for obj in objs:
                    str_wr = obj.gen_label_line()
                    f.write(str_wr)
        print('        Labels are written!')
        
            
        
        
        
        

if __name__=='__main__':        
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_path',help='Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders')
    parser.add_argument('--drive_ids', metavar='N', type=int, nargs='+',help='Drive ids whose labels will be converted. [0,1,2,3]')
    parser.add_argument('--new_drive_ids', metavar='N', type=int, nargs='+',help='The drives will be saved with these drive ids in the same folder after reversing the frames.[0,1,2,3]')
    args = parser.parse_args()
    assert(len(args.drive_ids) == len(args.new_drive_ids))
    reverse_drive(args.tracking_path,args.drive_ids,args.new_drive_ids)
    