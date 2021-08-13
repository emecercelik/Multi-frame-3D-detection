#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To combine labels of several drives in one drive.
Input drive labels should be in KITTI object detection dateset format, or should
have converted to that format beforehand.
Created on Fri Nov 20 11:46:45 2020

@author: emec
"""

import os
import argparse
from shutil import copyfile
import glob

def copy_labels(args):
    print('Started!')
    new_drive_path = os.path.join(args.root_dir,'{:04d}'.format(args.output_drive))
    
    try:
        os.makedirs(new_drive_path)
        print('New drive folder generated!')
    except:
        print('The output drive folder exists')
        return -1
    
    drives = args.drives
    drives.sort()
    global_label_counter = 0
    for drive in drives:
        print('Drive {} is being copied!'.format(drive))
        drive_path = os.path.join(args.root_dir,'{:04d}'.format(drive))
        path_files = glob.glob(os.path.join(drive_path,'*.txt'))
        l_labels = len(path_files)
        for i in range(l_labels):
            org_label_path = os.path.join(drive_path,'{:06d}.txt'.format(i))
            dest_label_path = os.path.join(new_drive_path,'{:06d}.txt'.format(global_label_counter))
            copyfile(org_label_path,dest_label_path)
            global_label_counter+=1
    print('Done!')
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drives', metavar='N', type=int, nargs='+',help='Drive IDs to be combined (Ex: 11 15 16 18)')
    parser.add_argument('--output_drive', type=int,help='Output drive ID after combining.')
    parser.add_argument('--root_dir',default=None,help='Path where the drive label folders are kept. Should contain folders 0011, 0015, 0016, 0018 etc. as drive IDs given in the example.')
    args = parser.parse_args()
    
    copy_labels(args)