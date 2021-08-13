#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:06:28 2020

@author: emec
"""

import os
import numpy as np
import argparse
import IPython
def read_one_log(root_path):
    path = os.path.join(root_path,'log_train.txt') 
    f = open(path,"r")
    lines = f.readlines() 
    ## Find folder name
    params = lines[0]
    sp_params = params.split(',')
    for pr in sp_params:
        if pr[1:8]=='log_dir':
            folder_name = os.path.basename(pr[8:])[:-1]
    
    res_5 = [] # acc with iou 0.5
    res_7 = [] # acc with iou 0.7
    res = dict()        
    for i_line, line in enumerate(lines):
        if "EVALUATION" in line:
            res_7.append(float(lines[i_line+5].split(':')[-1]))
            try:
                res_5.append(float(lines[i_line+6].split(':')[-1]))
            except:
                res_5.append(-99.)
            eval_id = int(line.split(' ')[2])
            res[eval_id] = {'0.5':res_5[-1],'0.7':res_7[-1]}
    i_max7 = np.argmax(res_7) ## ind of max value in iou 0.7
    max7 = np.max(res_7) # max accuracy in iou 0.7
    
    i_max5 = np.argmax(res_5) ## ind of max value in iou 0.5
    max5 = np.max(res_5) # max accuracy in iou 0.5
    f.close()
    return [folder_name,i_max7,max7,i_max5,max5]

def get_folder_names(path):
    log_directories = [os.path.join(path,p) for p in os.listdir(path)\
                       if os.path.isdir(os.path.join(path,p))]
    return log_directories

def write_logs_to_file(file_name,res_logs):
    f = open(file_name,'w')
    for log in res_logs:
        str_write = '{:17s} Accuracy IoU=0.5: {:.3f} Epoch:{:03d}, IoU=0.7: {:.3f} Epoch:{:03d}\n'\
            .format(log[0],log[4],log[3],log[2],log[1])
        f.write(str_write)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',default=None,help='Path to a single log folder that contains log files of a training. This is overwritten by log_dirs if given.')
    parser.add_argument('--log_dirs',default=None,help='Path to log folders. Inside there should be log_* folders that contains logs of a training.')
    args = parser.parse_args()
    res_logs = []
    if args.log_dirs is not None:
        log_directories = get_folder_names(args.log_dirs)
        log_directories.sort()
        for log_dir in log_directories:
            res_logs.append(read_one_log(log_dir))
            
        write_logs_to_file(os.path.join(args.log_dirs,'results_summary.txt'),res_logs)
        
    else:
        res_logs = [read_one_log(args.log_dir)]
        write_logs_to_file(os.path.join(args.log_dir,'results_summary.txt'),res_logs)
    
    