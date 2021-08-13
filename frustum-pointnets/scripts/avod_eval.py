#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:44:53 2020

@author: emec
"""
#python avod_eval.py --log_dir /root_avod/data3/outputs/pyramid_cars_with_aug_example/predictions98/ --gt_root_dir /root_avod/Kitti/object/val_labels/

#python avod_eval.py --log_dir /root_avod/data3/outputs/pyramid_people_example/predictions98/ --gt_root_dir /root_avod/Kitti/object/val_labels/

import os
import argparse
from argparse import Namespace
import IPython
import shutil
import numpy as np
import time
import subprocess as sp
from multiprocessing import Process
import IPython

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values



def get_folder_names(path):
    log_directories = [os.path.join(path,p) for p in os.listdir(path)\
                       if os.path.isdir(os.path.join(path,p))]
    return log_directories

def run_eval(log_dir,gt_root_dir):
    ckpts = get_folder_names(log_dir)
    for ckpt_name in ckpts:
        gt_dir = os.path.join(gt_root_dir)
        print(ckpt_name,gt_dir)
        command_eval = '../train/kitti_eval/evaluate_object_3d_offline {} {}'.format(\
                                                                             gt_dir,ckpt_name)
        os.system(command_eval)


def read_summary(sum_dir):
    f = open(sum_dir,"r")
    lines = f.readlines()
    res_dict = {}
    # read the summary file
    for line in lines:
        # Get the result type with the AP metrics
        key_raw,res_raw = line.split(':')
        # Split the raw result type
        key = key_raw.split(' ')[0]
        # Gen a new key wit hthe result type
        res_dict[key] = {}        
        # Read AP values
        _,easy,moderate,hard = res_raw.split(' ')
        # Record AP values acc to the diff level
        res_dict[key]['easy']=float(easy)
        res_dict[key]['moderate']=float(moderate)
        res_dict[key]['hard']=float(hard)
    return res_dict

def get_summary_drive(log_dir,model_name=None):
    '''
    To read evaluation results of AP, which are written in a summary.txt file
    in the respective folder that belongs to the evaluated drive id. 
    '''

    results_dict=dict()
    results_dict['ckpts']=get_folder_names(log_dir)
    
    for ckpt_dir in results_dict['ckpts']:
        sum_path = os.path.join(ckpt_dir,'plot','summary.txt')
        sum_dict=read_summary(sum_path)
        results_dict[os.path.basename(ckpt_dir)]=sum_dict
    
    return results_dict

def write_model_evals(log_dir,model_results):
    f = open(os.path.join(log_dir,'modelEvals.txt'),'w')
    model_names = list(model_results.keys())
    
    model_names_indices = [int(model_name) for model_name in model_names if model_name!='ckpts']
    model_names_indices.sort()
    #IPython.embed()
    cat_bests = dict()
    f.write('{:>15} {:>15} {:>15} {:>15} {:>15}\n'.format('Model Name','3D AP', 'Easy', 'Moderate', 'Hard '))
    for model_id in model_names_indices:
        model_name = '{}'.format(model_id)
        categories = model_results[model_name].keys()
        for cat in categories:
            if cat[-3:] == '_3d':
            
                if cat.split('_')[0] not in cat_bests.keys():
                    cat_bests[cat.split('_')[0]]=[]
                res = model_results[model_name][cat]
                f.write('{:>15} {:>15} {:>15} {:>15} {:>15}\n'.format(model_name,cat.split('_')[0], res['easy'], res['moderate'],res['hard']))
                cat_bests[cat.split('_')[0]].append(res['moderate'])
    
    for cat in cat_bests.keys():
        f.write('Max {}: {}\n'.format(cat,max(cat_bests[cat])))
            
    '''        
    for model_id in model_names_indices:
        model_name = 'model_{}.ckpt'.format(model_id)
        f.write('{}\n'.format(model_name))
        drive_sorted = model_results[model_name]['drives']
        drive_sorted.sort()
        for drive in model_results[model_name]['drives']:
            categories = model_results[model_name][drive].keys()
            f.write('{:>15} {:>15} {:>15} {:>15}\n'.format('3D AP', 'Easy', 'Moderate', 'Hard '))
            for cat in categories:
                if cat[-3:] == '_3d':
                    res = model_results[model_name][drive][cat]
                    f.write('{:>15} {:>15} {:>15} {:>15}\n'.format(cat.split('_')[0]+drive, res['easy'], res['moderate'],res['hard']))
    '''
    f.close()
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',default=None,help='Path to a single log folder that contains log files of a training. This is overwritten by log_dirs if given.')
    parser.add_argument('--log_dirs',default=None,help='Path to log folders. Inside there should be log_* folders that contains logs of a training.')
    parser.add_argument('--gt_root_dir',default=None,help='Path to root dir of ground-truth tracking labels')
    parser.add_argument('--pickle_name',default=None,help='Name of the pickle file to be used in inference. If this is None, the pickle file used for evaluation will be used.')
    parser.add_argument('--multi_model', action='store_true', default=False, help="To evaluate all the models that were saved in one training. ")
    parser.add_argument('--parallel', action='store_true', default=False, help="To evaluate in parallel when multi_model is used. ")
    args = parser.parse_args()
    
    #run_eval(args.log_dir,args.gt_root_dir)
    model_results = get_summary_drive(args.log_dir)
    np.save(os.path.join(args.log_dir,'modelEval.npy'),model_results)
    write_model_evals(args.log_dir,model_results)
    
    
            
