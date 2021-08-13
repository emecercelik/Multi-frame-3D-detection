#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:25:14 2020

@author: emec
"""
import argparse
import numpy as np
import os
import IPython
from argparse import Namespace
import json

def get_namespace(log_dir,parameters=['tau', 'track_net', 'track_features', 'learning_rate']):
    path1 = os.path.join(log_dir,'log_train.txt')
    # Read the network parameters from log file
    f = open(path1,"r")
    lines = f.readlines() 
    params = lines[0]
    # Arguments that the network trained with
    log_args = eval(params)
    
    param_dict = dict()
    str_param = ''
    for param in parameters:
        param_dict[param] = eval('log_args.{}'.format(param))
        str_param+='{}:{}   '.format(param,param_dict[param])
    return param_dict, str_param
    

def get_max_in_one_log(results_dict,detection_type='car_detection_3d',eval_drives=None):
    '''
    results_dict : A dictionary that contains the AP evaluation results in a hierarchy. The first level keys are model names. Each model name contains names of the drives in 'drives' key as well as the keys below.
        drive_name -> detection_type-> difficulty
        '0011' -> 'car_detection_3d' -> 'easy'
    
    detection_type : Type of evaluation. Might be 'car_detection', 'car_detection_ground', 'car_detection_3d', 'pedestrian_detection_3d', ...
    
    '''
    model_names = list(results_dict.keys())
    drive_names = list(results_dict[model_names[0]]['drives'])
    drive_names.sort()
    
    if eval_drives is not None:
        drive_names_2 = []
        for d in drive_names:
            if int(d) in eval_drives:
                drive_names_2.append(d)
        drive_names = drive_names_2
    max_results=dict()
    for dr_name in drive_names:
        max_results[dr_name]=dict()
        max_results[dr_name]['max_easy'] = 0
        max_results[dr_name]['max_moderate'] = 0
        max_results[dr_name]['max_hard'] = 0
        max_results[dr_name]['argmax_easy'] = 0
        max_results[dr_name]['argmax_moderate'] = 0
        max_results[dr_name]['argmax_hard'] = 0
        
    for model in model_names:
        for dr_name in drive_names:
            if detection_type in results_dict[model][dr_name].keys():
                result_per_drive = results_dict[model][dr_name][detection_type]
            else:
                result_per_drive = {'easy':0.0,'moderate':0.0,'hard':0.0}
            
            ## Get the max of easy evaluation among all model results
            if max_results[dr_name]['max_easy']<result_per_drive['easy']:
                max_results[dr_name]['max_easy'] = result_per_drive['easy']
                max_results[dr_name]['argmax_easy'] = int(model.split('_')[1].split('.')[0])
            
            ## Get the max of moderate evaluation among all model results
            if max_results[dr_name]['max_moderate']<result_per_drive['moderate']:
                max_results[dr_name]['max_moderate'] = result_per_drive['moderate']
                max_results[dr_name]['argmax_moderate'] = int(model.split('_')[1].split('.')[0])
            
            ## Get the max of hard evaluation among all model results
            if max_results[dr_name]['max_hard']<result_per_drive['hard']:
                max_results[dr_name]['max_hard'] = result_per_drive['hard']
                max_results[dr_name]['argmax_hard'] = int(model.split('_')[1].split('.')[0])
    
    return max_results
  
def sta_from_log_names(results_dict,log_names,detection_type):
    drive_names = list(results_dict[log_names[0]].keys())
    group_sta = dict()
    group_sta2 = dict()
    for dr_name in drive_names:
        group_sta[dr_name]=dict()
        group_sta2[dr_name]=dict()
        group_sta[dr_name]['easy']=[]
        group_sta[dr_name]['moderate']=[]
        group_sta[dr_name]['hard']=[]
        group_sta2[dr_name]['easy_sta'] = []
        group_sta2[dr_name]['moderate_sta'] = []
        group_sta2[dr_name]['hard_sta'] = []
        group_sta2[dr_name]['easy_max'] = 0
        group_sta2[dr_name]['moderate_max'] = 0
        group_sta2[dr_name]['hard_max'] = 0
    
    for log_name in log_names:
        for dr_name in drive_names:
            result_per_drive = results_dict[log_name][dr_name]
            group_sta[dr_name]['easy'].append(result_per_drive['max_easy'])
            group_sta[dr_name]['moderate'].append(result_per_drive['max_moderate'])
            group_sta[dr_name]['hard'].append(result_per_drive['max_hard'])
    
    for dr_name in drive_names:
        group_sta2[dr_name]['easy_sta'].append(np.mean(group_sta[dr_name]['easy']))
        group_sta2[dr_name]['easy_sta'].append(np.std(group_sta[dr_name]['easy']))
        
        group_sta2[dr_name]['moderate_sta'].append(np.mean(group_sta[dr_name]['moderate']))
        group_sta2[dr_name]['moderate_sta'].append(np.std(group_sta[dr_name]['moderate']))
        
        group_sta2[dr_name]['hard_sta'].append(np.mean(group_sta[dr_name]['hard']))
        group_sta2[dr_name]['hard_sta'].append(np.std(group_sta[dr_name]['hard']))
        

        group_sta2[dr_name]['easy_max']= max(group_sta[dr_name]['easy'])
        
        group_sta2[dr_name]['moderate_max'] = max(group_sta[dr_name]['moderate'])
        
        group_sta2[dr_name]['hard_max'] = max(group_sta[dr_name]['hard'])
        
        
    
    return group_sta2

              
def get_max_among_logs(results_dict,path_to_log,netw_params=['tau', 'track_net', 'track_features', 'learning_rate']):
    log_names = list(results_dict.keys())
    drive_names = list(results_dict[log_names[0]].keys())
    
    max_results=dict()
    for dr_name in drive_names:
        max_results[dr_name]=dict()
        max_results[dr_name]['easy'] = dict()
        max_results[dr_name]['easy']['max'] = 0.6
        max_results[dr_name]['easy']['argmax'] = 0
        max_results[dr_name]['easy']['log_name'] = None
        max_results[dr_name]['easy']['vicinity'] = []
        max_results[dr_name]['easy']['params'] = []
        
        max_results[dr_name]['moderate'] = dict()
        max_results[dr_name]['moderate']['max'] = 0.6
        max_results[dr_name]['moderate']['argmax'] = 0
        max_results[dr_name]['moderate']['log_name'] = None
        max_results[dr_name]['moderate']['vicinity'] = []
        max_results[dr_name]['moderate']['params'] = []
        
        max_results[dr_name]['hard'] = dict()
        max_results[dr_name]['hard']['max'] = 0.6
        max_results[dr_name]['hard']['argmax'] = 0
        max_results[dr_name]['hard']['log_name'] = None
        max_results[dr_name]['hard']['vicinity'] = []
        max_results[dr_name]['hard']['params'] = []
    
    param_sta = dict()
    for log_name in log_names:
        param_dict,str_param = get_namespace(os.path.join(path_to_log,log_name),netw_params)
        try:
            param_sta[str_param].append(log_name)
        except:
            param_sta[str_param]=[]
            param_sta[str_param].append(log_name)
            
        for dr_name in drive_names:
            if max_results[dr_name]['easy']['max'] < results_dict[log_name][dr_name]['max_easy']:
                max_results[dr_name]['easy']['max'] = results_dict[log_name][dr_name]['max_easy']
                max_results[dr_name]['easy']['argmax'] = results_dict[log_name][dr_name]['argmax_easy']
                max_results[dr_name]['easy']['log_name'] = log_name
                max_results[dr_name]['easy']['params'] = str_param
            
            if max_results[dr_name]['moderate']['max'] < results_dict[log_name][dr_name]['max_moderate']:
                max_results[dr_name]['moderate']['max'] = results_dict[log_name][dr_name]['max_moderate']
                max_results[dr_name]['moderate']['argmax'] = results_dict[log_name][dr_name]['argmax_moderate']
                max_results[dr_name]['moderate']['log_name'] = log_name
                max_results[dr_name]['moderate']['params'] = str_param
            
            if max_results[dr_name]['hard']['max'] < results_dict[log_name][dr_name]['max_hard']:
                max_results[dr_name]['hard']['max'] = results_dict[log_name][dr_name]['max_hard']
                max_results[dr_name]['hard']['argmax'] = results_dict[log_name][dr_name]['argmax_hard']
                max_results[dr_name]['hard']['log_name'] = log_name
                max_results[dr_name]['hard']['params'] = str_param
    
    for log_name in log_names:
        param_dict,str_param = get_namespace(os.path.join(path_to_log,log_name),netw_params)
        for dr_name in drive_names:
            if results_dict[log_name][dr_name]['max_easy']+0.5 >= max_results[dr_name]['easy']['max']:
                if log_name != max_results[dr_name]['easy']['log_name']:
                    max_results[dr_name]['easy']['vicinity'].append('{}: {}  '.format(log_name,param_dict))
            
            if results_dict[log_name][dr_name]['max_moderate']+0.5 >= max_results[dr_name]['moderate']['max']:
                if log_name != max_results[dr_name]['moderate']['log_name']:
                    max_results[dr_name]['moderate']['vicinity'].append('{}: {}  '.format(log_name,param_dict))
                
            if results_dict[log_name][dr_name]['max_hard']+0.5 >= max_results[dr_name]['hard']['max']:
                if log_name != max_results[dr_name]['hard']['log_name']:
                    max_results[dr_name]['hard']['vicinity'].append('{}: {}  '.format(log_name,param_dict))
    
    max_results['param_group'] = param_sta
    return max_results
            


def read_all_eval_dicts(main_path,prefix,log_indices,args):
    file_paths = []
    dicts = dict()
    max_logs = dict()
    for log_index in log_indices:
        log_name = '{}{}'.format(prefix,log_index)
        file_paths.append(os.path.join(main_path,log_name,'modelEval.npy'))
        dicts[log_name] = np.load(file_paths[-1],allow_pickle=True).item()
        max_logs[log_name] = get_max_in_one_log(dicts[log_name],detection_type=args.detection_type,eval_drives=args.eval_drives)
    
    max_all = get_max_among_logs(max_logs,main_path,netw_params=args.params_to_group)
    
    param_group_sta = dict()
    for param_group_name in max_all['param_group'].keys():
        log_names = max_all['param_group'][param_group_name]
        param_group_sta[param_group_name] = sta_from_log_names(max_logs,log_names,detection_type=args.detection_type)
    
    with open(os.path.join(main_path,args.output_name+'_sta.json'), 'w', encoding='utf-8') as f:
        
        json.dump(param_group_sta, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(main_path,args.output_name+'_max.json'), 'w', encoding='utf-8') as f:
        json.dump(max_all, f, ensure_ascii=False, indent=4)


def read_npy_files(file_name):
    np.load(file_name,allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path',default=None,help='')
    parser.add_argument('--output_name',default=None,help='')
    parser.add_argument('--prefix',default=None,help='')
    parser.add_argument('--log_indices', metavar='N', type=int, nargs='+',help='')
    parser.add_argument('--params_to_group', metavar='N', type=str, nargs='+',help='')
    parser.add_argument('--detection_type',default='car_detection_3d',help='detection type to be evaluated: car_detection_3d, pedestrian_detection_3d')
    parser.add_argument('--eval_drives', metavar='N', type=int, nargs='+', default=None,help='To evaluate only the given drives')
    
    args = parser.parse_args()
    print("*** Started {} summary!".format(args.output_name))
    read_all_eval_dicts(args.main_path,args.prefix,args.log_indices,args)
    
    
