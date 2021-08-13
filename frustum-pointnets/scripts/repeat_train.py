#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 13:12:50 2020

@author: emec
"""

import os
import argparse
from argparse import Namespace

def train_one_dir(log_dir_old,log_dir_new):
    print('!!!!! Started repeat in {} from {}'.format(log_dir_new,log_dir_old))
    # Get log file path
    path1 = os.path.join(log_dir_old,'log_train.txt')
    # Read the network parameters from log file
    f = open(path1,"r")
    lines = f.readlines() 
    params = lines[0]
    # Arguments that the network trained with
    log_args = eval(params)

    # Create the test command
    command = 'python ../train/train.py --gpu {} --num_point {} --model {} \
                --tau {} --track_net {} --track_features {} \
                --log_dir {} --max_epoch {} --batch_size {} \
                --decay_step {} --decay_rate {} --learning_rate {}'.format(\
                log_args.gpu, log_args.num_point, log_args.model, \
                log_args.tau,log_args.track_net, log_args.track_features,\
                log_dir_new, log_args.max_epoch, log_args.batch_size,\
                log_args.decay_step, log_args.decay_rate, log_args.learning_rate)
    
    try:
        layer_sizes = log_args.layer_sizes
        if layer_sizes is not None:
            cmd = ' --layer_sizes '
            for size in layer_sizes:
                cmd+='{} '.format(size)
            command+=cmd
    except:
        pass
    
    try:
        cos_loss = log_args.cos_loss
        if cos_loss:
            cmd = ' --cos_loss --cos_loss_batch_thr {} --cos_loss_prop {} --cos_loss_weight {}'.format(log_args.cos_loss_batch_thr, log_args.cos_loss_prop,log_args.cos_loss_weight )
            command+=cmd
    except:
        pass
    try:
        cell_type = log_args.rnn_cell_type
        cmd = ' --rnn_cell_type {}'.format(cell_type)
        command+=cmd
    except:
        pass
    
    try:
        if log_args.attention:
            command+=' --attention '
    except:
        pass
            
    
    try: 
        multi_layer_flag = log_args.multi_layer
        if multi_layer_flag:
            cmd = ' --multi_layer'
        command+= cmd
    except:
        pass
            
    if log_args.no_intensity:
        command += ' --no_intensity'
    if log_args.tracking:
        command += ' --tracking'
        data_path = 'frustum_carpedcyc_tracking'
        try:
            if log_args.pickle_name is not None:
                data_path = log_args.pickle_name
        except:
            pass
    else:
        data_path = 'frustum_carpedcyc'
        try:
            if log_args.pickle_name is not None:
                data_path = log_args.pickle_name
        except:
            pass
    if log_args.tracks:
        command += ' --tracks'
    
    command += ' --pickle_name {}'.format(data_path)

    # Run the command
    os.system(command)    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir_old',default=None,help='Path to a single log folder that contains log files of a training. This is overwritten by log_dirs if given.')
    parser.add_argument('--log_dir_new',default=None,help='Path to a single log folder that contains log files of a training. This is overwritten by log_dirs if given.')
    args = parser.parse_args()
    
    train_one_dir(args.log_dir_old, args.log_dir_new)