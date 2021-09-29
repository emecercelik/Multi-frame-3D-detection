#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 18:08:27 2020

@author: emec
"""
import argparse
from misc import calculate_recall
import IPython

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_predictions',help='Path to the text files that the predictions written in. The names of txt files determines the order. The names are assumed to be in KITTI format.')
    parser.add_argument('--path_to_labels',help='Path to the label map of the trained network.')
    parser.add_argument('--iou_thr', metavar='N', type=float, nargs='+', help='A list of IoU thresholds. The IoU values above the thresholds will be counted as correct detections. The thresolds are given for every class and the indices of classes defined in the label map will be used while calling the thresholds. ')
    parser.add_argument('--scr_thr', metavar='N', type=float, nargs='+',help='A list of score thresholds. Only the predictions that have a greater prediction score will be used while calculating recall. The indices of classes, given in the label map whose path defined with path_to_labels, will be used while taking this scores. ')
    parser.add_argument('--path_to_gts',help='Path to the ground-truth text files of frames in KITTI format')
    
    args = parser.parse_args()
    
    recalls = calculate_recall(args)
