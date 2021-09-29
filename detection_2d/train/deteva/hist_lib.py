#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:46:54 2020

@author: emec
"""
import numpy as np
from collections import defaultdict
import glob, os
try:
    from box_util import box3doverlap
    from munkres import Munkres
except:
    from deteva.box_util import box3doverlap
    from deteva.munkres import Munkres

import IPython 

class KittiObject3d(object):
    ''' 3d object label '''
    def __init__(self, file_line):
            
        data = file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.xcent2d = (self.xmax+self.xmin)/2.
        self.ycent2d = (self.ymax+self.ymin)/2.
        self.h2d = self.ymax - self.ymin
        self.w2d = self.xmax - self.xmin
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        self.box2d_xywh = np.array([self.xcent2d,self.ycent2d,self.w2d,self.h2d])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        if len(data) > 15: 
            self.score = float(data[15])
        else:
            self.score = 0.99
        if len(data) > 16: self.id = int(data[16])
        
        self.X = self.t[0] 
        self.Y = self.t[1]   
        self.Z = self.t[2]          
        self.yaw = self.ry
        self.check_val_type()
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % (self.t[0], self.t[1], self.t[2], self.ry))
        print('score: %f' % (self.score))

    def convert_to_label_str(self):
        return '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.truncation, self.occlusion, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
                self.h, self.w, self.l, self.t[0], self.t[1], self.t[2], self.ry)
    
    def convert_to_pred_str(self):
        return '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.truncation, self.occlusion, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
                self.h, self.w, self.l, self.t[0], self.t[1], self.t[2], self.ry, self.score)
            
    def check_val_type(self):
        if np.abs(self.ymin-self.ymax)>=40 and self.truncation<=0.15 and (self.occlusion in [0]):
            self.val_type = 'easy'
            self.val_type_id = 0
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.3 and (self.occlusion in [0,1]):
            self.val_type = 'moderate'
            self.val_type_id = 1
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.5 and (self.occlusion in [0,1,2]):
            self.val_type = 'hard'
            self.val_type_id = 2
        else:
            self.val_type = 'unknown'
            self.val_type_id = 3  
            
            
        
class KittiObject3dLabel(object):
    ''' 3d object label '''
    def __init__(self, file_line):
            
        data = file_line.split(' ')
        
        self.frame_id = int(data[0])
        self.obj_id = int(data[1])
        
        data = data[2:]
        data[1:] = [float(x) for x in data[1:]]
        
        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        
        self.X = self.t[0] 
        self.Y = self.t[1]   
        self.Z = self.t[2]          
        self.yaw = self.ry
        
        self.score = 0.99 #Dummy score
        
        self.check_val_type()
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % (self.t[0], self.t[1], self.t[2], self.ry))

    def convert_to_label_str(self):
        return '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' % \
            (self.type, self.truncation, self.occlusion, self.alpha, self.xmin, self.ymin, self.xmax, self.ymax,
                self.h, self.w, self.l, self.t[0], self.t[1], self.t[2], self.ry)
    def check_val_type(self):
        if np.abs(self.ymin-self.ymax)>=40 and self.truncation<=0.15 and (self.occlusion in [0]):
            self.val_type = 'easy'
            self.val_type_id = 0
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.3 and (self.occlusion in [0,1]):
            self.val_type = 'moderate'
            self.val_type_id = 1
        elif np.abs(self.ymin-self.ymax)>=25 and self.truncation<=0.5 and (self.occlusion in [0,1,2]):
            self.val_type = 'hard'
            self.val_type_id = 2
        else:
            self.val_type = 'unknown'
            self.val_type_id = 3  

def read_gt_objects(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [KittiObject3dLabel(line) for line in lines]
    return objects

def read_predicted_objects(filename):
    lines = [line.rstrip() for line in open(filename)]
    objects = [KittiObject3d(line) for line in lines]
    return objects


def read_ground_truth(sequence_name,class_of_interest=['car', 'pedestrian'],
                      difficulty_include=[0,1,2,3],track_obj_filter=False,
                      track_count=2):
    gt_frames = dict()
    for cl in class_of_interest:
        gt_frames[cl] = defaultdict(list)
        if track_obj_filter:
            gt_frames[cl]['track_ids'] = []
    
    for obj in read_gt_objects(sequence_name):
        if obj.type.lower() in class_of_interest:
            if obj.val_type_id in difficulty_include:
                gt_frames[obj.type.lower()][obj.frame_id].append(obj)
                if track_obj_filter:
                    gt_frames[obj.type.lower()]['track_ids'].append(obj.obj_id)
    if track_obj_filter:
        gt_frames_filtered = dict()
        for cl in class_of_interest:
            gt_frames_filtered[cl] = defaultdict(list)
        
        for obj in read_gt_objects(sequence_name):
            if obj.type.lower() in class_of_interest:
                if obj.val_type_id in difficulty_include:
                    if gt_frames[obj.type.lower()]['track_ids'].count(obj.obj_id)>=track_count:
                        gt_frames_filtered[obj.type.lower()][obj.frame_id].append(obj)
        gt_frames = gt_frames_filtered
    return gt_frames



def read_predictions(root_path,class_of_interest=['car', 'pedestrian']):
    frames = glob.glob(os.path.join(root_path, '*.txt'))
    
    predictions = dict()
    for cl in class_of_interest:
        predictions[cl] = defaultdict(list)
    
    for frame in frames:
        frame_id = frame.split('/')[-1]
        try:
            frame_id = int(frame_id.split('.')[0])
        except:
            continue
        objs = read_predicted_objects(frame)
    
        for obj in objs:
            if obj.type.lower() in class_of_interest:
                predictions[obj.type.lower()][frame_id].append(obj)
    
    return predictions


class BBox:
    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.iou3d = 0
    
    def iou3d_with_gt(self, iou):
        self.iou3d = iou

class DetectorStats:
    def __init__(self, predictions_, obj_types_):
        self.predictions = predictions_
        self.obj_types = obj_types_
        self.tp_in_frames = dict()
        self.fn_in_frames = dict()
        self.iou3d_in_frames = dict()
        self.matching = dict()
        self.best_prediction = dict()
        for typ in self.obj_types:
            self.tp_in_frames[typ] = defaultdict(list)
            self.fn_in_frames[typ] = defaultdict(list)
            self.iou3d_in_frames[typ] = defaultdict(list)
            self.matching[typ] = defaultdict()
            self.best_prediction[typ] = defaultdict(list)
            
def associate_mat(gt_frames, predictions,class_of_interest=['car', 'pedestrian'],min_cost=1.0-0.7,max_cost=100) -> DetectorStats:
    hm = Munkres()
    stats = DetectorStats(predictions, class_of_interest)
    
    for obj_type, frames in gt_frames.items():
        if obj_type in class_of_interest:
            for frame_id, frame in frames.items():
                cost_matrix = list()
                if len(frame)==0:
                    continue
                for g in frame:
                    row = list()
                    row_iou = list()
                    for p in predictions[obj_type][frame_id]:
                        iou3d = box3doverlap(g, p)
                        row_iou.append(iou3d)
                        cost = 1-iou3d
                        if cost < min_cost:
                            row.append(cost)
                        else:
                            row.append(max_cost)
                    
                    #maybe there is no prediction
                    if len(row) < 1:
                        row = [max_cost]*len(frame)
                        row_iou = [0]*len(frame)
                        
                    cost_matrix.append(row)
                    stats.iou3d_in_frames[obj_type][frame_id].append(row_iou)
                    stats.best_prediction[obj_type][frame_id].append(max(row_iou))
                
                association_mat = hm.compute(cost_matrix)
                matching = list()
                
                for gt_id, p_id in association_mat:
                    if cost_matrix[gt_id][p_id] < min_cost:
                        matching.append(p_id)
                    else:
                        matching.append(-1)
                        
                stats.matching[obj_type][frame_id]= matching
    #             print(stats.matching[obj_type][frame_id])
    #             print()
    #             print(association_mat)
    #             print()
    #             pprint(cost_matrix)
    #             break
    #         break
    return stats

def compare_stats(gt_frames, detector1, detector2, obj_type):
    
    if obj_type not in gt_frames.keys():
        raise Exception("This object class is not supported!")
    
    best_predictions1 = list()
    best_predictions2 = list()
    
#     gt_obj_track = defaultdict(list)
    
    for frame_id, frame in gt_frames[obj_type].items():
        for idx, g in enumerate(frame):
            best_predictions1 += detector1.best_prediction[obj_type][frame_id]
            best_predictions2 += detector2.best_prediction[obj_type][frame_id]
            
            
#             gt_obj_track[g.obj_id].append([np.ceil(detector1.matching[obj_type][frame_id][idx]), 
#                                            np.ceil(detector2.matching[obj_type][frame_id][idx])])

    return best_predictions1, best_predictions2
        
