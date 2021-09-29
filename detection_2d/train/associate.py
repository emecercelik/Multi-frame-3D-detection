#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 12:10:04 2020

@author: emec
"""
import argparse
from deteva import hist_lib
import misc
from object_detection.utils import label_map_util
import numpy as np
import os
from sort import Sort
import IPython
import cv2

import sys
sys.path.append('../')

# pip install torch==1.4.0
# pip install torchvision==0.5.0

from deep_sort.detector import build_detector
from deep_sort.deep_sort import build_tracker
from deep_sort.utils.draw import draw_boxes
from deep_sort.utils.parser import get_config
from deep_sort.utils.log import get_logger
from deep_sort.utils.io import write_results

def xywh2xyxy(boxes):
    if len(boxes)!=0:
        x1 = np.reshape(boxes[:,0]-boxes[:,2]/2.,(-1,1))
        x2 = np.reshape(boxes[:,0]+boxes[:,2]/2.,(-1,1))
        y1 = np.reshape(boxes[:,1]-boxes[:,3]/2.,(-1,1))
        y2 = np.reshape(boxes[:,1]+boxes[:,3]/2.,(-1,1))
        return np.hstack((x1,y1,x2,y2))
    else:
        return np.array([])
    
def xyxy2xywh(boxes):
    if len(boxes)!=0:
        x = np.reshape((boxes[:,0]+boxes[:,2])/2.,(-1,1))
        y = np.reshape((boxes[:,1]+boxes[:,3])/2.,(-1,1))
        w = np.reshape(boxes[:,2]-boxes[:,0],(-1,1))
        h = np.reshape(boxes[:,3]-boxes[:,1],(-1,1))
        return np.hstack((x,y,w,h))
    else:
        return np.array([])
    
def associate(args):
    '''
    Associates the 2D detections in successive frames to assign a track ID.
    '''
    print("Association will be done by {} using {} threshold".format(args.association_metric,args.iou_thr))
    category_index = label_map_util.create_category_index_from_labelmap(args.path_to_labels, use_display_name=True)
    
    pred_folder = args.path_to_predictions
    category_names=[]
    category_ind = {}
    for key in category_index.keys():
        category_names.append(category_index[key]['name'])
        category_ind[category_index[key]['name']]=key
    category_names = [category_index[key]['name'] for key in category_index.keys()]
    object_lists = hist_lib.read_predictions(pred_folder,class_of_interest=category_names)
    
    ## Get frame ids of all predictions
    frame_ids = set([])
    for cat_name in category_names:
        frame_ids = set.union(frame_ids, set(object_lists[cat_name].keys()))
    frame_ids = list(frame_ids)
    frame_ids.sort()
    
    ## Generate results list to be used for associations
    results_all=[[np.array([]),np.array([]),np.array([])] for i in range(max(frame_ids)+1)]
    image_arr=[[np.array([]),np.array([]),np.array([])] for i in range(max(frame_ids)+1)]
    for frame_id in frame_ids:
        boxes=[]
        scores=[]
        classes=[]
        for cat_name in category_names:
            ## Objects of the frame id that belongs to the current class
            objs_frame = object_lists[cat_name][frame_id]
            for obj in objs_frame:
                if args.association_metric == 'deep_sort':
                    #boxes.append(list(obj.box2d_xywh))
                    boxes.append(list(obj.box2d))
                else:
                    boxes.append(list(obj.box2d))
                scores.append(obj.score)
                classes.append(category_ind[obj.type.lower()])
        results_all[frame_id] = [np.array(boxes),np.array(scores),np.array(classes)]
        if args.save_images:
            image_arr[frame_id] = [os.path.join(args.path_to_images,'{:06d}.png'.format(frame_id))]
    
    if args.save_images:
        if os.path.isdir(args.output_images):
            print("Output images folder exists!")
        else:
            os.makedirs(args.output_images)
    if args.association_metric == 'iou' or args.association_metric == 'dist': 
        track_ids = misc.assign_tracking_ids_offline(results_all,images_array=image_arr,iou_threshold=args.iou_thr,folder_name=args.output_images,
                           scr_threshold=args.scr_thr,category_index=category_index,metric=args.association_metric)
    elif args.association_metric == 'sort':
        numerator = 99999
        tracker = Sort()
        track_ids=[]
        i_frame = 0
        for res in results_all:
            b,s,c = res # boxes, scores, classes
            if len(b)!=0:
                bs = np.asarray(np.hstack((b,np.reshape(s,(-1,1))))) # should be n,4+1
            else:
                bs = []
            t = tracker.update(bs)

            ids = np.zeros((len(b)),dtype=int)-1
            for row_t in t:
                row_id_in_b = np.argmin(np.sum(abs(b-row_t[:-1]),axis=1)) # find the correspondance between the given bbox and the tracker's bbox
                ids[row_id_in_b] = int(row_t[-1])
                results_all[i_frame][0][row_id_in_b] = row_t[:-1]
            for i_ids,ind in enumerate(ids):
                if ind==-1:
                    ids[i_ids]=numerator 
                    numerator-=1
            track_ids.append(list(ids))
            
            i_frame+=1
    elif args.association_metric == 'deep_sort':
        numerator = 99999
        image_paths = [os.path.join(args.path_to_images,'{:06d}.png'.format(frame_id)) for i in range(max(frame_ids)+1)]
        cfg2 = get_config() # To create an empty config
        cfg2.merge_from_file("../deep_sort/configs/deep_sort.yaml") # read the config file
        tracker = build_tracker(cfg2, use_cuda="use_cuda")
        track_ids=[]
        for i_frame,res in enumerate(results_all):
            fr_path = image_paths[i_frame]
            frame = cv2.imread(fr_path)
            track_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            b,s,c = res # boxes, scores, classes
            if len(b)!=0:
                bs = np.asarray(np.hstack((b,np.reshape(s,(-1,1))))) # should be n,4+1
            else:
                bs = []
            
            if len(b)== 0:
                t = []
            else:
                t = tracker.update(xyxy2xywh(b), s, track_im)
            #results_all[i_frame][0] = xywh2xyxy(b)
            # ASsociate tracker boxes to gt boxes to assign object IDs
            ids = np.zeros((len(b)),dtype=int)-1
            for row_t in t:
                row_id_in_b = np.argmin(np.sum(abs(b-row_t[:-1]),axis=1)) # find the correspondance between the given bbox and the tracker's bbox
                ids[row_id_in_b] = int(row_t[-1])
                try:
                    results_all[i_frame][0][row_id_in_b] = row_t[:-1]
                except:
                    IPython.embed()
            for i_ids,ind in enumerate(ids):
                if ind==-1:
                    ids[i_ids]=numerator 
                    numerator-=1
            track_ids.append(list(ids))

    else:
        pass
        

    
    prediction_str = ''
    for i_id, frame_id in enumerate(frame_ids):
        ## Frame ID is the ind of the image, from the name
        ## i_id is the index of entries in the results_all that corresponds to frame_id
        boxes_frame = results_all[frame_id][0]
        scores_frame = results_all[frame_id][1]
        classes_frame = results_all[frame_id][2]
        track_id_frame = track_ids[frame_id]
        if args.save_images:
            if len(boxes_frame)==0:
                x1 = np.random.rand()
                y1 = np.random.rand()
                box_cur = np.array([[x1,y1,x1+0.01,y1+0.01]])
                scr_cur = np.array([0.01])
                cls_cur = np.array([1])
            else:
                box_cur = boxes_frame
                scr_cur = scores_frame
                cls_cur = classes_frame
            misc.show_image_with_label(image_arr[frame_id][0],box_cur,save='{}/img{}.png'.format(args.output_images,frame_id),
                                       class_names=cls_cur,
                                      box_indices=track_id_frame, scores=scr_cur)
        for box,scr,cls_id,trck in zip(boxes_frame,scores_frame,classes_frame,track_id_frame):
            if args.with_score:
                prediction_str+='{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(frame_id,trck,category_index[cls_id]['name'].capitalize(),\
                                                 -1,-1,-10,\
                                                 box[0],box[1],box[2],box[3],\
                                                 -1, -1, -1,\
                                                 -1000,-1000,-1000,\
                                                 -10, scr)
            else:
                prediction_str+='{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(frame_id,trck,category_index[cls_id]['name'].capitalize(),\
                                                 -1,-1,-10,\
                                                 box[0],box[1],box[2],box[3],\
                                                 -1, -1, -1,\
                                                 -1000,-1000,-1000,\
                                                 -10)
    prediction_str = prediction_str[:-1]
    output_folder = os.path.dirname(args.output_path)
    
    if os.path.isdir(output_folder):
        print("Prediction folder exists!")
    else:
        os.makedirs(output_folder)
    
    with open(args.output_path,'w') as file_name:
        file_name.write(prediction_str)
    
    ## These are for evaluation of tracking: the ABD3DMOT evaluation code needs evaluate_tracking.seqmap.val file
    evaluation_file_name = os.path.join(output_folder,'evaluate_tracking.seqmap.val')
    
    d_ids = []
    file_exists = os.path.isfile(evaluation_file_name)
    if file_exists:
        with open(evaluation_file_name,'r') as eval_filename:
            for line in eval_filename:
                line = line.strip()
                #IPython.embed()
                line_list = line.split(' ')
                if len(line_list)>=4:
                    d_id, _, st_frm, end_frm = line_list[0:4]
                    d_ids.append(int(d_id))
    else:
        print('no evaluation mapping file for tracking detected!')
    
    d_id_self =int(os.path.basename(args.output_path).split('.')[0])
    if d_id_self not in d_ids:
        with open(evaluation_file_name,'a') as eval_filename:
            if file_exists:
                eval_filename.write('\n')
            else:
                pass
            
            eval_filename.write('{:04d} empty {:06d} {:06d}'.format(d_id_self,min(frame_ids),max(frame_ids)))
    else:
        print('This drive is already in the list!')
            
            
    
    

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_predictions',help='Path to the text files that the predictions written in. The names of txt files determines the order. The names are assumed to be in KITTI format.')
    parser.add_argument('--path_to_labels',help='Path to the label map of the trained network.')
    parser.add_argument('--iou_thr', default=0.55, type=float, help='IoU threshold to assign objects in successive frames. ')
    parser.add_argument('--scr_thr', metavar='N', type=float, nargs='+',help='A list of score thresholds for each class in the label map. Only the predictions above these thresholds will be associated. The thresholds of classes will be called using the indices in label maps.')
    parser.add_argument('--path_to_images',help='Path to the folder that contains images on which track ids will be drawn.')
    parser.add_argument('--output_images',help='Path showing where the images will be saved after the tracks are written on those.')
    parser.add_argument('--output_path',help='Path to the txt file to write the associated object predictions with track IDs in the KITTI tracking dataset format. Pay attention to with_score tag.')
    parser.add_argument('--with_score', action='store_true', help='To save predictions into txt files with the prediction score in KITTI tracking format.')
    parser.add_argument('--image_format',default='png',help='Format of the images that will be tested. By default they are considered as png. It can be jpg, jpeg, etc. ')    
    parser.add_argument('--save_images', action='store_true', help='To save images with the predicted boxes drawn. Images are saved in the images folder of output_path')
    parser.add_argument('--association_metric',default='iou',help='iou, dist, sort. If iou, the association between boxes_proposal and boxes_gt will be done using the intersection over union values between them. If dist, euclidean distance between the centers of the boxes will be used instead. The value used for dist is not directly euclidean distance, but 1-normalized_euc_dist. ')
    args = parser.parse_args()
    
    associate(args)
