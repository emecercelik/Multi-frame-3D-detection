#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:08:21 2019

@author: emec
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from deteva import hist_lib
import IPython

def center2corner(boxes,stacked=True,dtype=None):
    """
    boxes: A tensor containing bounding boxes with the shape of (?,4) in the order of [x_c,y_c,w,h]
    
    Returns the boxes with the same shape in the order of [x_min,y_min,x_max,y_max]
    """
    
    x_c,y_c,w,h = tf.unstack(boxes,axis=1)
    x_min = x_c - w/2
    y_min = y_c - h/2
    x_max = x_c + w/2
    y_max = y_c + h/2
    if dtype is not None:
        x_min = tf.cast(x_min,dtype)
        y_min = tf.cast(y_min,dtype)
        x_max = tf.cast(x_max,dtype)
        y_max = tf.cast(y_max,dtype)
        
    if stacked:
        return tf.stack([x_min,y_min,x_max,y_max],axis=1)
    else:
        return x_min,y_min,x_max,y_max

def corner2center(boxes,stacked=True,dtype=None):
    """
    boxes: A tensor containing bounding boxes with the shape of (?,4) in the order of [x_min,y_min,x_max,y_max]
    
    Returns the boxes with the same shape in the order of [x_c,y_c,w,h]
    """
    x_min,y_min,x_max,y_max = tf.unstack(boxes,axis=1)
    x_c = (x_min+x_max)/2
    y_c = (y_min+y_max)/2
    w = x_max - x_min
    h = y_max - y_min
    if dtype is not None:
        x_c = tf.cast(x_c,dtype)
        y_c = tf.cast(y_c,dtype)
        w = tf.cast(w,dtype)
        h = tf.cast(h,dtype)
    if stacked:
        return tf.stack([x_c,y_c,w,h],axis=1)
    else:
        return x_c,y_c,w,h
def tf_euc_dist(boxes1,boxes2):
    """
    boxes1 : bounding boxes with a shape of (m,4) in the [x_min,y_min,x_max,y_max] format
    boxes2 : bounding boxes with a shape of (n,4) in the [x_min,y_min,x_max,y_max] format
    
    Returns (1-euclidean_distance/max_dist) of bounding boxes with a shape of (m,n)
    """
    x1c,y1c,w1,h1 = tf.split(corner2center(boxes1,stacked=True),4, axis=1)
    x2c,y2c,w2,h2 = tf.split(corner2center(boxes2,stacked=True),4, axis=1)
    
    d_x = tf.square(tf.subtract(x1c,tf.transpose(x2c)))
    d_y = tf.square(tf.subtract(y1c,tf.transpose(y2c)))
    d2 = tf.add(d_x,d_y)
    d = tf.sqrt(d2)
    max_d = tf.reduce_max(d)
    norm_d = tf.divide(d,200.)
    return tf.subtract(1.0,norm_d),d
    
    

def tf_iou(boxes1, boxes2):
    """
    from https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    
    Calculates vectorized intersection area over union area of bounding boxes among each other
    
    boxes1 : bounding boxes with a shape of (m,4) in the [x_min,y_min,x_max,y_max] format
    boxes2 : bounding boxes with a shape of (n,4) in the [x_min,y_min,x_max,y_max] format
    
    Returns Intersection over Union (IoU) values of bounding boxes with a shape of (m,n)
    
    """
    x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1)

    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21 )

    iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

    return iou,boxAArea, boxBArea, interArea


def perc_proposal_iou(proposals,gts,thr):
    '''
    This method is to evaluate region proposal networks. Evaluation is done by checking if all of the ground-truth boxes have at least one proposal that overlaps with an IoU larger than the thr value. 
    
    proposals: Object proposals from a region proposal network with size (m,4) in the [x_min,y_min,x_max,y_max] format.
    gts : Ground-truth object bounding boxes with size (n,4) in the [x_min,y_min,x_max,y_max] format.
    thr : A threshold value for IoU to evaluate if there is at least one proposal that have a larger IoU than this.
    
    Returns a percentage showing that how many of the ground-truth boxes are overlapped by proposals with an IoU greater than the given threshold.
        
    '''
    ## Calculate Intersection over Union values between proposals and ground-truths
    iou,_,_,_ = tf_iou(proposals,gts) # -> (m,n)
    ## Check if IoU values are greater than the provided threshold
    iou_bool = tf.greater(iou,thr)
    ## Reduce the bool values in the direction of proposals  (rows)
    reduced_iou = tf.reduce_any(iou_bool,axis=0) 
    ## Cast the boolean values into float numbers 
    num_reduced = tf.cast(reduced_iou,tf.float32)
    ## Sum to find out how many of the ground-truths are overlapped by proposals
    sum_bool = tf.reduce_sum(num_reduced)
    ## Get how many ground-truth boxes are there
    num_gts = tf.cast(tf.shape(gts)[0],tf.float32)
    ## Calculate the percentage
    perc = tf.where(num_gts==0.0,1.0,sum_bool/num_gts)
    return perc

def rcnn_create_obj_bg_masks(boxes_proposal, boxes_gt, high_threshold=0.5, low_threshold=0.1,all_gt_assigned=True,metric='iou'):
    """
    To assign bbox proposals to the gt boxes according to Intersection over Union (IoU) metric.
    
    boxes_proposals: A list of proposal bounding boxes in the shape of (n,4) each of which in 
        [x_min,y_min,x_max,y_max] order.
    boxes_gt: A list of ground-truth bounding boxes in the shape of (m,4) each of which in 
        [x_min,y_min,x_max,y_max] order.
    high_threshold: A scalar that IoUs between proposal boxes and ground-truth boxes greater than this scalar
        will be considered as an object.
    low_threshold: IoUs between proposal boxes and ground-truth boxes that are between high_threshold and
        this scalar will be considered as a background.
    all_gt_assigned: if True all ground-truths will be assigned to at least one proposals independent of the IoU values. if False, assignment will be done considering only the respective IoU values.
    metric : iou or dist. If iou, the association between boxes_proposal and boxes_gt will be done using the intersection over union values between them. If dist, euclidean distance between the centers of the boxes will be used instead. The value used for dist is not directly euclidean distance, but 1-normalized_euc_dist. 
    Returns 
        positive_props, positive_gts, obj_mask, bg_mask, indx_for_props, indx_for_gts
        high_threshold = 0.39 # above this level of IoU indicates the bbox proposal as an object's
        low_threshold = 0.16 # below this level, the proposals will be considered as bg
    
        positive_props: Proposals that are assigned to a gt box 
            [[x_min0,y_min0,x_max0,y_max0],[x_min1,y_min1,x_max1,y_max1], ...] 
        positive_gts  : Ground-truth (gt) boxes of the positive proposals
            [[gt_x_min0,gt_y_min0,gt_x_max0,gt_y_max0],[gt_x_min1,gt_y_min1,gt_x_max1,gt_y_max1], ...]
        obj_mask      : Mask to get the positive proposals from the boxes_proposal list
        bg_mask       : Mask to get the negative (backgrounds) proposals from the boxes_proposal list
        indx_for_props: Indices of each proposal that corresponds to the gt boxes
            [proposal id for gt box 0, proposal id for gt box 1, proposal id for gt box 2, ... ]
        indx_for_gts: Indices of each gt box that is assigned to the proposals 
            [gt box id for proposal 0, gt box id for proposal 1, gt box id for proposal 2, ... ]
    
    """
    ## n box proposals and m ground-truth boxes
    dist=0
    # Calculate IoU (n x m matrix)
    if metric=='iou':
        IoU,_,_,_ = tf_iou(boxes_proposal,boxes_gt)
    elif metric=='dist':
        IoU,dist = tf_euc_dist(boxes_proposal,boxes_gt)
        

    # Create indices according to the number of box proposals and grount-truth boxes
    indx_props = tf.range(0,tf.shape(boxes_proposal)[0]) # (n indices) -> 0,1,2, ...,n
    indx_gt = tf.range(0,tf.shape(boxes_gt)[0]) # (m indices) -> 0,1,2, ...,m

    # Create a mesh grid for the indices of IoU matrix
    mesh_gt,mesh_props = tf.meshgrid(indx_gt,indx_props) 

    # Generate IoU mask that selects the entries greater than the bigger threshold 
    # This mask doesn't consider the ground-truth boxes that doesn't have proposal boxes assigned
    mask_IoU_only_obj_greater = IoU > high_threshold

    # Mask considering the maximum IoUs of each ground-truth if doesn't have a proposal assigned with an IoU greater than
    # the larger threshold

    ## mask to check whether all the gt boxes have at least a proposal assigned
    mask_gt_all_assigned = tf.reduce_any(mask_IoU_only_obj_greater,axis=0) # [False, True, True, True, False, False]
    nothing_assigned = -1*tf.ones(tf.shape(mask_gt_all_assigned),dtype=tf.float64)
    max_thr_cols = tf.reduce_max(IoU,axis=0) # Maximum IoUs for each gt box

    # assign max of each column if the corresponding gt doesn't have a proposal assigned to itself
    nothing_assigned_cast = tf.cast(nothing_assigned,max_thr_cols.dtype)
    max_assigned_gt = tf.where(mask_gt_all_assigned,nothing_assigned_cast,max_thr_cols)

    # generate a mask indicating places of proposals that have the max IoU with the ground-truth boxes whose
    # IoU doesn't exceed the larger threshold
    mask_find_max_thr_cols = tf.equal(max_assigned_gt,IoU)

    # Final mask showing the bounding box proposals that will be considered as object bboxes 
    # and the ground-truth boxes that they are assigned to
    if all_gt_assigned:
        final_mask = tf.logical_or(mask_IoU_only_obj_greater,mask_find_max_thr_cols)
    else:
        final_mask = mask_IoU_only_obj_greater

    # Indices showing the bbox proposals considered as objects and ground-truth boxes that the proposals are assigned to
    indx_for_props = tf.boolean_mask(mesh_props,final_mask)
    indx_for_gts = tf.boolean_mask(mesh_gt,final_mask)

    # The object bbox proposals and ground-truth boxes they are assigned to
    positive_props = tf.gather(boxes_proposal,indx_for_props)
    positive_gts = tf.gather(boxes_gt,indx_for_gts)

    ## to generate objectness mask (cls information to calculate cls-loss)
    obj_mask = tf.reduce_any(final_mask,axis=1)

    ## to generate bg mask (cls information to calculate cls-loss) 
    max_of_ious = tf.reduce_max(IoU,axis=1)
    bg_mask_init = tf.logical_and(tf.greater_equal(max_of_ious,low_threshold),tf.less(max_of_ious,high_threshold))
    # to avoid containing the proposals that have an IoU smaller than low_threshold with a gt box, but this IoU is
    # the greatest of the gt box have
    bg_mask = tf.logical_and(tf.logical_xor(obj_mask,True),bg_mask_init) 
    return positive_props, positive_gts, obj_mask, bg_mask, indx_for_props, indx_for_gts

def filter_threshold(scr_thr,cls_ind,cls_list,scr_list):
    '''
    To get indices of the detections that belong to different classes and filtered with different thresholds 
    scr_thr: The min score that will be included in the index list of detections
    cls_ind: Index of the class that will be considered
    cls_list: An array of classes of detections. For each detection, there should be one class prediction. N-length array for N detections
    scr_list: An array of scores of detections. For each detection, there should be one score. N-length score list for N detections
    
    Returns a list of indices showing which detections are above the given threshold and belongs to the given class at the same time
    '''
    scr_ind  = np.argwhere(scr_list>=scr_thr).flatten()
    cls_filt = np.argwhere(cls_list[scr_ind] == cls_ind).flatten()
    scr_ind = scr_ind[cls_filt]
    return scr_ind


def assign_tracking_ids_offline(results_array,images_array=[],iou_threshold=0.55,folder_name='video_images',
                                scr_threshold=None,category_index=None,metric='iou'):
    '''
    results_array : Contains predicted boxes, scores, and classes of images as a list. 
        Ex: [[predicted_boxes_frame_0,scores_frame_0,classes_frame_0],[predicted_boxes_frame_1,scores_frame_1,classes_frame_1],...]
        predicted_boxes = A list with shape (n,4) in the (x1,y1,x2,y2) format.
        scores = list of n scores that corresponds to each prediction in the predicted_boxes. length=n 
        classes= A list of n classes that corresponds to each prediction in the predicted_boxes. length=n
    images_array : A list that contains images in the same order with the results_array. This will be used to save the images annotated with the assigned indices of boxes, scores, and classes. Ex: [[img_array_0],[img_array_1],...]
    iou_threshold: Threshold of Intersection over Union that will be calculated between objects. Objects above this threshold will be associated to each other.
    folder_name  : Name of the folder that the images will be stored in.
    metric : iou or dist. If iou, the association between boxes_proposal and boxes_gt will be done using the intersection over union values between them. If dist, euclidean distance between the centers of the boxes will be used instead. The value used for dist is not directly euclidean distance, but 1-normalized_euc_dist. 
    Returns a list that contains tracking ids of objects in each frame in the order of provided bounding boxes.
        Ex: [[0,1],[0,1,2],[1,2,3],...] -> Tracking ids assigned to [[obj_0_frame_0,obj_1_frame_0],[obj_0_frame_1,obj_1_frame_1,obj_2_frame_1],[obj_0_frame_2,obj_1_frame_2, obj_2_frame_2], ...]
        
    Problems
    1. If an object disappears and appears again in a guture frame, it is counted as a new object
    2. If two objects of the current frame is associated with the same object of the current frame, the id given to those are the same
    Sol2. P2 can be solved by assigning the object of current frame with a higher score to the object of previous frame that is assigned to two or more objects in the current frame

    '''
    print("Association will be done by {} using {} threshold".format(metric,iou_threshold))
    if len(images_array)!= 0:
        save_images = True
    else:
        save_images = False
    ## Tensorflow graph for associations
    print(save_images)
    high_thr = iou_threshold # IoU above which will be associated
    low_thr = 0.1 # Not used here
    b1Ph = tf.placeholder(dtype=tf.float32, shape=[None,4], name='b1Ph')
    b2Ph = tf.placeholder(dtype=tf.float32, shape=[None,4], name='b2Ph')
    ## Function below returns 
    # as_box1: The associated boxes from b1Ph, as_box2: The respective boxes from b2Ph
    # mask_box1: Mask to get associated boxes from b1Ph, mask_bg: not used here
    # box2_ind_for_box1_assoc: Indices of boxes from b2Ph that are associated to the boxes from b1Ph
    # box1_ind_for_box2_assoc: Indices of boxes from b1Ph that are associated to the boxes from b2Ph
    as_box1, as_box2, mask_box1, mask_bg, box2_ind_for_box1_assoc, box1_ind_for_box2_assoc = rcnn_create_obj_bg_masks(\
                                                                                               b1Ph,b2Ph, high_thr,low_thr,\
                                                                                                all_gt_assigned=False,metric=metric)
    ## Calculations for associations
    real_indices = [] # Tracking IDs will be appended to this list
    
    filter_indices=[] # Indices of the boxes that are higher than the threshold
    num_detections=[] # Number of all detections for the given frames
    
    max_ind = 0 # Max index assigned so far
    if category_index is not None:
        cls_indices = category_index.keys()
    # Iterate through all frames
    for i_img,result_img in enumerate(results_array):
        ## if this is not the first image, store the previous frame's results
        if i_img>0:
            #img_ind_prev = i_img - 1 # Image index of previous frame
            box_prev = box_cur # Predicted boxes of the previous frame
            #scr_prev = scr_cur
            #cls_prev = cls_cur

        box_cur, scr_cur, cls_cur = result_img[0:3] # Get current data
        if scr_threshold is not None and category_index is not None:
            indices = np.array([])
            for cls_ind in cls_indices:
                indices = np.concatenate((indices,filter_threshold(scr_threshold[cls_ind],cls_ind,cls_cur,scr_cur)))
            
            indices = indices.astype(int)
        filter_indices.append(indices)
        num_detections.append(len(box_cur))
        
        box_cur = box_cur[indices]
        scr_cur = scr_cur[indices]
        cls_cur = cls_cur[indices]
        
        if len(box_cur) == 0:
            ## Add dummy box
            x1 = np.random.rand()
            y1 = np.random.rand()
            box_cur = np.array([[x1,y1,x1+0.01,y1+0.01]])
            scr_cur = np.array([0.01])
            cls_cur = np.array([list(cls_indices)[0],list(cls_indices)[0]])
        ## If it is not the first image, calculate the IoU and assignments 
        if i_img>0:
            with tf.Session() as sess:
                as1, as2, mask_as1, _, b2_ind_for_b1, b1_ind_for_b2 = sess.run([as_box1, as_box2, mask_box1, mask_bg, box2_ind_for_box1_assoc, box1_ind_for_box2_assoc],
                        feed_dict={b1Ph:box_prev,b2Ph:box_cur})
                #as1, as2, mask_as1, _, b2_ind_for_b1, b1_ind_for_b2 = sess.run([as_box1, as_box2, mask_box1, mask_bg, box2_ind_for_box1_assoc, box1_ind_for_box2_assoc],
                #        feed_dict={b1Ph:box_prev,b2Ph:box_cur})

            print("Index: {}".format(i_img))
            print("Boxes of prev frame:\n",box_prev)
            print("Associated boxes 1:\n",as1)
            print("Associated boxes 2:\n",as2)
            print("Boxes B1:\n",box_cur)
            print("B2 indices for B1:\n",b2_ind_for_b1)
            print("B1 indices for B2:\n",b1_ind_for_b2)
        
        ## Get the current image from the images 
        if save_images is True:
            img = images_array[i_img][0]
        
        ## Save the first image
        if i_img == 0:
            real_indices.append([i for i in range(len(scr_cur))])
            max_ind = np.max(real_indices)
            if save_images is True:
                if len(img)!=0:
                    show_image_with_label(img,box_cur,save='{}/img{}.png'.format(folder_name,i_img),class_names=cls_cur,
                                      box_indices=real_indices[-1], scores=scr_cur)
        
        ## Calculate the new indices of the current frame if the image ind is different from zero. 
        else:
            ### To discard multiple assignments
            ## Set of indices (doesn't contain multiple instances)
            set_ind_prev_frame = list(set(b2_ind_for_b1))
            ## This means two objects of the current frame are assigned to the same object of the prev frame
            if len(set_ind_prev_frame)!=len(b2_ind_for_b1):
                np_b2_ind_for_b1 = np.array(b2_ind_for_b1)
                ind_discarded = []
                for ind in set_ind_prev_frame:
                    ind_same_objs = np.hstack(np.argwhere(np_b2_ind_for_b1 == ind)) # Indices of the same objects
                    ## If the length is greater than 1, this means there are multiple assignment
                    if len(ind_same_objs)>1:
                        print('ind_same_objects', ind_same_objs)
                        scrs = np.array(scr_cur)[np.array(b1_ind_for_b2)[ind_same_objs]]  ## Scores of the objects in the current frame that are assigned to the same object of prev frame
                        ind_obj_hgst_scr = [ind_same_objs[np.argmax(scrs)]] # Index of the object with the highest score among objects that are assigned to the same object of the prev frame
                        print(ind_same_objs)
                        print(ind_obj_hgst_scr)
                        ind_discarded += list(set(list(ind_same_objs))-set(list(ind_obj_hgst_scr)))

                ## Indices that will stay
                ind_stay = [i for i in range(len(b2_ind_for_b1)) if i not in ind_discarded]
                b2_ind_for_b1 = list(np.array(b2_ind_for_b1)[ind_stay])
                b1_ind_for_b2 = list(np.array(b1_ind_for_b2)[ind_stay])
                
            ### Calculate indices after discarding multiple assignments
            ## Real Indices of objects from previous frame that have been associated to one of the objects in the current frame
            ind_from_prev_frame = list(np.array(real_indices[-1])[b2_ind_for_b1])
            l_b1 = len(box_cur) # Number of objects in the current frame
            ind_b1 = np.array([i for i in range(l_b1)]) # Indices of objects in the current frame
            ind_obj_not_assoc = list(set(ind_b1) - set(b1_ind_for_b2) ) # indices of the objects in the current frame that are not associated to an object in the previous frame
            len_obj_not_assoc = len(ind_obj_not_assoc) # Number of objects in the current frame that are not associated to an object in the previous frame
            ## Generate new indices for non-associated objects in the current frame
            new_indices =[]
            if len_obj_not_assoc>0:
                new_indices = [max_ind+1+i for i in range(len_obj_not_assoc)]
                max_ind = np.max(new_indices)

            ## b1_ind_for_b2 contains the indices of objects in the current frame associated to the objects with indices in the b2_ind_for_b1
            ## Associate the correct real_indices from the previous frame to the objects in the current frame
            ## Ex: b1_ind_for_b2 = [2,0,1,4,3] b2_ind_for_b1 = [0,1,2,3,4] means the obj-2 in the current frame is associated to the obj-0 of prev. frame, obj-0 of current frame is associated to the obj-1 of prev frame
            ind_b1[b1_ind_for_b2] = ind_from_prev_frame 

            ind_b1[ind_obj_not_assoc] = new_indices # Assign new indices to the non-associated objects of the current frame

            ## Append the real indices of the objects in the current frame        
            real_indices.append(list(ind_b1))
            print("Real indices of current frame:",real_indices[-1])
            print('')
            ## Save the image with annotations
            if save_images:
                if len(img)!=0:
                    show_image_with_label(img,box_cur,save='{}/img{}.png'.format(folder_name,i_img),class_names=cls_cur,\
                                  box_indices=real_indices[-1],scores=scr_cur)
        
        ## all_track_ids is a version of real_indices extended with the objects that are below threshold. Those are assigned new track ids that don't match with any other object
        all_track_ids = []
        for frame_track_ids,num_objs,filter_inds in zip(real_indices,num_detections,filter_indices):
            all_track_ids.append([-1 for i in range(num_objs)]) # Assign dummy ids to all objects in the current frame
            # write the associated objects' track ids into the new list
            for ii,ind in enumerate(filter_inds):
                all_track_ids[-1][ind] = frame_track_ids[ii]
            ## fill in the non-associated objects with new track ids
            for ii,t_id in enumerate(all_track_ids[-1]):
                if t_id == -1:
                    all_track_ids[-1][ii]=max_ind+1
                    max_ind = max_ind + 1
                    
                
            
    return all_track_ids
            
def show_image_with_label(image,box_proposals=None,save=None, class_names=[], box_indices=[],scores=[],line_width=2):
    """
    image: Numpy array of the RGB image with a shape of (h,w,ch=3)  or path to the image
    box_proposals: None if there are no proposals to draw. Otherwise list of [x1,y1,x2,y2] bboxes
        Example: [[0,0,5,5],[10,12,23,25]] or [[3,5,12,16]]
    save: either a name that the image to be saved as or None not to save 
    class_names: Class names of the proposed boxes. Length of this list should be the same with the number of proposals
    box_indices: Indices of proposed boxes. Length of this list should be the same with the number of proposals
    line_width : Width of the line to draw boxes
    
    Example: 
        dataset.show_image(9,box_proposals=[[50,190,200,500.],[400,200,200,310]],\
            save="deneme",class_names=['car','cyclist'], box_indices=[1,0],2)

    """
    if type(image)==str:
        img = Image.open(image).convert("RGB")
    else:
        if image.dtype != np.dtype('uint8'):
            img = image.astype(np.uint8)
            
        img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype("ubuntu/Ubuntu-B.ttf", 12)
    except:
        fnt = ImageFont.load_default()
    
    if box_proposals is not None:
        '''
        if len(class_names)==0:
            class_names = [-1 for prop in box_proposals]
        if len(box_indices)==0:
            box_indices = [i for i in range(len(box_proposals))]
        if len(scores) == 0:
            scores = [-9 for i in range(len(box_proposals))]
        '''
        for i_box,box_proposal in enumerate(box_proposals):
            bx_print = list(box_proposal)
            draw.rectangle(bx_print,outline="green",width=line_width)
            if len(box_indices)!=0:
                try:
                    str_to_be_written = '{}'.format(box_indices[i_box])
                except:
                    str_to_be_written = ''
            else:
                str_to_be_written = '{}'.format(i_box)
                
            if len(class_names)!= 0:
                str_to_be_written+=': {:.0f}'.format(class_names[i_box])
            else:
                str_to_be_written+=': '.format(class_names[i_box])
            if len(scores)!= 0:
                str_to_be_written+=': {:.2f}'.format(scores[i_box])
            else:
                str_to_be_written+=': '.format(scores[i_box])
            
            draw.text([bx_print[0]+25.,bx_print[(i_box%2)*2+1]+2],str_to_be_written,font=fnt,fill=(0,153,0,128))

    if save is not None:
        
        img.save(save,"PNG")
    plt.imshow(img)
    
    
def calculate_recall(args):
    '''
    Calculates recalls and returns recall values each frame and classes separately. 
    args should contain the followings
    path_to_labels      : Path to the label map of the trained network.
    path_to_predictions : Path to the text files that the predictions written in. The names of txt files determines the order. The names are assumed to be in KITTI format.
    path_to_gts         : Path to the ground-truth text files of frames in KITTI format
    scr_thr             : A list of score thresholds. Only the predictions that have a greater prediction score will be used while calculating recall. The indices of classes, given in the label map whose path defined with path_to_labels, will be used while taking this scores. 
    iou_thr             : A list of IoU thresholds. The IoU values above the thresholds will be counted as correct detections. The thresolds are given for every class and the indices of classes defined in the label map will be used while calling the thresholds. 
    '''
    
    
    
    category_index = label_map_util.create_category_index_from_labelmap(args.path_to_labels, use_display_name=True)
    
    pred_folder = args.path_to_predictions
    gt_folder = args.path_to_gts
    category_names=[]
    category_ind = {}
    for key in category_index.keys():
        category_names.append(category_index[key]['name'])
        category_ind[category_index[key]['name']]=key
    category_names = [category_index[key]['name'] for key in category_index.keys()]
    object_lists = hist_lib.read_predictions(pred_folder,class_of_interest=category_names)
    gt_lists = hist_lib.read_ground_truth(gt_folder,class_of_interest=category_names)
    
    ## Get frame ids of all predictions
    frame_ids_pred = set([])
    for cat_name in category_names:
        frame_ids_pred = set.union(frame_ids_pred, set(object_lists[cat_name].keys()))
    frame_ids_pred = list(frame_ids_pred)
    frame_ids_pred.sort()
    
    ## Get frame ids of all gts
    frame_ids_gt = set([])
    for cat_name in category_names:
        frame_ids_gt = set.union(frame_ids_gt, set(gt_lists[cat_name].keys()))
    frame_ids_gt = list(frame_ids_gt)
    frame_ids_gt.sort()
    
    ## setup recall calculation graph
    pred_obj_ph = tf.placeholder(dtype=tf.float32, shape=[None,4], name='pred_obj_ph')
    gt_obj_ph = tf.placeholder(dtype=tf.float32, shape=[None,4], name='gt_obj_ph')
    iou_thr_ph = tf.placeholder(dtype=tf.float32,shape=[],name='iou_thr_ph')
    _recall = perc_proposal_iou(pred_obj_ph,gt_obj_ph,iou_thr_ph)
    sess = tf.Session()
    ## Generate results list to be used for associations
    recalls = {}
    for frame_id in frame_ids_gt:
        
        recalls[frame_id] = {}
        for cat_name in category_names:
            boxes=[]
            scores=[]
            classes=[]
            gt_boxes=[]
            ## Get the predicted objects and ground-truths of a frame that belongs to the given category
            pred_objs_frame = object_lists[cat_name][frame_id]
            gt_objs_frame = gt_lists[cat_name][frame_id]
            ## predicted objects frame by frame to their features
            for obj in pred_objs_frame:
                boxes.append(list(obj.box2d))
                scores.append(obj.score)
                classes.append(category_ind[obj.type.lower()])
            for obj in gt_objs_frame:
                gt_boxes.append(list(obj.box2d))
            ## filter predicted objects according to the score threshold
            indices = np.array([])
            for key in category_index.keys():
                indices = np.concatenate((indices,filter_threshold(args.scr_thr[key],key,np.array(classes),np.array(scores))))
            indices = indices.astype(int)
            filtered_pred_boxes = np.array(boxes)[indices]
            if len(filtered_pred_boxes)==0:
                filtered_pred_boxes=np.array([[0.,0.,0.5,0.5]])
            #IPython.embed()
            if len(gt_boxes)!=0.0:
                a={}
                a['recall']=sess.run(_recall,feed_dict={pred_obj_ph:filtered_pred_boxes,gt_obj_ph:gt_boxes,iou_thr_ph:args.iou_thr[category_ind[cat_name]]})
                a['n_gt'] = len(gt_boxes)
                recalls[frame_id][cat_name]=a
    
    frames = recalls.keys()
    mean_recalls=dict()
    for cat_name in category_names:
        mean_recalls[cat_name] = dict()
        mean_recalls[cat_name]['recall']=0.0
        mean_recalls[cat_name]['counter']=0.
    for frame in frames:
        for cat_name in category_names:
            try:
                #mean_recalls[cat_name]['recall']+=recalls[frame][cat_name]['recall']
                mean_recalls[cat_name]['recall']+=recalls[frame][cat_name]['recall']*recalls[frame][cat_name]['n_gt']
                mean_recalls[cat_name]['counter']+=(recalls[frame][cat_name]['n_gt']+0.0)
            except:
                #print('no instance')
                pass
    print('****')
    print(pred_folder)            
    for cat_name in category_names:
        print('Recall for {}: {:.3f} for IoU threshold: {} and Score threshold: {}'.format(cat_name, 
              mean_recalls[cat_name]['recall']/(mean_recalls[cat_name]['counter']+1.0),
              args.iou_thr[category_ind[cat_name]],
              args.scr_thr[category_ind[cat_name]]))
    sess.close()
    return recalls
    
    
