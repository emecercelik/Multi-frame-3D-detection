#!/bin/bash

#--path_to_predictions	: Path to the text files that the predictions written in. The names of txt files determines the order. The names are assumed to be in KITTI format.
#--path_to_labels	: Path to the label map of the trained network.
#--iou_thr		: A list of IoU thresholds. The IoU values above the thresholds will be counted as correct detections. The thresolds are given for every class and the indices of classes defined in the label map will be used while calling the thresholds.
#--scr_thr		: A list of score thresholds. Only the predictions that have a greater prediction score will be used while calculating recall. The indices of classes, given in the label map whose path defined with path_to_labels, will be used while taking this scores.
#--path_to_gts	        : Path to the ground-truth text files of frames in KITTI format.

DRIVE_ID=0011
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 
				
DRIVE_ID=0015
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 

DRIVE_ID=0016
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 

DRIVE_ID=0018
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 
				
				


DRIVE_ID=0011
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/pretrained_kitti/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 
				
DRIVE_ID=0015
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/pretrained_kitti/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 

DRIVE_ID=0016
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/pretrained_kitti/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 

DRIVE_ID=0018
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/pretrained_kitti/$DRIVE_ID/predictions/ \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
				--iou_thr 0.0 0.7 0.5 0.5 \
				--scr_thr 0.0 0.8 0.4 0.4 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_ID.txt 

