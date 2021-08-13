#!/bin/bash

## Arguments
# --demo 		: Run demo
# --gen_train 		: Generate KITTI train split frustum data with perturbed GT 2D boxes
# --gen_val 		: Generate KITTI val split frustum data with GT 2D boxes 
# --gen_vkitti_train 	: To generate training data from Virtual KITTI
# --gen_vkitti_val 	: To generate val data from Virtual KITTI
# --car_only'		: Only generate cars; otherwise cars, peds and cycs
# --perturb2d 		: To apply 2D perturbation as augmentation
# --perturb3d 		: To apply 3D perturbation as augmentation
# --video_train 1 6 20 	: To generate training data from Virtual KITTI drives of 1,6, and 20
# --video_val 2 18 	: To generate val data from Virtual KITTI drives of 2 and 18
# --name_train _2d3d 	: To add a postfix(_2d3d here) to the name of saved pickle file for training data
# --name_val _2d3d 	: To add a postfix (_2d3d here) to the name of saved pickle file for val
# --num_point		: Number of points per frustum [default: 1024] only for vkitti
# --vkitti_path 	: Path to the Virtual KITTI dataset.
# --kitti_path 		: Path to the KITTI dataset. Inside should be training/-calib,-image_2,-label_2,-velodyne. default: 'dataset/KITTI/object'
# --gen_tracking_train : To generate training data from KITTI Tracking.
# --gen_tracking_val   : To generate validation data from KITTI Tracking.
# --tracking_path      : Path to the KITTI tracking dataset. Inside should be data_tracking_<name>/<split>/<name>. <name>:calib,image_02,label_02,velodyne. <split>:training,testing
# --rgb_detection_path	: Path to the KITTI tracking rgb_detections. Inside should be <drive_name>/rgb_detection.txt files according to drives stated in video_val
# --gen_val_rgb_detection	: Generate val split frustum data with RGB detection 2D boxes 
# --apply_num_point_thr: To exclude frustums that have less than minimum and more than maximum number of points given with the flags
# --max_num_point_thr	: Maximum number of points per frustum if apply_num_point_thr is set
# --min_num_point_thr	: Minimum number of points per frustum if apply_num_point_thr is set
# --augmentX		: Shows how many times a 2D box will be augmented

# To prepare frustum data from KITTI Tracking dataset

			
#python prepare_data.py --gen_tracking_train \
#			--gen_tracking_val \
#			--video_train 0 2 3 4 5 6 7 8 9 10 12 13 14 17 19 20\
#			--video_val 11 15 16 18\
#			--tracking_path '/kitti_root_tracking'\
#			--perturb2d \
#			--name_train '_kitti_tracking_augment5' \
#			--name_val '_kitti_tracking_augment5' 
			
#python prepare_data.py --gen_tracking_val \
#			--gen_val_rgb_detection \
#			--video_val 11 15 16 18\
#			--tracking_path '/kitti_root_tracking'\
#			--name_val '_kitti_2dgt_sort' \
#			--rgb_detection_path /root_2d_log/rgb_detections_from_predictions/gt_boxes
			
python prepare_data.py --gen_tracking_val \
			--gen_val_rgb_detection \
			--video_val 11 15 16 18\
			--tracking_path '/kitti_root_tracking'\
			--name_val '_kitti_2dgt_deepsort' \
			--rgb_detection_path /root_2d_log/rgb_detections_from_predictions/deep_sort
			
    


