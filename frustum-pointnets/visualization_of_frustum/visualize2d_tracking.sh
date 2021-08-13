#/bin/bash

#--root_dir_tracking 	: Root directory to KITTI Tracking dataset. root_dir should contain:data_tracking_image_2/<split>/image_02/, data_tracking_label_2/<split>/label_02/,data_tracking_velodyne/<split>/velodyne/,data_tracking_calib/<split>/calib/
#--drives 		: Indices of KITTI tracking dataset drives, from which the images will be generated.
#--save_path 		: Path to save images.
#--pred_path		: Path to the predictions. This should contain folders as following: pred_path/0000/data/000000.txt,000001.txt,...;0011/data/000000.txt,000001.txt,...
#--gt_path		: Path to the KITTI tracking ground-truth labels arranged in KITTI object detection format. Inside this folder, there should be folders with drive id such that gt_path/0000/000000.txt,000001.txt,...;0001/000000.txt,000001.txt,...
LOG_DIR=kitti_attention_whl_caronly
LOG_NAME=log_time_s11
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\

LOG_DIR=kitti_onlycar_time
LOG_NAME=log_normal_s22
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_onlycar_time
LOG_NAME=log_normal_s4
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_onlycar_time
LOG_NAME=log_normal_s17
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_attention_whl
LOG_NAME=log_time_s24
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_attention_whl
LOG_NAME=log_time_s0
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_attention_whl_caronly
LOG_NAME=log_time_s3
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\


LOG_DIR=kitti_onlycar_time
LOG_NAME=log_time_s49
python viz_util_image.py \
	--root_dir_tracking /kitti_root_tracking \
	--drives 11 15 16 18\
	--save_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/imgs\
	--pred_path /root_3d_log/$LOG_DIR/$LOG_NAME/detection_results/data/\
	--gt_path /kitti_root_tracking/drives_in_kitti\



