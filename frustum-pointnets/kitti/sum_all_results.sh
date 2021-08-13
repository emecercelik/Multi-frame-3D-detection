#!/bin/bash

## Arguments
#--class_names	: Names of classes of interest. Ex: car pedestrian
#--path_to_log	: Path to the log folder of the frustum pointnet. The path should contain detection_results/data/<drive_id>/data folders.
#--drive_ids	: Drive IDs that will be evaluated to generate histogram.
#--path_to_gts	: Path that contains ground-truth track information in KITTI format. path/0011.txt etc.
#--path_to_hist	: Path where the histograms will be saved.
#--inference    : If given, the predictions will be checked under inference_results instead of detection_results.
#--difficulty 	: By default ground-truth boxes from all difficulty levels are included. To see results on only some of the difficulty levels, give the difficulty id. 0: easy, 1:moderate, 2:hard, 3: unknown

# BMW: 3 4 43 44
# KITTI: 11 15 16 18
#LOG_PATH=/root_3d_log/bmw_time_series_size_check
LOG_PATH=/root_3d_log/kitti_attention_time

GTS_PATH=/kitti_root_tracking/data_tracking_label_2/training/label_02
#GTS_PATH=/root_bmw_tracking/data_tracking_label_2/training/label_02

python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s0 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3


python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s10 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3


python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s11 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3


python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s12 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3


python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s13 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3


python sum_all_results.py	--class_names car \
				--path_to_log $LOG_PATH/log_time_s14 \
				--drive_ids 11 15 16 18\
				--path_to_gts $GTS_PATH \
				--path_to_hist $LOG_PATH/ \
				--difficulty 0 1 2 3



