#!/bin/bash

#NAME=kitti_carpedcyc_time
NAME=kitti_global_temp_fusion
python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \
				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02

#NAME=kitti_attention_max_pool
#python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \
#				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02

#NAME=kitti_attention_time
#python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \
#				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02

#NAME=kitti_attention_whl
#python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \#
#				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02

#NAME=kitti_attention_carpedcyc_time
#python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \
#				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02

#NAME=kitti_onlycar_time
#python get_all_summaries.py 	--log_dirs /root_3d_log/$NAME \
#				--gt_root_dir /kitti_root_tracking/data_tracking_label_2/training/label_02
				

