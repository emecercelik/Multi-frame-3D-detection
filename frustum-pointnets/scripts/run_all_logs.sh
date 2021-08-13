#!/bin/bash
## Used to test the trained model and evaluate for AP 
#--log_dir	: Path to a single log folder that contains log files of a training. This is overwritten by log_dirs if given.
#--log_dirs	: Path to log folders. Inside there should be log_* folders that contains logs of a training.
#--gt_root_dir	: Path to root dir of ground-truth tracking labels
#--pickle_name	: Name of the pickle file to be used in inference. If this is None, the pickle file used for evaluation will be used. If pickle name is given, the outputs are written in inference_results of the log_dir instead of detection_results.

# Copy model_util with kitti mean sizes for anchors
#cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py model_util.py
cp /frustum_framework/frustum-pointnets/models/model_util_bmw.py model_util.py

#python run_all_logs.py --log_dirs /root_3d_log/time_series --gt_root_dir /kitti_root_tracking/drives_in_kitti

python run_all_logs.py 	--log_dir /root_3d_log/bmw_time_series_size_check/log_time_s35 \
			--gt_root_dir /root_bmw_tracking/drives_in_kitti \
			--pickle_name frustum_caronly_tracking_bmw_rgb_detection.pickle



