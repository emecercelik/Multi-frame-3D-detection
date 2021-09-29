#!/bin/bash
#--path_to_predictions 	: Path that contains track results in KITTI format. path/0011.txt etc.
#--path_to_gts 		: Path that contains ground-truth track information in KITTI format. path/0011.txt etc.
#--two_d		: Flag to evaluate 2D tracking. If 3D is also set, this will be overwritten as False.
#--three_d		: Flag to evaluate 3D tracking.

METRIC="dist"
THR="0.6"
TRAINING_FOLDER="train_v6"

python evaluate_kitti3dmot.py 	--path_to_predictions /root_2d_log/validation/$TRAINING_FOLDER/predictions_$METRIC/$THR/track_labels \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02 \
				--two_d 
				#--three_d
