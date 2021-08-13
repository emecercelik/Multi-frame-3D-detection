#!/bin/bash
#--gpu       : GPU to use [default: GPU 0]
#--num_point : Point Number [default: 1024]
#--model     : Model name [default: frustum_pointnets_v1]
#--model_path: model checkpoint file path [default: log/model.ckpt]
#--batch_size: batch size for inference [default: 32]
#--output    : output file/folder name [default: test_results]
#--data_path : frustum dataset pickle filepath [default: None]
#--from_rgb_detection : To run test on pickle files generated using rgb detection.
#--idx_path  : filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]
#--dump_result        : If true, also dump results to .pickle file
#--no_intensity       : Only use XYZ for training
#--vkitti    : To use virtual KITTI dataset. If this is True, tracking dataset can't be used. [default: False]
#--tracking  : To indicate whether the KITTI tracking dataset will be used. [default: False]
#--tracks    : To indicate whether to read track information (track_ids, drive_ids) from the prepared pickle files. [default:False]
#--tau       : Number of time steps in total to be processed by temporal processing layers [default: 3]
#--track_net : Network type to process temporal information: lstm or conv [default: conv]
#--track_features     : Defines from which layer of 3d box estimation network to take features (global: 512+k, fc1: 512, fc2:256) [default: global]
  
TIME_PATH="/root_3d_log/time_series"
NORMAL_PATH="/root_3d_log/normal"
LOG_NAME="log_tracking_lstm_s13"

DATA_NAME="frustum_carpedcyc_tracking_val.pickle"

# Copy model_util with kitti mean sizes
cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py model_util.py

python ../train/test_tracks.py  --gpu 0 \
				--num_point 1024 \
				--model frustum_pointnets_v1 \
				--model_path $TIME_PATH/$LOG_NAME/model.ckpt \
				--output $TIME_PATH/$LOG_NAME/detection_results \
				--data_path ../kitti/$DATA_NAME \
				--no_intensity \
				--dump_result \
				--tracking \
				--tracks \
				--tau 3\
				--track_net lstm\
				--track_features fc1
				
				#--idx_path ../kitti/image_sets/val.txt 
				#--from_rgb_detection




