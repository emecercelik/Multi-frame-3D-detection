#!/bin/bash
#--frozen_graph_path : Path to the exported frozen graph of the network that will be used for inference.
#--path_to_labels    : Path to the label map of the trained network.
#--path_to_images    : Path to the images that will be tested. All images in this directory will be tested.
#--image_format      : Format of the images that will be tested. By default they are considered as png. It can be jpg, jpeg, etc. 
#--output_path       : Path that the predictions will be recorded in KITTI format. Names of the text files will be indices that the images have.
#--scr_thr           : A list of score thresholds for each class in the label map. The thresholds of classes will be called using the indices in label maps. Only the predictions above this threshold will be written in the predictions. [0.0,0.8,0.4] for KITTI, 0.0 is dummy since indices start with 1 in tensorflow formatting.
#--with_score        : To save predictions into txt files with the prediction score in KITTI format.
#--save_images       : To save images with the predicted boxes drawn. Images are saved in the images folder of output_path

DRIVE_ID=0011
python inference.py 	--frozen_graph_path /root_2d_log/train_tracking_3cl_v2/export2/frozen_inference_graph.pb \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--image_format png \
			--output_path /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions \
			--scr_thr 0.0 0.8 0.4 0.4\
			--with_score \
			--save_images 

DRIVE_ID=0015
python inference.py 	--frozen_graph_path /root_2d_log/train_tracking_3cl_v2/export2/frozen_inference_graph.pb \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--image_format png \
			--output_path /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions \
			--scr_thr 0.0 0.8 0.4 0.4\
			--with_score \
			--save_images 


DRIVE_ID=0016
python inference.py 	--frozen_graph_path /root_2d_log/train_tracking_3cl_v2/export2/frozen_inference_graph.pb \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--image_format png \
			--output_path /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions \
			--scr_thr 0.0 0.8 0.4 0.4\
			--with_score \
			--save_images 


DRIVE_ID=0018
python inference.py 	--frozen_graph_path /root_2d_log/train_tracking_3cl_v2/export2/frozen_inference_graph.pb \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map_3.pbtxt \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--image_format png \
			--output_path /root_2d_log/validation/train_tracking_3cl_v2/$DRIVE_ID/predictions \
			--scr_thr 0.0 0.8 0.4 0.4\
			--with_score \
			--save_images 

