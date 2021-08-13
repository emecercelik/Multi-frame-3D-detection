#!/bin/bash
# tracking_path: should contain the predictions in the following folder format: <drive_id>/track_labels/<drive_id>.txt Ex: /root_2d_log/validation/pretrained_kitti/0011/track_labels/0011.txt
# drive_ids : Drive IDs to generate rgb_detection files from. 
# rgb_det_path: Folder where the rgb_detection.txt files for each drive will be generated. Ex: <rgb_det_path>/0011/rgb_detection.txt


#TRAINING_NAME='train_bmw_v3'
#python tracking2rgb_detections.py 	--tracking_path /root_2d_log/validation/$TRAINING_NAME/predictions_dist/0.9 \
#					--drive_ids 3 4 43 44 \
#					--rgb_det_path /root_2d_log/rgb_detections_from_predictions/$TRAINING_NAME/
					
					
#TRAINING_NAME='pretrained_kitti'
#python tracking2rgb_detections.py 	--tracking_path /root_2d_log/validation/$TRAINING_NAME/ \
#					--drive_ids 11 15 16 18 \
#					--rgb_det_path /root_2d_log/rgb_detections_from_predictions/$TRAINING_NAME/
					
# Inside the tracking_path, don't forget to copy track_labels as defined above.
# Class IDs 1: Pedestrian, 2: Car, 3: Cyclist in the rgb_detection.txt files
#TRAINING_NAME='gt_boxes'
#python tracking2rgb_detections.py 	--tracking_path /root_2d_log/validation/$TRAINING_NAME/ \
#					--drive_ids 11 15 16 18 \
#					--rgb_det_path /root_2d_log/rgb_detections_from_predictions/$TRAINING_NAME/
					
TRAINING_NAME='deep_sort'
python tracking2rgb_detections.py 	--tracking_path /root_2d_log/validation/$TRAINING_NAME/ \
					--drive_ids 11 15 16 18 \
					--rgb_det_path /root_2d_log/rgb_detections_from_predictions/$TRAINING_NAME/



