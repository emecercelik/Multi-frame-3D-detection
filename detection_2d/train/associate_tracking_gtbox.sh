#!/bin/bash

#--path_to_predictions	: Path to the text files that the predictions written in. The names of txt files determines the order. The names are assumed to be in KITTI format.
#--path_to_labels	: Path to the label map of the trained network.
#--iou_thr		: IoU threshold to assign objects in successive frames. 
#--scr_thr		: A list of score thresholds for each class in the label map. Only the predictions above these thresholds will be associated. The thresholds of classes will be called using the indices in label maps.
#--path_to_images	: Path to the folder that contains images on which track ids will be drawn.
#--output_images	: Path showing where the images will be saved after the tracks are written on those.
#--output_path		: Path to the txt file to write the associated object predictions with track IDs in the KITTI tracking dataset format. Pay attention to with_score tag.
#--with_score		: To save predictions into txt files with the prediction score in KITTI tracking format.
#--image_format		: Format of the images that will be tested. By default they are considered as png. It can be jpg, jpeg, etc. 
#--save_images		: To save images with the predicted boxes drawn. Images are saved in the images folder of output_path
#--association_metric	: iou or dist. If iou, the association between boxes_proposal and boxes_gt will be done using the intersection over union values between them. If dist, euclidean distance between the centers of the boxes will be used instead. The value used for dist is not directly euclidean distance, but 1-normalized_euc_dist. 
THR='0.3' # IoU threshold that will be used for assigning 
METRIC='sort'
PRED_FOLDER_NAME='gt_boxes' 
DRIVE_ID=0018
python associate.py 	--path_to_predictions /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 0.45\
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--output_images /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_images \
			--output_path /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_labels/$DRIVE_ID.txt \
			--with_score \
			--association_metric $METRIC


DRIVE_ID=0015
python associate.py 	--path_to_predictions /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 0.45\
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--output_images /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_images \
			--output_path /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_labels/$DRIVE_ID.txt \
			--with_score \
			--association_metric $METRIC


DRIVE_ID=0016
python associate.py 	--path_to_predictions /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 0.45\
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--output_images /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_images \
			--output_path /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_labels/$DRIVE_ID.txt \
			--with_score \
			--association_metric $METRIC


DRIVE_ID=0011
python associate.py 	--path_to_predictions /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 0.45\
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_ID/ \
			--output_images /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_images \
			--output_path /root_2d_log/validation/$PRED_FOLDER_NAME/$DRIVE_ID/predictions_$METRIC/$THR/track_labels/$DRIVE_ID.txt \
			--with_score \
			--association_metric $METRIC
			
			

