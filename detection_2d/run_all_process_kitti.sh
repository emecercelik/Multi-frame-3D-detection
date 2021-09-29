#!/bin/bash
#--frozen_graph_path : Path to the exported frozen graph of the network that will be used for inference.
#--path_to_labels    : Path to the label map of the trained network.
#--path_to_images    : Path to the images that will be tested. All images in this directory will be tested.
#--image_format      : Format of the images that will be tested. By default they are considered as png. It can be jpg, jpeg, etc. 
#--output_path       : Path that the predictions will be recorded in KITTI format. Names of the text files will be indices that the images have.
#--scr_thr           : A list of score thresholds for each class in the label map. The thresholds of classes will be called using the indices in label maps. Only the predictions above this threshold will be written in the predictions. [0.0,0.8,0.4] for KITTI, 0.0 is dummy since indices start with 1 in tensorflow formatting.
#--with_score        : To save predictions into txt files with the prediction score in KITTI format.
#--save_images       : To save images with the predicted boxes drawn. Images are saved in the images folder of output_path

#11, 15, 16, and 18
LOG_DIR="train_v6"
DRIVE_NAME="0018"
CKPT_ID="1650000"
### Please adjust here
# config that is used to train the network that will be exported
PIPELINE_CONFIG_PATH="/root_2d_log/$LOG_DIR/faster_rcnn_resnet101_kitti.config"
# Path to the model checkpoint
TRAINED_CKPT_PREFIX="/root_2d_log/$LOG_DIR/model.ckpt-$CKPT_ID"
# Where the frozen graph will be exported
EXPORT_DIR="/root_2d_log/$LOG_DIR/export/"

### No need to change
# Path to the object detection api of tensorflow
object_detection="/usr/local/lib/python3.6/dist-packages/tensorflow_core/models/research/object_detection"
# Input type of tensorflow
INPUT_TYPE="image_tensor"
python $object_detection/export_inference_graph.py \
    --input_type=$INPUT_TYPE \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --trained_checkpoint_prefix=$TRAINED_CKPT_PREFIX \
    --output_directory=$EXPORT_DIR


cd train
python inference.py 	--frozen_graph_path /root_2d_log/$LOG_DIR/export/frozen_inference_graph.pb \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_NAME/ \
			--image_format png \
			--output_path /root_2d_log/validation/$LOG_DIR/predictions/$DRIVE_NAME \
			--scr_thr 0.0 0.5 0.3 \
			--with_score \
			--save_images 
printf "\n\n"
printf "*******************\n"
printf "$LOG_DIR $DRIVE_NAME Recall\n"
printf "*******************\n"
printf "\n\n"
python calculate_recall.py 	--path_to_predictions /root_2d_log/validation/$LOG_DIR/predictions/$DRIVE_NAME \
				--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
				--iou_thr 0.0 0.5 0.5 \
				--scr_thr 0.0 0.1 0.1 \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02/$DRIVE_NAME.txt 

printf "\n\n"
printf "*******************\n"
printf "$LOG_DIR $DRIVE_NAME Associate - dist\n"
printf "*******************\n"
printf "\n\n"

THR='0.9'
METRIC='dist'
python associate.py 	--path_to_predictions /root_2d_log/validation/$LOG_DIR/predictions/$DRIVE_NAME \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_NAME/ \
			--output_images /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_images \
			--output_path /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_labels/$DRIVE_NAME.txt \
			--with_score \
			--save_images \
			--association_metric $METRIC

printf "\n\n"
printf "*******************\n"
printf "$LOG_DIR $DRIVE_NAME Evaluate - dist\n"
printf "*******************\n"
printf "\n\n"

cd ../evaluation/
python evaluate_kitti3dmot.py 	--path_to_predictions /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_labels \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02 \
				--two_d 
				#--three_d

printf "\n\n"
printf "*******************\n"
printf "$LOG_DIR $DRIVE_NAME Associate - iou \n"
printf "*******************\n"
printf "\n\n"

cd ../train
THR='0.55'
METRIC='iou'
python associate.py 	--path_to_predictions /root_2d_log/validation/$LOG_DIR/predictions/$DRIVE_NAME \
			--path_to_labels /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt \
			--iou_thr $THR \
			--scr_thr 0.0 0.98 0.45 \
			--path_to_images /kitti_root_tracking/data_tracking_image_2/training/image_02/$DRIVE_NAME/ \
			--output_images /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_images \
			--output_path /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_labels/$DRIVE_NAME.txt \
			--with_score \
			--save_images \
			--association_metric $METRIC

printf "\n\n"
printf "*******************\n"
printf "$LOG_DIR $DRIVE_NAME Evaluate iou\n"
printf "*******************\n"
printf "\n\n"

cd ../evaluation/
python evaluate_kitti3dmot.py 	--path_to_predictions /root_2d_log/validation/$LOG_DIR/predictions_$METRIC/$THR/$DRIVE_NAME/track_labels \
				--path_to_gts /kitti_root_tracking/data_tracking_label_2/training/label_02 \
				--two_d 
				#--three_d
