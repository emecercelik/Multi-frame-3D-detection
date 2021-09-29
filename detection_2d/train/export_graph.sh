#!/bin/bash

### Please adjust here
# config that is used to train the network that will be exported
PIPELINE_CONFIG_PATH="/root_2d_log/train_tracking_3cl_v2/faster_rcnn_resnet101_kitti_tracking.config"
# Path to the model checkpoint
TRAINED_CKPT_PREFIX="/root_2d_log/train_tracking_3cl_v2/model.ckpt-900000"
# Where the frozen graph will be exported
EXPORT_DIR='/root_2d_log/train_tracking_3cl_v2/export2/'

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
