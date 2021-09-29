#!/bin/bash

# Config file to set up Faster RCNN
PIPELINE_CONFIG_PATH="/frustum_framework/detection_2d/config/faster_rcnn_resnet101_kitti.config"
## The output directory that all the logs of training will be written in 
MODEL_DIR="/root_2d_log/train_v9/"
## Number of steps that the training takes place
NUM_TRAIN_STEPS=700000
## During validation to decrease number of samples used for validation
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

# Check if the log directory exists to avoid overwriting
if [ -d $MODEL_DIR ] 
then
    echo "Directory $MODEL_DIR exists."
    read -p "Do you wish to continue? " yn
    case $yn in
        [Yy]* ) ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no."; exit;;
    esac
else
    echo "Directory $MODEL_DIR does not exist."
    mkdir $MODEL_DIR
fi

cp $PIPELINE_CONFIG_PATH $MODEL_DIR
python model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr
