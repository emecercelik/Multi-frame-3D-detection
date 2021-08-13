#!/bin/bash

FOLDER_NAME1="log_tracking_v2_s4"

PATH1="/media/HDD2/trainings/frustum_pointnet/time_series"
PATH2="/media/HDD2/trainings/frustum_pointnet/normal"

tensorboard --logdir conv_cos:$PATH1/$FOLDER_NAME1  --port 6006

