#!/bin/bash

# Port bindings: -p host_port:container_port
read -p "Tensorboard port (host) "  TboardHostPort
read -p "Jupyter Notebook port (host) "  JupyHostPort

codes="/home/emec/Desktop/projects/arnds_work/"
# Path where the KITTI object detection dataset for 2D detection is kept
kitti_root_2d="/media/HDD/object_detection_work/Datasets/KITTI/object_detection_2012/"
# Path where the KITTI tracking dataset for 2D and 3D detection is kept
kitti_root_tracking="/media/HDD/tracking_dataset/KITTI"
# Path where the 3d detection log files are generated
root_log="/media/HDD2/trainings/tf_frcnn"
frustum_framework="/home/emec/Desktop/projects/frustum_framework"
docker run -it --gpus all --rm  -v $codes:/codes \
				-v $kitti_root_2d:/kitti_root_2d \
				-v $kitti_root_tracking:/kitti_root_tracking \
				-v $root_log:/root_log \
				-v $frustum_framework:/frustum_framework \
				-p $JupyHostPort:8888 \
				-p $TboardHostPort:6006 \
				emecercelik/tum-i06-object_detection:faster_frustum_nonroot 
				#-u 0 \

