#!/bin/bash

# Port bindings: -p host_port:container_port
# Port bindings: -p host_port:container_port
read -p "Tensorboard port (host) "  TboardHostPort
read -p "Jupyter Notebook port (host) "  JupyHostPort

# Path where the KITTI object detection dataset tfrecords files are kept or to be saved
obj_det_tfrecords="/media/HDD/object_detection_work/Datasets/KITTI/frustum_splits"
# Path where the KITTI tracking dataset tfrecords files are kept or to be saved
tracking_tfrecords="/media/HDD/tracking_dataset/KITTI/tracking_tfrecords"
# Path where the KITTI object detection dataset for 2D detection is kept
kitti_root_2d="/media/HDD/object_detection_work/Datasets/KITTI/object_detection_2012/"
# Path where the KITTI object detection dataset for 3D detection is kept
kitti_root_3d="/media/HDD/object_detection_work/Datasets/KITTI/"
# Path where the KITTI tracking dataset for 2D and 3D detection is kept
kitti_root_tracking="/media/HDD/tracking_dataset/centertrack_kitti_tracking/"
# Path where the 3d detection log files are generated
root_3d_log="/media/HDD2/trainings/frustum_pointnet"
# Path where the 3d detection log files are generated
root_2d_log="/media/HDD2/trainings/tf_frcnn"

# Path to the codes
frustum_framework="/home/emec/Desktop/projects/frustum_framework"
centertrack="/home/emec/Desktop/projects/frustum_framework/detection_2d/CenterTrack/models"
centertrack_local="/home/emec/Desktop/projects/frustum_framework/detection_2d/CenterTrack"
docker run -it --gpus '"device=0,1"' --rm --shm-size 8G \
				-v $obj_det_tfrecords:/obj_det_tfrecords \
				-v $tracking_tfrecords:/tracking_tfrecords \
				-v $kitti_root_2d:/kitti_root_2d \
				-v $kitti_root_3d:/kitti_root_3d \
				-v $kitti_root_tracking:/centertrack/data/kitti_tracking \
				-v $root_3d_log:/root_3d_log \
				-v $root_2d_log:/root_2d_log \
				-v $frustum_framework:/frustum_framework \
				-v $centertrack:/centertrack/models \
				-v $centertrack_local:/centertrack_local \
				-p $JupyHostPort:8888 \
				-p $TboardHostPort:6006 \
				emecercelik/centertrackv3
