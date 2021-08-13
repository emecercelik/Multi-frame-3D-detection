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
kitti_root_tracking="/media/HDD/tracking_dataset/KITTI/"
# Path where the 3d detection log files are generated
root_3d_log="/media/HDD2/trainings/frustum_pointnet"
# Path where the 3d detection log files are generated
root_2d_log="/media/HDD2/trainings/tf_frcnn"
# Path to bmw data
root_bmw_tracking="/media/HDD/datasets/bmw_data/kitti_format_tracking"
# Path to the folder where bmw tfrecord files are kept
root_bmw_tfrecord="/media/HDD/datasets/bmw_data"
# Path to the codes
frustum_framework="/home/emec/Desktop/projects/frustum_framework"
docker run -it --gpus all --rm  -v $obj_det_tfrecords:/obj_det_tfrecords \
				-v $tracking_tfrecords:/tracking_tfrecords \
				-v $kitti_root_2d:/kitti_root_2d \
				-v $kitti_root_3d:/kitti_root_3d \
				-v $kitti_root_tracking:/kitti_root_tracking \
				-v $root_3d_log:/root_3d_log \
				-v $root_2d_log:/root_2d_log \
				-v $frustum_framework:/frustum_framework \
				-v $root_bmw_tracking:/root_bmw_tracking \
				-v $root_bmw_tfrecord:/root_bmw_tfrecord \
				-p $JupyHostPort:8888 \
				-p $TboardHostPort:6006 \
				emecercelik/tum-i06-object_detection:faster_frustum_nonroot_v3 
				#-u 0
