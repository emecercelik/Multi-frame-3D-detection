#/bin/bash

VEL_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/data_object_velodyne/training/velodyne/
LAB_PATH=/home/emec/Desktop/projects/frustum-pointnets/train/detection_results_v1/data/
IMG_PATH=vel.jpg
LIDAR_ID=19
CALIB_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/object_detection_2012/data_object_calib/training/calib/
GT_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/object_detection_2012/data_object_label_2/training/label_2/

python viz_util_3d.py --label_path $LAB_PATH --lidar_path $VEL_PATH --calib_path $CALIB_PATH --gt_label_path $GT_PATH --lidar_ind $LIDAR_ID --fig_name $IMG_PATH

#python viz_util_emec.py --lidar_path $VEL_PATH --calib_path $CALIB_PATH --gt_label_path $GT_PATH --lidar_ind $LIDAR_ID --fig_name $IMG_PATH

#python viz_util_emec.py --label_path $LAB_PATH --lidar_path $VEL_PATH --calib_path $CALIB_PATH --lidar_ind $LIDAR_ID --fig_name $IMG_PATH
