#/bin/bash

IMG_PATH=/media/HDD/tracking_dataset/KITTI/data_tracking_image_2/training/image_02/0011
LAB_PATH=/media/HDD2/trainings/frustum_pointnet/time_series/log_tracking_lstm_s5/detection_results/data/0011/data/
IMG_ID=10
FIG_NAME=$IMG_ID+_2d.png
CALIB_PATH=/media/HDD/tracking_dataset/KITTI/data_tracking_calib/training/calib/
GT_PATH=/media/HDD/tracking_dataset/KITTI/drives_in_kitti/0011

python viz_util_2d.py --label_path $LAB_PATH --image_path $IMG_PATH --calib_path $CALIB_PATH --gt_label_path $GT_PATH --image_ind $IMG_ID --fig_name $FIG_NAME



#IMG_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/object_detection_2012/data_object_image_2/training/image_2/
#LAB_PATH=/home/emec/Desktop/projects/frustum-pointnets/train/detection_results_v1/data/
#FIG_NAME=2d.png
#IMG_ID=1
#CALIB_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/object_detection_2012/data_object_calib/training/calib/
#GT_PATH=/media/emec/15dd644e-cbac-4ee7-820d-b0b0ecb0e813/object_detection_work/Datasets/KITTI/object_detection_2012/data_object_label_2/training/label_2/

#python viz_util_2d.py --label_path $LAB_PATH --image_path $IMG_PATH --calib_path $CALIB_PATH --gt_label_path $GT_PATH --image_ind $IMG_ID --fig_name $FIG_NAME

