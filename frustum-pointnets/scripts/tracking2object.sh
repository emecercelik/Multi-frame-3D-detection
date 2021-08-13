#!/bin/bash

## Arguments
# --tracking_path : Path to the KITTI tracking dataset. Inside: data_tracking_image_2, data_tracking_calib, data_tracking_label_2, data_tracking_velodyne folders
# --drive_ids     : Drive ids whose labels will be converted. [0,1,2,3]
# --output_path   : Path showing where the generated labels will be saved. This defines the root path and each drive will be separately saved under a folder with the drive name: 0000/000000.txt,000001.txt,...; 0011/000000.txt,000001.txt,... If None, the same folders are generated under <tracking_path>/drives_in_kitti.

python ../kitti/tracking2object.py \
	--tracking_path /kitti_root_tracking \
	--drive_ids 11 15 16 18 \
	--output_path /kitti_root_tracking/drives_in_kitti2
    


