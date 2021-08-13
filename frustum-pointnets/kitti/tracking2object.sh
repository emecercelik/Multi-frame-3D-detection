#!/bin/bash

# to convert KITTI tracking dataset format into KITTI object detection format
python tracking2object.py 	--tracking_path /kitti_root_tracking/ \
				--drive_ids 0 2 3 4 5 6 7 8 9 10 12 13 14 17 19 20 \
				--output_path /kitti_root_tracking/drives_in_kitti 
				
				
				
#python tracking2object.py 	--tracking_path /kitti_root_tracking/ \
#				--drive_ids 30 31 32 33 \
#				--output_path /kitti_root_tracking/drives_in_kitti 

