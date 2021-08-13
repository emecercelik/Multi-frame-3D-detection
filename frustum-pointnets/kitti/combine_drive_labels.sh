#!/bin/bash

# To combine given drive labels in one drive to be able to evaluate all of the drive predictions at the same time

python combine_drive_labels.py 	--drives 11 15 16 18 \
				--output_drive 98 \
				--root_dir /kitti_root_tracking/drives_in_kitti



