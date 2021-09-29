#!/bin/bash

#--data_dir		: The full path to the unzipped folder containing the unzipped data from data_object_image_2 and data_object_label_2.zip for KITTI tracking. For any dataset, folder structure is assumed to be: data_dir/data_object_label_2/training/label_2/<drive_id>.txt (annotations) and data_dir/data_object_image_2/training/image_2/<drive_id>/ (images).
#--output_path		: The path to which TFRecord files will be written. The TFRecord with the training set will be located at: <output_path>_train.tfrecord. And the TFRecord with the validation set will be located at: <output_path>_val.tfrecord
#--classes_to_use	: List of strings naming the classes for which data should be converted. Use the same names as presented in the KIITI README file. Adding dontcare class will remove all other bounding boxes that overlap with areas marked as dontcare regions.
#--label_map_path	: Path to label map proto (*.pbtxt file)
#--is_training		: If set, the training data will be generated (<output_path>_train.tfrecord). Otherwise, validation data will be generated (<output_path>_val.tfrecord).
#--drive_ids		: The drive IDs that will be used while generating tfrecords.
#--min_area		: if True, boxes that have smaller area than a square, whose edge is defined with min_area_one_edge, will be discarded.
#--min_area_one_edge    : if min_area flag is True, this defines one edge of a sqaure box, boxes smaller than whose area will be discarded.

python create_kitti_tracking_tf_record.py \
	--data_dir /kitti_root_tracking \
	--output_path /root_2d_log/kitti_tfapi_tracking_7cl \
	--classes_to_use 'car ,van ,truck ,pedestrian ,cyclist ,tram ,dontcare' \
	--label_map_path /frustum_framework/detection_2d/dataset/kitti_label_map_7.pbtxt \
	--is_training \
	--drive_ids "0 1 2 3 4 5 6 7 8 9 10 12 13 14 17 19 20" \
	--statistics_path /root_2d_log/kitti_statistics2/

python create_kitti_tracking_tf_record.py \
	--data_dir /kitti_root_tracking \
	--output_path /root_2d_log/kitti_tfapi_tracking_7cl \
	--classes_to_use 'car,pedestrian,cyclist' \
	--label_map_path /frustum_framework/detection_2d/dataset/kitti_label_map_7.pbtxt \
	--drive_ids "11 15 16 18" \
	--statistics_path /root_2d_log/kitti_statistics3/
