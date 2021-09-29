#!/bin/bash
#--data_dir		: The full path to the unzipped folder containing the unzipped data from data_object_image_2 and data_object_label_2.zip for KITTI tracking. For any dataset, folder structure is assumed to be: data_dir/data_object_label_2/training/label_2/<drive_id>.txt (annotations) and data_dir/data_object_image_2/training/image_2/<drive_id>/ (images).
#--output_path		: The path to which TFRecord files will be written. The TFRecord with the training set will be located at: <output_path>_train.tfrecord. And the TFRecord with the validation set will be located at: <output_path>_val.tfrecord
#--classes_to_use	: List of strings naming the classes for which data should be converted. Use the same names as presented in the KIITI README file. Adding dontcare class will remove all other bounding boxes that overlap with areas marked as dontcare regions.
#--label_map_path	: Path to label map proto (*.pbtxt file)
#--is_training		: If set, the training data will be generated (<output_path>_train.tfrecord). Otherwise, validation data will be generated (<output_path>_val.tfrecord).
#--drive_ids		: The drive IDs that will be used while generating tfrecords.

object_detection="/usr/local/lib/python3.6/dist-packages/tensorflow_core/models/research/object_detection"

python $object_detection/dataset_tools/create_kitti_tf_record.py \
	--data_dir /kitti_root_2d \
	--output_path /root_2d_log/kitti_tfapi \
	--classes_to_use 'car,pedestrian,cyclist,dontcare' \
	--label_map_path /frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt
