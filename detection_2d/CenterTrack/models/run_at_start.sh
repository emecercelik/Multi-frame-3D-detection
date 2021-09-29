#!/bin/bash

cp /centertrack_local/src/lib/logger.py /centertrack/src/lib/logger.py
cp /centertrack_local/src/tools/convert_kittitrack_to_coco.py /centertrack/src/tools/convert_kittitrack_to_coco.py

cp /centertrack_local/experiments/kitti_track_temp.sh /centertrack/src/

cp -r /centertrack_local/src/tools/eval_kitti_track/data/tracking/label_02_val_temp /centertrack/src/tools/eval_kitti_track/data/tracking/label_02_val_temp

cp /centertrack_local/src/tools/eval_kitti_track/data/tracking/evaluate_trackingval_temp.seqmap /centertrack/src/tools/eval_kitti_track/data/tracking/

cp /centertrack_local/src/tools/eval_kitti_track/evaluate_tracking.py /centertrack/src/tools/eval_kitti_track/
