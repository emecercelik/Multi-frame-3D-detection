#/bin/bash

## Train on KITTI tracking data

# paths to save data and logs
TIME_PATH="/root_3d_log/time_series"
NORMAL_PATH="/root_3d_log/normal"
LOG_NAME="log_tracking_lstm_s16"
LOG_PATH=$TIME_PATH/$LOG_NAME

# Copy model_util with kitti mean sizes
cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py model_util.py

# Please see the README in this folder for the meanings of flags
# Train with  the provided parameters
python ../train/train.py 	--gpu 0 \
				--tracking \
				--model frustum_pointnets_v1 \
				--log_dir $LOG_PATH \
				--num_point 1024 \
				--max_epoch 101 \
				--batch_size 32 \
				--decay_step 800000 \
				--decay_rate 0.5 \
				--no_intensity \
				--tracks \
				--track_net lstm\
				--track_features fc1 \
				--tau 3
				--cos_loss \
				--cos_loss_weight 1.0 \
				--cos_loss_prop 2\
				--cos_loss_batch_thr 0\

