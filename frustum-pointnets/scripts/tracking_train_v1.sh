#/bin/bash

## Train on KITTI tracking data

# paths to save data and logs
TIME_PATH="/root_3d_log/time_series"
NORMAL_PATH="/root_3d_log/normal"
LOG_NAME="log_tracking_lstm_s44_baseline"
LOG_PATH=$TIME_PATH/$LOG_NAME

# Check if the log directory exists to avoid overwriting
if [ -d $LOG_PATH ] 
then
    echo "Directory $LOG_PATH exists."
    read -p "Do you wish to continue? " yn
    case $yn in
        [Yy]* ) ;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no."; exit;;
    esac
else
    echo "Directory $LOG_PATH does not exist."
fi

# Copy model_util with kitti mean sizes
cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py model_util.py

# Please see the README in this folder for the meanings of flags
# Train with  the provided parameters
python ../train/train.py 	--gpu 1 \
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
				--tau 3 \
				#--cos_loss \
				--cos_loss_weight 1.0 \
				--cos_loss_prop 2\
				--cos_loss_batch_thr 0\
#--restore_model_path ../train/log_kitti_v1/model.ckpt

