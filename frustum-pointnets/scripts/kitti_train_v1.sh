#/bin/bash

# Train on KITTI object detection

TIME_PATH="/root_3d_log/time_series"
NORMAL_PATH="/root_3d_log/normal"
LOG_NAME="log_kitti_s1"
LOG_PATH=$NORMAL_PATH/$LOG_NAME

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

# Default pickle file directory : 'kitti/frustum_carpedcyc_train.pickle', 'kitti/frustum_carpedcyc_val.pickle'
# Track features cannot be used since this dataset doesn't contain any temporal data
# Please see the README in this folder for the meanings of flags
# Train with  the provided parameters
python ../train/train.py 	--gpu 0 \
				--model frustum_pointnets_v1 \
				--log_dir $LOG_PATH \
				--num_point 1024 \
				--max_epoch 51 \
				--batch_size 32 \
				--decay_step 800000 \
				--decay_rate 0.5 \
				--no_intensity

#--restore_model_path ../train/log_kitti_v1/model.ckpt

