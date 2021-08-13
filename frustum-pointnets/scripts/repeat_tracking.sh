#/bin/bash

## Train on KITTI tracking data

# paths to save data and logs

TIME_PATH=/root_3d_log/kitti_repeats


# Copy model_util with kitti mean sizes
cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py /frustum_framework/frustum-pointnets/models/model_util.py

# Please see the README in this folder for the meanings of flags
# Train with  the provided parameters
MAIN_PATH=/root_3d_log/kitti_attention_max_pool
MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s0"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti


MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s1"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s2"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################
MAIN_PATH=/root_3d_log/kitti_attention_max_pool
MAIN_LOG="log_time_s6"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s3"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s6"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s4"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s6"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s5"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################
MAIN_PATH=/root_3d_log/kitti_attention_max_pool
MAIN_LOG="log_time_s7"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s6"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s7"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s7"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s7"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s8"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################
MAIN_PATH=/root_3d_log/kitti_attention_time
MAIN_LOG="log_time_s22"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s9"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s22"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s10"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s22"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s11"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################

MAIN_PATH=/root_3d_log/kitti_attention_whl
MAIN_LOG="log_time_s24"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s12"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s24"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s13"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s24"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s14"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################

MAIN_PATH=/root_3d_log/kitti_attention_whl
MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s15"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s16"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s3"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s17"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################

MAIN_PATH=/root_3d_log/kitti_attention_whl
MAIN_LOG="log_time_s4"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s18"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s4"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s19"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

MAIN_LOG="log_time_s4"
cp $MAIN_PATH/$MAIN_LOG/frustum_pointnets_v1.py /frustum_framework/frustum-pointnets/models/frustum_pointnets_v1.py
cp $MAIN_PATH/$MAIN_LOG/lstm_helper.py /frustum_framework/frustum-pointnets/train/lstm_helper.py

LOG_NAME="log_time_s20"
LOG_PATH=$TIME_PATH/$LOG_NAME
python repeat_train.py 	--log_dir_old $MAIN_PATH/$MAIN_LOG --log_dir_new $LOG_PATH

python run_all_logs.py --log_dir $LOG_PATH --gt_root_dir /kitti_root_tracking/drives_in_kitti

##########################################################################################################

