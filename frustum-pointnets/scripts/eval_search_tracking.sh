#/bin/bash



start=0
N=19
for i in $(seq 0 $N)
do
LOG_ID=$(($start+$i))

python run_all_logs.py --log_dir /root_3d_log/rnn_fc1_tau4/log_time_s${LOG_ID} --gt_root_dir /kitti_root_tracking/drives_in_kitti --multi_model  --parallel --eval_drives 98 --skip_test
done


