#!/bin/bash

python modelEval_sum.py		--main_path /root_3d_log/rnn_fc1_tau4 \
				--output_name time-car\
				--prefix log_time_s \
				--log_indices 0 1 2 3 4 5 6 7 8 \
				--params_to_group tau track_net learning_rate only_whl \
				--detection_type car_detection_3d \
				--eval_drives 98

python modelEval_sum.py		--main_path /root_3d_log/rnn_fc1_tau4 \
				--output_name time-ped\
				--prefix log_time_s \
				--log_indices 0 1 2 3 4 5 6 7 8 \
				--params_to_group tau track_net learning_rate only_whl\
				--detection_type pedestrian_detection_3d\
				--eval_drives 98

python modelEval_sum.py		--main_path /root_3d_log/rnn_fc1_tau4 \
				--output_name time-cyc\
				--prefix log_time_s \
				--log_indices 0 1 2 3 4 5 6 7 8  \
				--params_to_group tau track_net learning_rate only_whl\
				--detection_type cyclist_detection_3d\
				--eval_drives 98

