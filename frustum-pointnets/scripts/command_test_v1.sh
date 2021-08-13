#/bin/bash
#--gpu       : GPU to use [default: GPU 0]
#--num_point : Point Number [default: 1024]
#--model     : Model name [default: frustum_pointnets_v1]
#--model_path: model checkpoint file path [default: log/model.ckpt]
#--batch_size: batch size for inference [default: 32]
#--output    : output file/folder name [default: test_results]
#--data_path : frustum dataset pickle filepath [default: None]
#--from_rgb_detection : To run test on pickle files generated using rgb detection.
#--idx_path  : filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]
#--dump_result        : If true, also dump results to .pickle file
#--no_intensity       : Only use XYZ for training

# Copy model_util with kitti mean sizes
cp /frustum_framework/frustum-pointnets/models/model_util_kitti.py model_util.py

python train/test.py 	 --gpu 0\
			 --num_point 1024\
			 --model frustum_pointnets_v1\
			 --model_path ../train/log_v1/model.ckpt\
			 --output ../train/detection_results_v1\
			 --data_path ../kitti/frustum_carpedcyc_val_rgb_detection.pickle \
			 --from_rgb_detection \
			 --idx_path ../kitti/image_sets/val.txt\
			 --from_rgb_detection

#train/kitti_eval/evaluate_object_3d_offline dataset/KITTI/object/training/label_2/ train/detection_results_v1
