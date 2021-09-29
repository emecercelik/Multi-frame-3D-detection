# train
python main.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version train_temp --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 16 --load_model ../models/nuScenes_3Ddetection_e140.pth
# test
python test.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version val_temp --pre_hm --track_thresh 0.4 --resume


# train
python main.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version train_temp --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 8 --load_model ../models/nuScenes_3Ddetection_e140.pth --master_batch_size 2 --lr 0.625e-4
# test
python test.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version val_temp --pre_hm --track_thresh 0.4 --resume

cp -r  /centertrack/exp/tracking/kitti_temp /centertrack/data/kitti_tracking/

# To visualize outcomes
python test.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version val_temp --pre_hm --track_thresh 0.4 --resume --debug 4

# Cp previously trained folder to the centertrack main folder for test
cp -r /centertrack/data/kitti_tracking/kitti_temp /centertrack/exp/tracking/
