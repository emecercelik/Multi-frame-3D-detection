## 2D detection and tracking
This module is designed for 2D detection and tracking of objects in autonomous driving scenarios. The 2D detection network and training pipeline are based on [Tensorflow's Object Detection API (TF OD API)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1.md). The module is designed for [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). The general pipeline is listed below in order:
<ol>
<li>Generate tfrecords for training Faster RCNN (dataset/gen_kitti_tfrecords.sh)</li>
<li>Train Faster RCNN with tfrecords for 2D object detection task (train/train.sh)</li>
<li>Export the graph of trained network for inference (train/export_graph.sh)</li>
<li>Run inference to detect object in the test data(train/inference.sh)</li>
<li>Calculate recall to see the percentage of the ground-truth objects detected by the network(train/calculate_recall.sh)</li>
<li>Associate detections in the successive frames using IoU (train/associate.sh)</li>
<li>Evaluate tracking for the tracking metrics(evaluate/evaluate.sh)</li>
<li>[Detection and Tracking](#centertrack) with [CenterTrack](https://github.com/xingyizhou/CenterTrack/) </li>
</ol>

## Generate tfrecords
The Faster RCNN is constructed based on Tensorflow's Object Detection API 1 models. This model consumes data in tfrecords format. To generate tfrecords for KITTI tracking dataset, the following script can be run:

```bash
./frustum_framework/detection_2d/dataset/gen_kitti_tracking_tfrecords.sh
```

## Train Faster RCNN

To train Faster RCNN with KITTI tracking dataset, the following script can be run:

```bash
./frustum_framework/detection_2d/train/train.sh
```
The parameters to pass in training script is explained in the shell script. The config file passed to the training script defines the model, training parameters, and paths to the tfrecords and pre-trained weights. The config file used for training with KITTI dataset can be found in "/frustum_framework/detection_2d/config/faster_rcnn_resnet101_kitti.config". This is generated based on [Tensorflow's Object Detection API's config format](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). The parameters are also self-explanatory since they are directly related to Faster RCNN's architecture. For data augmentation methods and their usage, please refer to [preprocessor](https://github.com/tensorflow/models/blob/eab781187e1b3de41301f87434fca025f3c4bf10/research/object_detection/core/preprocessor.py) page of TF OD API. The config file also requires a label map in addition to the generated tfrecords files. The label map is used to map indices of classes to the names of classes as they defined with names in the ground-truth annotations. The label map is generated in Python dictionary format and can be adjusted or extended with new classes by changing the txt files. One example can be found in "/frustum_framework/detection_2d/dataset/kitti_label_map.pbtxt" file. 


Training pipeline is started with the pre-trained weights obtained by Coco object detection dataset as defined by "fine_tune_checkpoint" in the config file. Further information can be found in [TF OD API training documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_training_and_evaluation.md). This parameter can be also used to continue a previously stopped training. 

Tensorboard can be used to visualize the training process, loss graphs, and detections on sampled images. For this, the following command can be run with the log directory of the training process.

```bash
tensorboard --logdir /root_2d_log/log_folder/
```
Tensorboard publishes the outcomes of training process from port 6006. This can be reached from the localhost:<port_host>, where port_host defines the port attached to the 6006 port of Docker container. 

## <a name="graph"></a>Export the trained graph

The trained Faster RCNN model can be exported to be used for inference. To export the graph, the following script can be run for KITTI data.

```bash
./frustum_framework/detection_2d/train/export_graph.sh
```

The config file used for training, the checkpoint of the model to be exported, and an export directory should be passed to the python code as explained in the scripts. This process exports a *.pb file, which will be consumed in the Inference step. 

## <a name="inference"></a>Run Inference

Inference step utilizes a trained network (exported graph in [Graph](#graph)) to provide bounding box predictions on the given images. Different from training that uses tfrecords as inputs, inference directly predicts bounding boxes on images. The graph of the trained network, the label map, path to the image folder should be passed as explained in the following script to run inference used for KITTI. 

```bash
./frustum_framework/detection_2d/train/inference.sh
```

Inference step also generates txt files with predictions in KITTI format by filtering the detected objects according to the scores. It is also possible to get the predicted boxes drawn on the images as defined with the flags in the given scripts. 

## Calculate Recall

[Recall](https://en.wikipedia.org/wiki/Precision_and_recall) defines the percentage of the detected objects among the all detectable objects (ones annotated in ground-truth labels). This metric is particularly important for Frustum Pointnet (FP) architecture, since FP processes LiDAR points only from the regions of 2D bounding boxes. Therefore, it is important to cover almost all objects on the 2D image, even though the precision is low. 

To calculate the recall, the script below can be run for KITTI data:

```bash
./frustum_framework/detection_2d/train/calculate_recall.sh
```

These scripts calculate recalls for each class separately. The parameters are explained inside the scripts.

## Associate Detected Boxes in Successive Frames

In order to make use of recurrent layers, the track IDs should be predicted by a tracker. Therefore, the predicted bounding boxes in successive frames are required to be associated to each other correctly. Several methods have been proposed for this association step. In this project, four methods can be used interchangeably, which are IoU-based (intersection over union) and distance-based association as well as SORT and DEEP SORT. 

IoU-based association considers the overlapping areas of the 2D bounding boxes in frame t and frame t-1. The predicted bounding boxes of successive frames, which have an IoU (overlap) above a threshold, belong to the same class, and have the maximum IoU among all combinations, are associated. If a bounding box of frame t has 0.8 and 0.85 IoU values with two bounding boxes of frame t-1 and the IoU threshold is defined as 0.5, the association considers the IoU with 0.85 value and discards the other. 

Distance-based association considers the pixel-wise Euclidean distance between the centers of 2D bounding boxes from successive frames. Similar to the IoU-based association, only the association possibilities that have a calculated distance lower than a chosen maximum distance value are taken into account. The predicted objects from successive frames are matched based on the minimum distance. 

These two approaches are based on the assumption that the objects in successive frames do not move with large steps. To hold this assumption, the frames should be sampled with a relatively high frequency. KITTI dataset provides images sampled with 10 Hz. 

The association step utilizes the labels generated by the [Inference](#inference) step. The script below can be used to associate KITTI labels. 

```bash
./frustum_framework/detection_2d/train/associate.sh
```

The script can be configured to associate only 2D bounding boxes with a high confidence score defined by `scr_thr` in the script. It is also possible to draw track IDs and 2D bounding boxes on images. Explanations of flags are provided in the scripts.

The association generates a txt file with the 2D bounding boxes, track IDs, and frame indices in KITTI format. This file can be used while evaluating the tracking or to be consumed by the 3D detection network. Association script also generates a mapping file (evaluate_tracking.seqmap.val) that will be used by the evaluation script. This file shows the drive id with the starting and ending indices of frames of the drive. 

It is possible to track objects using SORT and DEEP SORT algorithms. For that the association metric arguement should be set as 'sort' or 'deep_sort', respectively. Please check the format of --path_to_predictions folder that should be given for tracking. 

## Evaluate Tracking

In this project, mainly three tracking metrics are considered, but many of them are calculated using the script given below for KITTI: 

```bash
./frustum_framework/detection_2d/evaluation/evaluate.sh
```

The three metrics considered in this project are "mostly tracked", "partly tracked", and "mostly lost". These are defined taking the life span of a ground-truth box into account. If a gt box is tracked 80% of its life span or more, it is considered as mostly tracked. If a gt box is tracked 20% of its life span or less, it is considered as mostly lost. The objects inbetween are counted as partly tracked. Since the object detection is already evaluated by the recall and average precision metrics, these three metrics provide intuitive idea about the tracking performance. 

The detailed explanation of other tracking metrics can be found in [3D Multi-Object Tracking: A Baseline and New Evaluation Metrics](http://www.xinshuoweng.com/papers/AB3DMOT/camera_ready.pdf) and in [MOTChallenge 2015:Towards a Benchmark for Multi-Target Tracking](https://arxiv.org/pdf/1504.01942.pdf). 

The flags and explanations are provided in the scripts for the evaluation. 

## <a name="centertrack"></a>Detection and Tracking with CenterTrack
CenterTrack detects the objects in a given sequence using its CenterNet detector and tracks using the features of it. The provided pre-trained networks for KITTI Tracking dataset are trained on a different split from used in this project. Therefore, it is necessary to train the network using the pretrained checkpoint on nuScenes. 

<ol>
<li>Installation</li>
The installation is explained in the [CenterTrack repository](https://github.com/xingyizhou/CenterTrack/blob/master/readme/INSTALL.md). As an alternative, it is possible to use the following docker image. 

```bash
docker pull emecercelik/centertrackv3:latest
```

The necessary folder mountings can be seen in the docker run script below:

```bash
./frustum_framework/detection_2d/centertrack_run.sh
```

After starting the container, please run the following script to copy necessary documents from the local version of the centernet to the container version of the centernet folder. 

```bash
./centertrack/models/run_at_start.sh
```

<li>Data preparation</li>
As a first step, the data should be prepared for the desired split. Data preparation is explained [here](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md#kitti-tracking) for KITTI Tracking dataset. By default, tthe splits generated with the `CenterTrack/src/toolsconvert_kittitrack_to_coco.py` file are train_half, val_half, train, test. Train and test takes all the available KITTI tracking drives and use them for training or testing, respectively. In train_half and val_half splits, one drive is split into two halves. One half is used for training and the other is for validation. It is possible to add new splits by extending the lines 11,12, and 13 of the `CenterTrack/src/toolsconvert_kittitrack_to_coco.py` files similar to the following example:

```python
SPLITS = ['train_half', 'val_half', 'train', 'test','train_temp','val_temp']
VIDEO_SETS = {'train': range(21), 'test': range(29), 
  'train_half': range(21), 'val_half': range(21), 
  'train_temp':[0,2,3,4,5,6,7,8,9,10,12,13,14,17,19,20],
  'val_temp':[11,15,16,18]} 
```

In this example, `train_temp` and `val_temp` are added as additional splits, which can be used for training and validation. In the `train_temp` split, the drives `0,2,3,4,5,6,7,8,9,10,12,13,14,17,19,20` are used for training with all frames inside them. In the val_temp split, all frames of drives `11,15,16,18` are used to generate that split.

The new validation split also requires a new `seqmap` for evaluation that indicates the drive names with the starting and ending frames. The format should be similar to the following in a seqmap file, whose name is extended by the split name: `evaluate_tracking<split_name>.seqmap`

```bash
0011 empty 000000 000372
0015 empty 000000 000375
0016 empty 000000 000208
0018 empty 000000 000338
```

The labels for evaluation should be gathered in a folder with the name `label_02_<split_name>`. The structure of the folder is the following:

```bash
/centertrack/src/tools/eval_kitti_track/data/tracking/label_02_val_temp
----0011.txt
----0015.txt
----0016.txt
----0018.txt
```

<li>Training</li>

After preparing the data, the training can be done with the following command (in `/centertrack/src` if running in the Docker container):

```bash
python main.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version train_temp --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1 --gpus 0,1 --batch_size 8 --load_model ../models/nuScenes_3Ddetection_e140.pth --master_batch_size 2 --lr 0.625e-4
```

The `--dataset_version` indicates the data split that will be used for training. `--master_batch_size` determines how many of the batches will be run on the master GPU. The options for training can be seen in `CenterTrack/src/lib/opts.py`. They generate the KITTI tracking results in the [paper](http://arxiv.org/abs/2004.01177) by starting the KITTI tracking training with a checkpoint pretrained on nuScenes. 

<li>Testing</li>
The following command (run in `/centertrack/src` if running in the Docker container) tests the trained network on the given dataset split and writes the results in ` /centertrack/exp/tracking/<exp_id>`. The `exp_id` should be same as the training `exp_id` to be used. The test script takes the latest checkpoint by default unless it is indicated explicitly. 

```bash
python test.py tracking --exp_id kitti_temp --dataset kitti_tracking --dataset_version val_temp --pre_hm --track_thresh 0.4 --resume
```

Adding `--debug 4` option generates the visualizations of the tracking results on images as well as the heatmaps. The options can be viewed in `CenterTrack/src/lib/opts.py`.

If the CenterTrack is utilized in the container, the results should be taken to a host folder not to lose after stopping the container. An example command to do that:

```bash
cp -r /centertrack/data/kitti_tracking/kitti_temp /centertrack/exp/tracking/
```





