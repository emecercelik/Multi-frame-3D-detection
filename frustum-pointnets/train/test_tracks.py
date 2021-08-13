''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
try:
    import cloudpickle as pickle
except:
    import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider
from train_util import get_batch

from lstm_helper import update_batch_features, get_batch_features,batch_track_num
import IPython
import shutil
MIN_MASK = 1
MIN_NUMBER = 0.0000000000001

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--vkitti', action='store_true', default=False)
parser.add_argument('--tracking', action='store_true', default=False)
parser.add_argument('--tracks', action='store_true', default=False)
parser.add_argument('--multi_layer', action='store_true', default=False)
parser.add_argument('--tau', type=int, default=3, help='Number of time steps in total to be processed by temporal processing layers [default: 3]')
parser.add_argument('--track_net', default='conv', help='Network type to process temporal information: lstm or conv [default: conv]')
parser.add_argument('--track_features', default='global', help='Defines from which layer of 3d box estimation network to take features (global: 512+k, fc1: 512, fc2:256) [default: global]')
parser.add_argument('--rnn_cell_type', default='gru', help='RNN cell type. Either gru or lstm. [default:gru]')
parser.add_argument('--layer_sizes', metavar='N', type=int, default=None, nargs='+',help='Layer sizes for the temporal data processing layers. This will be only considered if the multi_layer flag is set. If track_net flag is conv, only the number of layers will be considered, not the sizes (128 64 64 indicates 3 layers). If the track_net flag is lstm, the given list will indicate number of cells in each layer. The last layer will be appended by a fully-connected layerto match the final shape.  ')
parser.add_argument('--two_branch', action='store_true', default=False, help='The track features will be processed in a separate branch and the original branch for amodal bbox estimation network will be kept. The outputs of two branches will be fused.')
parser.add_argument('--only_whl', action='store_true', default=False, help='Only the w,h,l sizes of a bbox will be estimated using track features. Only works if two_branch flag is set.')
parser.add_argument('--temp_attention', action='store_true', default=False, help='Attention weights will be used while fusing the temporal features of the same object from successive frames.')
parser.add_argument('--add_center', action='store_true', default=False, help='The center estimation from T-Net will be used to expand the feature vector from the Amodal Bbox estimation network.')
parser.add_argument('--output_attention', action='store_true', default=False, help='Attention weights will be used while fusing two branches to estimate output (bbox parameters).')
parser.add_argument('--time_indication', action='store_true', default=False, help='Adds time index to the temporal features before feeding into temporal convolutional fusion. ')
parser.add_argument('--dropout', action='store_true', default=False, help='Whether to use dropout for the time inputs. If True, feature vecs of time-steps will be dropped with 0.5 probability. ')
parser.add_argument('--random_time_sampling', action='store_true', default=False, help='If True, n of the tau-1 time steps will be randomly sampled and concatenated. ')
parser.add_argument('--random_n', type=int, default=2, help='Number of time steps that will be sampled randomly from tau-1 previous time steps. random_time_sampling flag should be set. [default: 2]')
parser.add_argument('--preprocess_lstm_input', action='store_true', default=False,help='To apply ResNet block-like preprocessing before the sequence enters RNN.')
parser.add_argument('--add_segm_map', action='store_true', default=False, help='To add the segmentation network features to the rnn features. It is possible to add the conv6 (512), conv7(256), conv9(128), or conv9_dp(128 w/ dropout) features of segmentation network using the segm_map flag. The features a applied a max pool operation and concatenated. ')
parser.add_argument('--segm_map', default='seg_conv9_dp', help='Name of the segmentation network layer to be added to the rnn features. seg_conv6 (w/ 512 feature length), seg_conv7 (w/ 256 feature length), seg_conv9 (w/ 128 feature length), seg_conv9_dp (w/ 128 feature length and dropout). [Default:seg_conv9_dp]')

FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4  # point feature channel
VKITTI = FLAGS.vkitti
if not FLAGS.vkitti:
    TRACKING = FLAGS.tracking

if FLAGS.track_features == 'global': 
    feat_len = 515
elif FLAGS.track_features == 'fc1':
    feat_len = 512
elif FLAGS.track_features == 'fc2':
    feat_len = 256
else: 
    feat_len = 515

if FLAGS.add_center:
    feat_len +=3

if FLAGS.add_segm_map:
    if FLAGS.segm_map == 'seg_conv9_dp' or FLAGS.segm_map == 'seg_conv9':
        feat_len += 128
    elif FLAGS.segm_map == 'seg_conv7':
        feat_len += 256
    elif FLAGS.segm_map == 'seg_conv6':
        feat_len += 512
    else:
        print('!!!! Unrecognized segmentation layer !!!!!!!')



lstm_parameters={}
lstm_parameters['n_batch'] = BATCH_SIZE
lstm_parameters['feat_vec_len'] = feat_len
if FLAGS.random_time_sampling:
    lstm_parameters['tau'] = FLAGS.random_n + 1 
else:
    lstm_parameters['tau'] = FLAGS.tau
lstm_parameters['cos_loss'] = None
lstm_parameters['cos_loss_propagate'] = None
lstm_parameters['global_step'] = None
lstm_parameters['cos_loss_batch_thr'] = None
lstm_parameters['track_net'] = FLAGS.track_net
lstm_parameters['track_features'] = FLAGS.track_features
lstm_parameters['layer_sizes'] = FLAGS.layer_sizes
lstm_parameters['cell_type'] = FLAGS.rnn_cell_type
lstm_parameters['multi_layer'] = FLAGS.multi_layer
lstm_parameters['flags'] = dict()
lstm_parameters['flags']['two_branch'] = FLAGS.two_branch
if not FLAGS.two_branch:
    lstm_parameters['flags']['one_branch'] = True
else:
    lstm_parameters['flags']['one_branch'] = False
lstm_parameters['flags']['only_whl'] = FLAGS.only_whl
lstm_parameters['flags']['temp_attention'] = FLAGS.temp_attention
lstm_parameters['flags']['add_center'] = FLAGS.add_center
lstm_parameters['flags']['output_attention'] = FLAGS.output_attention
lstm_parameters['flags']['time_indication'] = FLAGS.time_indication
lstm_parameters['flags']['dropout'] = FLAGS.dropout
lstm_parameters['flags']['random_time_sampling'] = False
lstm_parameters['random_n'] = FLAGS.random_n
lstm_parameters['flags']['preprocess_lstm_input'] = FLAGS.preprocess_lstm_input
lstm_parameters['flags']['add_segm_map'] = FLAGS.add_segm_map
lstm_parameters['flags']['segm_map'] = FLAGS.segm_map
# Load Frustum Datasets.
if VKITTI:
    # Load Virtual KITTI dataset
    overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_caronly_vkitti_val.pickle')
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', 
                                           rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_val_data_path,
                                           tracks=FLAGS.tracking,tau=FLAGS.tau,feat_len=feat_len)
elif TRACKING:
    # Load KITTI tracking dataset
    if FLAGS.from_rgb_detection:
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/tracking_val_rgb_detection.pickle')
    else:
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_tracking_val.pickle')
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', 
                                           rotate_to_center=True, one_hot=True,
                                           from_rgb_detection=FLAGS.from_rgb_detection,
                                           overwritten_data_path=FLAGS.data_path,
                                           tracks=FLAGS.tracking,tau=FLAGS.tau,feat_len=feat_len)
else:
    # Load KITTI dataset
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
        rotate_to_center=True, overwritten_data_path=FLAGS.data_path,
        from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)

def get_session_and_ops(batch_size, num_point, num_channel,lstm_params=None):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(batch_size, num_point, num_channel)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl,track=FLAGS.tracks,lstm_params=lstm_params)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size,track_data = None):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score 
   
    ep = ops['end_points'] 
    for i in range(int(num_batches)):
        if FLAGS.tracks:
            batch_feat_lstm,batch_seq_len = track_data
            feed_dict = {
                ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
                ops['is_training_pl']: False,
                ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}
            
        else:
            feed_dict = {
                ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
                ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
                ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)

        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,

        mask_mean_prob[mask_mean_prob == 0] = 1
        batch_seg_mask_sum = np.sum(batch_seg_mask, 1)
        batch_seg_mask_sum[batch_seg_mask_sum == 0] = MIN_MASK

        mask_mean_prob = mask_mean_prob / batch_seg_mask_sum # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        #batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        batch_scores = (mask_mean_prob + heading_prob + size_prob)/3
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
        # Finished computing scores
        #IPython.embed()
    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list,
                            heading_cls_list, heading_res_list,
                            size_cls_list, size_res_list,
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

def write_track_detection_results(result_dir, id_list, type_list, box2d_list, center_list,
                            heading_cls_list, heading_res_list,
                            size_cls_list, size_res_list,
                            rot_angle_list, score_list,dataset):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10.0 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        world_id = dataset.world_id_list[idx]
        frame_id = dataset.image_id_list[idx]
        if world_id not in results: results[world_id]={}
        if frame_id not in results[world_id]: results[world_id][frame_id]=[]
        results[world_id][frame_id].append(output_str)
        #if idx not in results: results[idx] = []
        #results[idx].append(output_str)
    drives = list(set(dataset.world_id_list))
    drives.sort()
    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    
    i_common_folder_fr = 0 ## Frame ID to use while writing predictions to the common folder (the folder that contains all validation drives)
    for i_dr,drive in enumerate(drives):
        ## Each drive will be written to separate folders 
        drive_path = os.path.join(output_dir,'%04d'%drive)
        if not os.path.exists(drive_path): os.mkdir(drive_path)
        drive_path = os.path.join(drive_path,'data')
        if not os.path.exists(drive_path): 
            os.mkdir(drive_path)
        else:
            shutil.rmtree(drive_path)
            os.mkdir(drive_path)
        
        ## In addition, all results of all validation drives will be written to the same folder for an evaluation at once
        drive_path_same = os.path.join(output_dir,'%04d'%98)
        if not os.path.exists(drive_path_same): os.mkdir(drive_path_same)
        drive_path_same = os.path.join(drive_path_same,'data')
        if not os.path.exists(drive_path_same): 
            os.mkdir(drive_path_same)
        else:
            if i_dr == 0:
                shutil.rmtree(drive_path_same)
                os.mkdir(drive_path_same)
                
        ## Write results to the drive folder
        res_drive = results[drive]
        for fr_id in res_drive.keys():
            pred_filename = os.path.join(drive_path, '%06d.txt'%(fr_id))
            fout = open(pred_filename, 'w')
            for line in res_drive[fr_id]:
                fout.write(line+'\n')
            fout.close() 
        ## Write results to the common folder
        #IPython.embed()
        for fr_id in res_drive.keys():
            pred_filename = os.path.join(drive_path_same, '%06d.txt'%(i_common_folder_fr+fr_id))
            fout = open(pred_filename, 'w')
            for line in res_drive[fr_id]:
                fout.write(line+'\n')
            fout.close() 
        
        i_common_folder_fr +=  (dataset.drive_sizes[drive]+1)
        

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def test_from_rgb_detection(output_filename, result_dir=None):
    global lstm_parameters
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    print(len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = len(TEST_DATASET)/batch_size#int((len(TEST_DATASET)+batch_size-1)/batch_size)
    
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT, num_channel=NUM_CHANNEL,
                                    lstm_params=lstm_parameters)
    
    # To get features of all frames
    if FLAGS.tracks:
        for batch_idx in range(int(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            # E: Get also batch_indices which shows the (world_id,frame_id,track_id) of the objects in the batch
            # E: Batch indices are valid (non-empty) only if the tracks flag is True
            '''
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec, batch_indices = \
            '''
            batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec, batch_indices = \
                get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                          NUM_POINT, NUM_CHANNEL,tracks=FLAGS.tracks, from_rgb_detection=True)

            # Emec added the feature line
            # E: Get the features at the prev time steps of the objects in the batch
            batch_feat_lstm = get_batch_features(TEST_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=FLAGS.tau,
                                                 feat_len=feat_len,rev_order=True)
            # E: Get the number of tracks at the tau prev. time steps for each object in the batch: How many of the tau-1 frames before the current frames of the objects contain the same object with the same track id 
            batch_seq_len = batch_track_num(feature_dict=TEST_DATASET.feature_dict,wfts=batch_indices)

            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['is_training_pl']: False,
                         ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                         ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}

            box_est_feature_vec = \
                sess.run(ops['end_points']['box_est_feature_vec'],
                         feed_dict=feed_dict)
                
            update_batch_features(feature_dict=TEST_DATASET.feature_dict,batch_wft=batch_indices,
                                  batch_feat_vecs=box_est_feature_vec)
            
    for batch_idx in range(int(num_batches)):
        # print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec, batch_indices = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

        # Run one batch inference
        if FLAGS.tracks:
            batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data_to_feed,
                    batch_one_hot_to_feed, batch_size=batch_size,
                        track_data=[batch_feat_lstm,batch_seq_len])
        else:
            batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data_to_feed,
                    batch_one_hot_to_feed, batch_size=batch_size)

        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            #score_list.append(batch_scores[i])
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score
            onehot_list.append(batch_one_hot_vec[i])

    if FLAGS.dump_result:
        with open(output_filename, 'wb') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
            pickle.dump(onehot_list, fp)

    # Write detection results for KITTI evaluation
    # print('Number of point clouds: %d' % (len(ps_list)))
    # Write detection results for KITTI evaluation
    if TRACKING:
        write_track_detection_results(result_dir, TEST_DATASET.id_list,
            TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
            heading_cls_list, heading_res_list,
            size_cls_list, size_res_list, rot_angle_list, score_list,dataset=TEST_DATASET)
        
    #write_detection_results(result_dir, TEST_DATASET.id_list,
    #    TEST_DATASET.type_list, TEST_DATASET.box2d_list,
    #    center_list, heading_cls_list, heading_res_list,
    #    size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    '''
    output_dir = os.path.join(result_dir, 'data')
    if FLAGS.idx_path is not None:
        to_fill_filename_list = [line.rstrip()+'.txt' \
            for line in open(FLAGS.idx_path)]
        fill_files(output_dir, to_fill_filename_list)
    '''

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    global lstm_parameters
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = len(TEST_DATASET)/batch_size

    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT, num_channel=NUM_CHANNEL,
                                    lstm_params=lstm_parameters)
    correct_cnt = 0
    
    # To get features of all frames
    if FLAGS.tracks:
        for batch_idx in range(int(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            # E: Get also batch_indices which shows the (world_id,frame_id,track_id) of the objects in the batch
            # E: Batch indices are valid (non-empty) only if the tracks flag is True
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec, batch_indices = \
                get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                          NUM_POINT, NUM_CHANNEL,tracks=FLAGS.tracks)

            # Emec added the feature line
            # E: Get the features at the prev time steps of the objects in the batch
            batch_feat_lstm = get_batch_features(TEST_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=FLAGS.tau,
                                                 feat_len=feat_len,rev_order=True)
            # E: Get the number of tracks at the tau prev. time steps for each object in the batch: How many of the tau-1 frames before the current frames of the objects contain the same object with the same track id 
            batch_seq_len = batch_track_num(feature_dict=TEST_DATASET.feature_dict,wfts=batch_indices)

            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: False,
                         ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                         ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}

            box_est_feature_vec = \
                sess.run(ops['end_points']['box_est_feature_vec'],
                         feed_dict=feed_dict)
                
            update_batch_features(feature_dict=TEST_DATASET.feature_dict,batch_wft=batch_indices,
                                  batch_feat_vecs=box_est_feature_vec)

    
    print('Inference started!')
    for batch_idx in range(int(num_batches)):
        
        # print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        
        if FLAGS.tracks:
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec,batch_indices = \
                get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL,tracks=FLAGS.tracks)
            batch_feat_lstm = get_batch_features(TEST_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=FLAGS.tau,
                                                 feat_len=feat_len,rev_order=True)
            batch_seq_len = batch_track_num(feature_dict=TEST_DATASET.feature_dict,wfts=batch_indices)
            batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data,
                    batch_one_hot_vec, batch_size=batch_size,
                    track_data=[batch_feat_lstm,batch_seq_len])
            
            
        else:
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec,batch_indices = \
                get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                    NUM_POINT, NUM_CHANNEL,tracks=FLAGS.tracking)
            batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data,
                    batch_one_hot_vec, batch_size=batch_size)

        correct_cnt += np.sum(batch_output==batch_label)

        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])

    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(batch_size*num_batches*NUM_POINT)))

    if FLAGS.dump_result:
        with open(output_filename, 'wb') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    if TRACKING:
        write_track_detection_results(result_dir, TEST_DATASET.id_list,
            TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
            heading_cls_list, heading_res_list,
            size_cls_list, size_res_list, rot_angle_list, score_list,dataset=TEST_DATASET)
    else:
        write_detection_results(result_dir, TEST_DATASET.id_list,
            TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
            heading_cls_list, heading_res_list,
            size_cls_list, size_res_list, rot_angle_list, score_list)

if __name__=='__main__':
    if FLAGS.from_rgb_detection:
        test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    else:
        test(FLAGS.output+'.pickle', FLAGS.output)
