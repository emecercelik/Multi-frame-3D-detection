''' Training Frustum PointNets.

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
from datetime import datetime
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch
from lstm_helper import update_batch_features, get_batch_features,batch_track_num
import IPython

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--vkitti', action='store_true', default=False)
parser.add_argument('--tracking', action='store_true', default=False)
parser.add_argument('--ground', action='store_true', default=False)
parser.add_argument('--tracks', action='store_true', default=False)
parser.add_argument('--cos_loss', action='store_true', default=False)
parser.add_argument('--multi_layer', action='store_true', default=False)
parser.add_argument('--cos_loss_weight', type=float, default=1.0, help='Weight of cosine loss [default: 1.0]')
parser.add_argument('--tau', type=int, default=3, help='Number of time steps in total to be processed by temporal processing layers [default: 3]')
parser.add_argument('--cos_loss_prop', type=int, default=2, help='The number of previous steps that the cosine loss will be applied. Max can be tau-1')
parser.add_argument('--cos_loss_batch_thr', type=int, default=-1, help='The loss will take effect after this number of batches processed.')
parser.add_argument('--track_net', default='conv', help='Network type to process temporal information: lstm or conv [default: conv]')
parser.add_argument('--track_features', default='global', help='Defines from which layer of 3d box estimation network to take features (global: 512+k, fc1: 512, fc2:256) [default: global]')
parser.add_argument('--rnn_cell_type', default='gru', help='RNN cell type. Either gru or lstm. [default:gru]')
parser.add_argument('--layer_sizes', metavar='N', type=int, default=None, nargs='+',help='Layer sizes for the temporal data processing layers. This will be only considered if the multi_layer flag is set. If track_net flag is conv, only the number of layers will be considered, not the sizes (128 64 64 indicates 3 layers). If the track_net flag is lstm, the given list will indicate number of cells in each layer. The last layer will be appended by a fully-connected layerto match the final shape.  ')
parser.add_argument('--pickle_name', default=None, help='Pickle name to use for training and validation. This name will be extended with _train.pickle and _val.pickle for training and validation data respectively. [Default:None]')
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
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4  # point feature channel
NUM_CLASSES = 2  # segmentation has two classes
VKITTI = FLAGS.vkitti
if not FLAGS.vkitti:
    TRACKING = FLAGS.tracking
GROUND = FLAGS.ground

# Pay attention, tracks flag is set when the features will be tracked by the track id. This is useless when the ground-truth doesn't have any track ids
#FLAGS.tracks=True
#FLAGS.tau = 3
#FLAGS.cos_loss = True
#FLAGS.cos_loss_prop = 1 # The number of previous steps that the cosine loss will be applied. If 1, cosine loss will be calculated between the features of the current frame and that of one step previous. If 2, then the cosine loss will be calculated with two step previous and the average will be taken.


MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
try:
    os.system('cp %s %s' % (os.path.join(BASE_DIR, 'lstm_helper.py'), LOG_DIR))
except:
    pass
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
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
        

# Load Frustum Datasets. Use default data paths.
if VKITTI:
    overwritten_train_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_caronly_vkitti_train.pickle')
    overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_caronly_vkitti_val.pickle')

    if GROUND:
        overwritten_train_data_path = os.path.join(ROOT_DIR, 'kitti/ground_caronly_vkitti_train.pickle')
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/ground_caronly_vkitti_val.pickle')

    TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train', rotate_to_center=True, random_flip=True,
                                            random_shift=True, one_hot=True,
                                            overwritten_data_path=overwritten_train_data_path,tracks=FLAGS.tracks,tau=FLAGS.tau,feat_len=feat_len)
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_val_data_path,tracks=FLAGS.tracks,tau=FLAGS.tau,feat_len=feat_len)
elif TRACKING:
    if FLAGS.pickle_name is None:
        overwritten_train_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_tracking_train.pickle')
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_tracking_val.pickle')
    else:
        name_train = FLAGS.pickle_name+'_train.pickle'
        name_val = FLAGS.pickle_name+'_val.pickle'
        overwritten_train_data_path = os.path.join(ROOT_DIR, 'kitti',name_train)
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti',name_val)
    print(overwritten_train_data_path,overwritten_val_data_path)
    TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train', rotate_to_center=True, random_flip=True,
                                            random_shift=True, one_hot=True,
                                            overwritten_data_path=overwritten_train_data_path,tracks=FLAGS.tracks,tau=FLAGS.tau,feat_len=feat_len)
    TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, one_hot=True,
                                           overwritten_data_path=overwritten_val_data_path,tracks=FLAGS.tracks,tau=FLAGS.tau,feat_len=feat_len)
else:
    if FLAGS.pickle_name is None:
        TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train', rotate_to_center=True, random_flip=True,
                                            random_shift=True, one_hot=True,feat_len=feat_len)
        TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, one_hot=True,feat_len=feat_len)
    else:
        name_train = FLAGS.pickle_name+'_train.pickle'
        name_val = FLAGS.pickle_name+'_val.pickle'
        overwritten_train_data_path = os.path.join(ROOT_DIR, 'kitti',name_train)
        overwritten_val_data_path = os.path.join(ROOT_DIR, 'kitti',name_val)
        TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train', rotate_to_center=True, random_flip=True,
                                                random_shift=True, one_hot=True,feat_len=feat_len,overwritten_data_path=overwritten_train_data_path)
        TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val', rotate_to_center=True, one_hot=True,feat_len=feat_len,\
                                               overwritten_data_path=overwritten_val_data_path)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    ''' Main function for training and simple evaluation. '''
    
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_CHANNEL)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            
            # Add lstm parameters in end_points to feed while training
            lstm_parameters = {}
            lstm_parameters['n_batch'] = BATCH_SIZE
            if FLAGS.track_features == 'global':
                print('Using global features for tracking!')
                lstm_parameters['feat_vec_len'] = feat_len
            elif FLAGS.track_features == 'fc1':
                print('Using fc1 features for tracking!')
                lstm_parameters['feat_vec_len'] = feat_len
            elif FLAGS.track_features == 'fc2':
                print('Using fc2 features for tracking!')
                lstm_parameters['feat_vec_len'] = feat_len
            else:
                print('Using global features for tracking!')
                lstm_parameters['feat_vec_len'] = feat_len

            lstm_parameters['tau'] = FLAGS.tau
            lstm_parameters['cos_loss'] = FLAGS.cos_loss
            lstm_parameters['cos_loss_propagate'] = FLAGS.cos_loss_prop
            lstm_parameters['global_step'] = batch
            lstm_parameters['cos_loss_batch_thr'] = FLAGS.cos_loss_batch_thr
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
            lstm_parameters['flags']['random_time_sampling'] = FLAGS.random_time_sampling
            lstm_parameters['flags']['preprocess_lstm_input'] = FLAGS.preprocess_lstm_input
            lstm_parameters['flags']['add_segm_map'] = FLAGS.add_segm_map
            lstm_parameters['flags']['segm_map'] = FLAGS.segm_map
            lstm_parameters['random_n'] = FLAGS.random_n
            # Get model and losses
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl, bn_decay=bn_decay, 
                                         track=FLAGS.tracks,lstm_params=lstm_parameters)
            loss = MODEL.get_loss(labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points, cos_loss=FLAGS.cos_loss,
                                  cos_loss_weight=FLAGS.cos_loss_weight)
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
                    
            
            
            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [end_points['center'], end_points['heading_scores'],
                                                                     end_points['heading_residuals'],
                                                                     end_points['size_scores'],
                                                                     end_points['size_residuals'], centers_pl,
                                                                     heading_class_label_pl, heading_residual_label_pl,
                                                                     size_class_label_pl, size_residual_label_pl],
                                        [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds
            iou3ds_wo_NAN = tf.identity(iou3ds)
            iou3ds_wo_NAN.set_shape([None])
            
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))
            tf.summary.scalar('iou_3d_wo_NAN', tf.reduce_mean(tf.boolean_mask(iou3ds_wo_NAN,tf.is_finite(iou3ds_wo_NAN))))

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                               tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                       float(BATCH_SIZE * NUM_POINT)
            tf.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            if FLAGS.tracks:
                try:
                    for key in end_points['lstm_layer_summary'].keys():
                        tf.summary.histogram(key,end_points['lstm_layer_summary'][key])
                except:
                    pass

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            try:
                saver.restore(sess, FLAGS.restore_model_path)
            except:
                init = tf.global_variables_initializer()
                sess.run(init)
                vars_in_checkpoint = tf.train.list_variables(FLAGS.restore_model_path)
                var_list = [v[0] for v in vars_in_checkpoint]
                curr_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                filtered_curr_vars = [v for v in curr_variables if v.name.split(':')[0] in var_list]
                saver2 = tf.train.Saver(max_to_keep=50,var_list=filtered_curr_vars)
                saver2.restore(sess, FLAGS.restore_model_path)

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
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
        ress = []
        eval_box_est_acc_max = -0.01
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            # E: added lstm parameters feed
            res = train_one_epoch(sess, ops, train_writer,tracks=FLAGS.tracks,
                                  lstm_params=lstm_parameters)
            # E: added lstm parameters feed
            eval_box_est_acc = eval_one_epoch(sess, ops, test_writer,tracks=FLAGS.tracks,
                                              lstm_params=lstm_parameters)
            
            save_st = time.time()
            # Save the variables to disk.
            if eval_box_est_acc_max < eval_box_est_acc:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                eval_box_est_acc_max = eval_box_est_acc
                if epoch>=30:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model_{}.ckpt".format(epoch)))
                
            #if epoch % 10 == 0:
            #    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #    log_string("Model saved in file: %s" % save_path)
            
            if epoch == MAX_EPOCH-1:
                if not os.path.exists(os.path.join(LOG_DIR,'tr_all')): os.mkdir(os.path.join(LOG_DIR,'tr_all'))
                saver.save(sess, os.path.join(LOG_DIR,'tr_all', "model.ckpt"))
                saver.save(sess, os.path.join(LOG_DIR, "model_{}.ckpt".format(epoch)))
            save_end = time.time()
            print("{} seconds while saving.".format(save_end-save_st))
            #ress.append(res)
        return res


def train_one_epoch(sess, ops, train_writer,tracks=False,lstm_params=None):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    
    tracks: To utilize the track ids from the dataset and also generate feature dictionaries
    '''
    is_training = True
    log_string(str(datetime.now()))

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) / BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    all_return = []
    # Training with batches
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        # Emec added batch indices
        # E: Get also batch_indices which shows the (world_id,frame_id,track_id) of the objects in the batch
        # E: Batch indices are valid (non-empty) only if the tracks flag is True
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec,batch_indices = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                      NUM_POINT, NUM_CHANNEL,tracks=tracks)
        # E: If the lstm layers will be used 
        if tracks:
            # Emec added the feature line
            # E: Get the features at the prev time steps of the objects in the batch
            batch_feat_lstm = get_batch_features(TRAIN_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=lstm_params['tau'],
                                                 feat_len=lstm_params['feat_vec_len'],rev_order=True)
            # E: Get the number of tracks at the tau prev. time steps for each object in the batch: How many of the tau-1 frames before the current frames of the objects contain the same object with the same track id 
            batch_seq_len = batch_track_num(feature_dict=TRAIN_DATASET.feature_dict,wfts=batch_indices)
            # E: Append the feed dictionary with the lstm parameters
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: is_training,
                         ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                         ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}
  
            # Emec added box_est_feature_vec
            
            summary, step, _, loss_val, logits_val, centers_pred_val, iou2ds, iou3ds, box_est_feature_vec = \
                sess.run(
                    [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['logits'], ops['centers_pred'],
                     ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_est_feature_vec'] ],
                    feed_dict=feed_dict)
        else:
            
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: is_training}
            # Emec added box_est_feature_vec
            summary, step, _, loss_val, logits_val, centers_pred_val, iou2ds, iou3ds, box_est_feature_vec = \
                sess.run(
                    [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['logits'], ops['centers_pred'],
                     ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],ops['end_points']['box_est_feature_vec'] ],
                    feed_dict=feed_dict)
            
            

        train_writer.add_summary(summary, step)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.nansum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds >= 0.7)
        # Emec added below 
        
        if tracks:
            all_return.append([batch_indices,box_est_feature_vec,batch_feat_lstm])
            update_batch_features(feature_dict=TRAIN_DATASET.feature_dict,batch_wft=batch_indices,
                                  batch_feat_vecs=box_est_feature_vec)

    log_string('number of batches: %d' % num_batches)
    log_string('training mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('training segmentation accuracy: %f' % (total_correct / float(total_seen)))
    log_string('training box IoU (ground/3D): %f / %f' % (
        iou2ds_sum / float(num_batches * BATCH_SIZE), iou3ds_sum / float(num_batches * BATCH_SIZE)))
    log_string(
        'training box estimation accuracy (IoU=0.7): %f' % (float(iou3d_correct_cnt) / float(num_batches * BATCH_SIZE)))
    if tracks:
        return all_return


def eval_one_epoch(sess, ops, test_writer,tracks=False,lstm_params=None):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET) / BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0
    iou3d_correct_cnt_old = 0
    iou3d_correct_cnt_05=0
    # E: This is necessary to collect features of batches before the evaluation
    if tracks:
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
                          NUM_POINT, NUM_CHANNEL,tracks=tracks)

            # Emec added the feature line
            # E: Get the features at the prev time steps of the objects in the batch
            batch_feat_lstm = get_batch_features(TEST_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=lstm_params['tau'],
                                                 feat_len=lstm_params['feat_vec_len'],rev_order=True)
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
                         ops['is_training_pl']: is_training,
                         ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                         ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}
            '''
            summary, step, loss_val, logits_val, iou2ds, iou3ds, box_est_feature_vec = \
                sess.run([ops['merged'], ops['step'],
                          ops['loss'], ops['logits'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],
                          ops['end_points']['box_est_feature_vec']],
                         feed_dict=feed_dict)
            '''
            box_est_feature_vec = \
                sess.run(ops['end_points']['box_est_feature_vec'],
                         feed_dict=feed_dict)
                
            update_batch_features(feature_dict=TEST_DATASET.feature_dict,batch_wft=batch_indices,
                                  batch_feat_vecs=box_est_feature_vec)

    # Simple evaluation with batches
    for batch_id in range(int(num_batches)):
        start_idx = batch_id * BATCH_SIZE
        end_idx = (batch_id + 1) * BATCH_SIZE
        # E: Get also batch_indices which shows the (world_id,frame_id,track_id) of the objects in the batch
        # E: Batch indices are valid (non-empty) only if the tracks flag is True
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec, batch_indices = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                      NUM_POINT, NUM_CHANNEL,tracks=tracks)
        
        if tracks:
            # Emec added the feature line
            # E: Get the features at the prev time steps of the objects in the batch
            batch_feat_lstm = get_batch_features(TEST_DATASET.feature_dict,
                                                 batch_wft=batch_indices,tau=lstm_params['tau'],
                                                 feat_len=lstm_params['feat_vec_len'],rev_order=True)
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
                         ops['is_training_pl']: is_training,
                         ops['end_points']['lstm_layer']['feat_input']:batch_feat_lstm,
                         ops['end_points']['lstm_layer']['pf_seq_len']:batch_seq_len}
    
            summary, step, loss_val, logits_val, iou2ds, iou3ds, box_est_feature_vec = \
                sess.run([ops['merged'], ops['step'],
                          ops['loss'], ops['logits'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],
                          ops['end_points']['box_est_feature_vec']],
                         feed_dict=feed_dict)
            
            update_batch_features(feature_dict=TEST_DATASET.feature_dict,batch_wft=batch_indices,
                                  batch_feat_vecs=box_est_feature_vec)
        else:
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: is_training}
    
            summary, step, loss_val, logits_val, iou2ds, iou3ds = \
                sess.run([ops['merged'], ops['step'],
                          ops['loss'], ops['logits'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                         feed_dict=feed_dict)
        test_writer.add_summary(summary, step+batch_id)

        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label == l)
            total_correct_class[l] += (np.sum((preds_val == l) & (batch_label == l)))
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.nansum(iou3ds)
        #IPython.embed()
        iou3d_correct_cnt_old += np.sum(iou3ds >= 0.7)
        #iou3d_correct_cnt += np.sum(iou3ds >= 0.7)
        # class specific IoU-based accuracy calculation
        cl = np.argmax(batch_one_hot_vec,axis=1)
        cl_ids = list(set(cl))
        for _cl in cl_ids:
            cl_iou3ds = iou3ds[np.where(cl==_cl)]
            if _cl == 0:
                iou3d_correct_cnt += np.sum(cl_iou3ds>=0.7)
            else:
                iou3d_correct_cnt += np.sum(cl_iou3ds>=0.5)
                
        iou3d_correct_cnt_05 += np.sum(iou3ds >= 0.5)
        for i in range(BATCH_SIZE):
            segp = preds_val[i, :]
            segl = batch_label[i, :]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                    part_ious[l] = 1.0  # class not present
                else:
                    part_ious[l] = np.sum((segl == l) & (segp == l)) / \
                                   float(np.sum((segl == l) | (segp == l)))
                                   
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval segmentation accuracy: %f' % \
               (total_correct / float(total_seen)))
    log_string('eval segmentation avg class acc: %f' % \
               (np.mean(np.array(total_correct_class) / \
                        np.array(total_seen_class, dtype=np.float))))
    log_string('eval box IoU (ground/3D): %f / %f' % \
               (iou2ds_sum / float(num_batches * BATCH_SIZE), iou3ds_sum / \
                float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7&0.5): %f' % \
               (float(iou3d_correct_cnt) / float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.7): %f' % \
               (float(iou3d_correct_cnt_old) / float(num_batches * BATCH_SIZE)))
    log_string('eval box estimation accuracy (IoU=0.5): %f' % \
               (float(iou3d_correct_cnt_05) / float(num_batches * BATCH_SIZE)))

    EPOCH_CNT += 1
    eval_box_est_acc = float(iou3d_correct_cnt_old) / float(num_batches * BATCH_SIZE)
    return eval_box_est_acc

if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()

