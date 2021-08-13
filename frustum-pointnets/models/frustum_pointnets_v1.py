''' Frustum PointNets v1 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'train'))

import tf_util

from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss

from lstm_helper import lstm_layer_box_est,conv_layer_box_est,\
                        multilayer_lstm_box_est,multilayer_conv_box_est

import IPython 

def get_instance_seg_v1_net(point_cloud, one_hot_vec,
                            is_training, bn_decay, end_points):
    ''' 3D instance segmentation PointNet v1 network.
    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        end_points: dict
    Output:
        logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
        end_points: dict
    '''
    num_point = point_cloud.get_shape()[1].value

    net = tf.expand_dims(point_cloud, 2)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point, 1],
                                     padding='VALID', scope='maxpool')

    global_feat = tf.concat([global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3)

    end_points['global_features'] = global_feat

    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])

    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = tf_util.conv2d(concat_feat, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    end_points['seg_conv6'] = tf.squeeze(tf_util.max_pool2d(net,[num_point, 1],padding='VALID', scope='maxpool_conv6'))
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    end_points['seg_conv7'] = tf.squeeze(tf_util.max_pool2d(net,[num_point, 1],padding='VALID', scope='maxpool_conv7'))
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    end_points['seg_conv9'] = tf.squeeze(tf_util.max_pool2d(net,[num_point, 1],padding='VALID', scope='maxpool_conv9'))
    net = tf_util.dropout(net, is_training, 'dp1', keep_prob=0.5)
    end_points['seg_conv9_dp'] = tf.squeeze(tf_util.max_pool2d(net,[num_point, 1],padding='VALID', scope='maxpool_conv9_dp'))
    logits = tf_util.conv2d(net, 2, [1, 1],
                            padding='VALID', stride=[1, 1], activation_fn=None,
                            scope='conv10')

    logits = tf.squeeze(logits, [2])  # BxNxC
    return logits, end_points


def get_3d_box_estimation_v1_net(object_point_cloud, one_hot_vec,
                                 is_training, bn_decay, end_points,
                                 track=False, lstm_params=None,center_est=None):
    ''' 3D Box Estimation PointNet v1 network.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in object coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            including box centers, heading bin class scores and residuals,
            and size cluster scores and residuals
    '''
    def add_track_layers(track,lstm_params,net,end_points,is_training,bn_decay,center=None):
        '''
        lstm_parameters['layer_sizes'] = FLAGS.layer_sizes
            lstm_parameters['cell_type'] = FLAGS.rnn_cell_type
            lstm_parameters['multi_layer'] = FLAGS.multi_layer
        '''
        lstm_params['is_training'] = is_training
        net_exp = net
        if lstm_params['flags']['add_center']:
            net = tf.concat((net,center_est),axis=1)

        
        if lstm_params['flags']['add_segm_map']:
            net = tf.concat((net,end_points[lstm_params['flags']['segm_map']]),axis=1)
        
        box_est_feature_vec = net
        if track and lstm_params is not None:
            if lstm_params['track_net'] == 'lstm':
                # If the features will be used in the lstm layer, add lstm layer before the 
                # fully-connected layers after the global features 
                if lstm_params['multi_layer']:
                    net_exp = multilayer_lstm_box_est(feature_vec=box_est_feature_vec,n_batch=lstm_params['n_batch'],
                                       tau=lstm_params['tau'],vec_len=lstm_params['feat_vec_len'],
                                       end_points=end_points,cos_loss = lstm_params['cos_loss'],
                                       cos_loss_prop=lstm_params['cos_loss_propagate'],
                                       lstm_params=lstm_params)
                else:
                    
                    net_exp = lstm_layer_box_est(feature_vec=box_est_feature_vec,n_batch=lstm_params['n_batch'],
                                       tau=lstm_params['tau'],vec_len=lstm_params['feat_vec_len'],
                                       end_points=end_points,cos_loss = lstm_params['cos_loss'],
                                       cos_loss_prop=lstm_params['cos_loss_propagate'],
                                       lstm_params=lstm_params)
                
            # Using time features with 2D convs
            elif lstm_params['track_net'] == 'conv':
                if lstm_params['multi_layer']:
                    net_exp = multilayer_conv_box_est(feature_vec=box_est_feature_vec,n_batch=lstm_params['n_batch'],
                                   tau=lstm_params['tau'],vec_len=lstm_params['feat_vec_len'],
                                   bn_decay=bn_decay,is_training=is_training,end_points=end_points,
                                   cos_loss = lstm_params['cos_loss'],
                                   cos_loss_prop=lstm_params['cos_loss_propagate'],lstm_params=lstm_params)
                else:
                    net_exp = conv_layer_box_est(feature_vec=box_est_feature_vec,n_batch=lstm_params['n_batch'],
                                       tau=lstm_params['tau'],vec_len=lstm_params['feat_vec_len'],
                                       bn_decay=bn_decay,is_training=is_training,end_points=end_points,
                                       cos_loss = lstm_params['cos_loss'],
                                       cos_loss_prop=lstm_params['cos_loss_propagate'],lstm_params=lstm_params)
            else:
                print('The chosen tracking network is not supported!')
        
        return net_exp,box_est_feature_vec
    
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool2')
    net = tf.squeeze(net, axis=[1, 2])
    net = tf.concat([net, one_hot_vec], axis=1)
    
    
    #######################3
    ## SINGLE BRANCH NETWORK
    #######################
    if lstm_params['flags']['one_branch'] or not track:
        ### Use global feature as tracking features
        if lstm_params['track_features'] == 'global':
            net,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                       is_training,bn_decay,center=center_est)
            
        net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)
        ## Use fc1 outputs as features of track module
        if lstm_params['track_features'] == 'fc1':
            net,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                       is_training,bn_decay,center=center_est)
        net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)
        ## Use fc2 outputs as features of track module
        if lstm_params['track_features'] == 'fc2':
            net,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                       is_training,bn_decay,center=center_est)
            
        # The first 3 numbers: box center coordinates (cx,cy,cz),
        # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
        # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        output = tf_util.fully_connected(net, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4, activation_fn=None,
                                     scope='fc3')
        end_points['bbox_features'] = output
        end_points['box_est_feature_vec'] = box_est_feature_vec
        
        return output, end_points
    #######################
    ## TWO BRANCH NETWORK
    #######################
    elif lstm_params['flags']['two_branch']:
        ###########
        # Global 
        ###########
        ## Track branch: Use global features as track features
        if lstm_params['track_features'] == 'global':
            net_track,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                             is_training,bn_decay,center=center_est)
        ## Track branch: global -> fc1 
        if track and lstm_params['track_features'] == 'global':
            net_track = tf_util.fully_connected(net_track, 512, scope='fc1-track', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)
        ## Original branch: global -> fc1 
        net = tf_util.fully_connected(net, 512, scope='fc1', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)
        
        ###########
        # fc1 
        ########### 
        ## Track branch: Use fc1 features as track features 
        if lstm_params['track_features'] == 'fc1':
            net_track,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                             is_training,bn_decay,center=center_est)
        ## Track branch: fc1 -> fc2 
        if track:
            if lstm_params['track_features'] == 'global' or lstm_params['track_features'] == 'fc1':
                net_track = tf_util.fully_connected(net_track, 256, scope='fc2-track', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)   
        ## Original branch: fc1 -> fc2  
        net = tf_util.fully_connected(net, 256, scope='fc2', bn=True,
                                      is_training=is_training, bn_decay=bn_decay)
        
        ###########
        # fc2
        ########### 
        ## Track branch: Use fc2 features as track features
        if lstm_params['track_features'] == 'fc2':
            net_track,box_est_feature_vec = add_track_layers(track,lstm_params,net,end_points,
                                                             is_training,bn_decay,center=center_est)
            
        ###########
        # Output
        ########### 
        # The first 3 numbers: box center coordinates (cx,cy,cz),
        # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
        # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        
        ## Only estimates whl sizes using tracks
        if lstm_params['flags']['only_whl']:
            ## Track branch: fc2 -> fc3 (Output)
            output_track = tf_util.fully_connected(net_track, NUM_SIZE_CLUSTER * 4,
                                                   activation_fn=None, scope='fc3-track')
            ## Original branch: fc2 -> fc3 (Output)
            output_orig = tf_util.fully_connected(net, 3 + NUM_HEADING_BIN * 2 ,
                                             activation_fn=None, scope='fc3-regular')
            output = tf.concat((output_orig,output_track),axis=1)
            end_points['bbox_features'] = output
            end_points['box_est_feature_vec'] = box_est_feature_vec
            return output, end_points
        
            
        ## Track branch: fc2 -> fc3 (Output)
        output_track = tf_util.fully_connected(net_track, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
                                               activation_fn=None, scope='fc3-track')
        ## Original branch: fc2 -> fc3 (Output)
        output_orig = tf_util.fully_connected(net, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
                                         activation_fn=None, scope='fc3-regular')
        ## Output attention determines the contribution of track output and original output at the final output
        ## feature vector. This is constructed per element of feature vector.
        if lstm_params['flags']['output_attention']:
            ## Prepare both output vectors for the convolution operation
            exp_output = tf.expand_dims(tf.expand_dims(output_orig,axis=2),axis=3)
            exp_output_track = tf.expand_dims(tf.expand_dims(output_track,axis=2),axis=3)
            output_concat = tf.concat((exp_output,exp_output_track),axis=3)
            ## Convolution to obtain attention weights: (batch, feature_len,1,2) -> 
            ## 2 represents track and original branches
            outp_attention = tf_util.conv2d(output_concat, 2, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=is_training,
                                 scope='outp-attention', bn_decay=bn_decay)
            ## Softmax to have sum of weights of the same feature in two channels as 1.
            outp_softmax = tf.nn.softmax(outp_attention,axis=3)
            ## Modulate output brances with softmax weights
            output_before_sum = output_concat * outp_softmax
            ## Output is the weighted-sum of two branches
            output = tf.squeeze(output_before_sum[:,:,:,0:1]+output_before_sum[:,:,:,1:2])
            try:
                end_points['lstm_layer_summary']['output_attention_wei'] = outp_softmax
            except:
                end_points['lstm_layer_summary'] = dict()
                end_points['lstm_layer_summary']['output_attention_wei'] = outp_softmax
                
        else:
            ## If output-attention is not used, the normal averaging to two branches will be applied
            output = (output_track + output_orig) / 2.
        
        end_points['bbox_features'] = output
        end_points['box_est_feature_vec'] = box_est_feature_vec
        return output, end_points


def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None,track=False,
              lstm_params=None):
    ''' Frustum PointNets model. The model predict 3D object masks and
    amodel bounding boxes for objects in frustum point clouds.

    Input:
        point_cloud: TF tensor in shape (B,N,4)
            frustum point clouds with XYZ and intensity in point channels
            XYZs are in frustum coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        is_training: TF boolean scalar
        bn_decay: TF float scalar
        track   : true to use track ids for the amodal box estimation pointnet
        lstm_params: A dictionary that contains lstm parameters; n_batch, tau, feat_vec_len
            This is only used if track is True. 
    Output:
        end_points: dict (map from name strings to TF tensors)
    '''
    end_points = {}
    lstm_params['bn_decay']=bn_decay
    # 3D Instance Segmentation PointNet
    logits, end_points = get_instance_seg_v1_net(point_cloud, one_hot_vec, is_training, bn_decay, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(point_cloud, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(object_point_cloud_xyz, one_hot_vec, is_training, bn_decay,
                                                         end_points)
    stage1_center = center_delta + mask_xyz_mean  # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = get_3d_box_estimation_v1_net(object_point_cloud_xyz_new, one_hot_vec, is_training, bn_decay,
                                                      end_points,track=track, lstm_params=lstm_params,center_est=stage1_center)
    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center  # Bx3
    
    return end_points


# if __name__=='__main__':
#     with tf.Graph().as_default():
#         with tf.Session() as sess:
#             inputs = tf.zeros((25,1024,4))
#             outputs = get_model(inputs, tf.ones((25,1)), tf.constant(True))
#             for key in outputs:
#                 print((key, outputs[key]))
#             loss = get_loss(tf.random.uniform((25,1024),minval=0,maxval=2, dtype=tf.int32),
#                 tf.random.uniform((25,3),minval=0,maxval=2,), tf.random.uniform((25,),minval=0,maxval=2,dtype=tf.int32),
#                 tf.random.uniform((25,),minval=0,maxval=2,), tf.random.uniform((25,),minval=0,maxval=2,dtype=tf.int32),
#                 tf.random.uniform((25,3),minval=0,maxval=2,), outputs)
#             print(loss)

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((25, 1024, 4))
        outputs = get_model(inputs, tf.ones((25, 1)), tf.constant(True))
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.random.uniform((25, 1024), minval=0, maxval=2, dtype=tf.int32),
                        tf.random.uniform((25, 3), minval=0, maxval=2, ),
                        tf.random.uniform((25,), minval=0, maxval=2, dtype=tf.int32),
                        tf.random.uniform((25,), minval=0, maxval=2, ),
                        tf.random.uniform((25,), minval=0, maxval=2, dtype=tf.int32),
                        tf.random.uniform((25, 3), minval=0, maxval=2, ), outputs)
        print(loss)
