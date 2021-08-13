#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:03:58 2020

@author: emec
"""
import numpy as np
import tensorflow as tf
import tf_util
import IPython
# lstm helper funtions for frustum pnet

def return_random_set(w_l,f_l,t_l,length):
    l = len(w_l)
    ind = np.random.randint(low=0,high=l,size=length)
    return w_l[ind],f_l[ind],t_l[ind],ind

def gen_feature_dict(w,f,t,features=None,tau=None,feat_len=515):
    '''
    To generate dictionary from world_id, frame_id, and track_id including features of related objects.
    
    w : A list of world_ids for all objects: [0,0,0,0,...,1,1,1,...,2,2,2], each entry is an object
    f : A list of frame_ids for all objects: [0,0,1,1,...,0,1,1,...,151,151,151], each entry is an object
    t : A list of track_ids for all objects: [0,1,0,1,...,0,0,1,...,50,51,52], each entry is an object
    features : A list of features for all objects
    tau      : total number of frames to be fed into recurrent layers including the current frame.
    Returns a dictionary that contains all the given information
        feature_dict = {world_id_0 : { 
                            frame_id_0: { 
                                track_id_0: {
                                    'feature': []
                                    'track'  : []
                                },
                                track_id_1: {
                                    'feature': []
                                    'track'  : []
                                }, 
                                ...},
                            frame_id_1: {...},}
                        world_id_1: {...}
                        }
       
    '''
    feature_dict = {}
    for i,(ww,ff,tt) in enumerate(zip(w,f,t)):
        
        if ww not in feature_dict.keys():
            feature_dict[ww]={}
            feature_dict[ww][ff]={}
        else:
            if ff not in feature_dict[ww].keys():
                feature_dict[ww][ff] = {}
        
        if features is None:
            feature = np.zeros(feat_len)
        else:
            feature = features[i]
        feature_dict[ww][ff][tt] = {'feature':feature}
        if tau is not None:
            gen_track_ids(feature_dict,[ww,ff,tt],tau=tau)
    return feature_dict

def gen_track_ids(feature_dict,wft,tau=5):
    '''
    To assign objects in different frames to themselves using track IDs provided in the feature_dict
    
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    wft         : A tuple or list of (world_id, frame_id, track_id) to get the related tracks
    tau         : Total number of steps including the current time step (therefore # of track ids that will be returned is tau-1)
    
    In the end, this method adds 'track' dict into the related sub-dictionary of feature_dict with wft that contains track ids with world_id,frame_id,track_id
    The track_ids of objects in different frames are the same with each other.
    '''
    w,f,t = wft
    tracks = []
    if w in feature_dict.keys():
        for i in range(tau-1):
            #frame_id = f-i-1 # To start from the closest frame
            frame_id = f-(tau-1)+i # To start from the farthest frame
            if frame_id in feature_dict[w].keys():
                if t in feature_dict[w][frame_id].keys():
                    tracks.append([w,frame_id,t])
                else:
                    tracks.append(None)
            else:
                tracks.append(None)
    else:
        for i in range(tau-1):
            tracks.append(None)
    feature_dict[w][f][t]['track']=tracks

def get_features(feature_dict,wft,tau,feat_len,rev_order=False):
    '''
    To get features of each track concatenated for feeding into lstm layers
    
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    wft         : A tuple or list of (world_id, frame_id, track_id) to get the related tracks
    tau         : Max number of features or tracks ()
    feat_len    : Length of feature vecs
    rev_order   : Returns features in the reverse array order => [t-n, t-n+1, ..., t-2,t-1]
    Returns a numpy array that contains features with shape (tau-1,len_feature_vector). ind 0 is the closest to the given object wft in time
        feat_arr[0] belongs to the frame-1
        feat_arr[1] belongs to the frame-2
        ...
    '''
    w,f,t = wft
    trcks = feature_dict[w][f][t]['track']
    features = []
    for trk in trcks:
        if trk is not None:
            ww,ff,tt = trk
            features.append(feature_dict[ww][ff][tt]['feature'])
        else:
            features.append(np.zeros(feat_len))
    #return np.array(features)[0:tau-1,:]   
    features = np.reshape(features,(tau-1,feat_len))
    if rev_order:
        return np.flip(features,axis=0)
    else:
        return features

def get_batch_features(feature_dict,batch_wft,tau,feat_len,rev_order=False):
    '''
    To get features of a batch of objects 
    
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    batch_wft   : List of tuples that contain (world_id, frame_id, track_id) of the objects in the batch 
    tau         : Max number of features or tracks ()
    feat_len    : Length of feature vecs
    rev_order   : Returns features in the reverse array order. If True, order in axis=1 => [t-n, t-n+1, ..., t-2,t-1]
    Returns 
        batch_features : len_batch x tau-1 x feat_len shape of numpy array
    '''
    batch_features = []
    for wft in batch_wft:
        # Features that belong to an object in tau-1 frames: If no track assigned the features are zeros
        feat_trck = get_features(feature_dict,wft,tau,feat_len,rev_order=rev_order)
        batch_features.append(feat_trck)
    
    return np.array(batch_features)

def update_features(feature_dict,wft,new_feat_vec):
    '''
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    wft         : A tuple or list of (world_id, frame_id, track_id) to specify where the feature updated
    new_feat_vec: A feature vector that will be placed in the feature_dict
    '''
    #world_id, frame_id, track_id
    w,f,t = wft
    feature_dict[w][f][t]['feature'] = new_feat_vec
    
def update_batch_features(feature_dict,batch_wft,batch_feat_vecs):
    '''
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    batch_wft   : List of tuples that contain (world_id, frame_id, track_id) of the objects in the batch to specify where the feature updated
    batch_feat_vecs: numpy array of n_batch x l_vec that contains feature vector of each object in the batch that will be placed in the feature_dict
    '''
    for wft,feat in zip(batch_wft,batch_feat_vecs):
        update_features(feature_dict,wft,feat)


def multilayer_lstm_box_est(feature_vec,n_batch,tau,vec_len,end_points=None,cos_loss=True,
                       cos_loss_prop=1,lstm_params=None):
    '''
    LSTM layer to process feature vectors of v1 Amodal 3D Box Estimation PointNet
    
    feature_vec : Tensor of feature vec in the PointNet (batch x feature_length)
    n_batch     : Number of objects in a batch
    tau         : Number of time steps for the recurrent layers
    vec_len     : Length of feature vector tensor
    end_points  : If None the method returns the outputs of last step and states 
        with the lstm_input placeholder to feed in the features from the prev. time steps and 
        a placeholder to feed in the number of time steps for each entry in the batch. 
        If a dictionary is fed, the placeholders and the outputs are directly entered in the 
        dictionary and only the output of the last tau from lstm_layers is returned
        If None, returns [outputs[:,-1,:],states],[lstm_input1,pf_seq_len],loss_tens
        if a dict provided, returns outputs[:,-1,:]
        else returns [outputs[:,-1,:],states],[lstm_input1,pf_seq_len],loss_tens
    cos_loss    : If true, the loss is calculated between the features of the current frame and last two frames.
        The loss tensor is added into the end_points or returned 
        end_points['lstm_layer']['loss']
    cos_loss_prop: The number of previous steps that the cosine loss will be applied. 
        If 1, cosine loss will be calculated between the features of the current frame and 
        that of one step previous. If 2, then the cosine loss will be calculated with 
        two step previous and the average will be taken.
    lstm_params : A dictionary to be able to use other lstm parameters from the user. 
    Returns the output of recurrent layers from the last time step and all states
    '''
    # Placeholder to feed previous features of the objects in the batch
    lstm_input1 = tf.placeholder(dtype=tf.float32,shape=(n_batch,tau-1,vec_len),name='feature_placeholder')
    
    if lstm_params['flags']['random_time_sampling']:
        pass
    
    dropout = lstm_params['flags']['dropout']
    if dropout:
        lstm_input1 = tf_util.dropout(lstm_input1,lstm_params['is_training'], 'lstm_dropout',
                                      keep_prob=0.6, noise_shape=[n_batch,1,vec_len])
        #keep_prob=0.2, noise_shape=[n_batch,tau-1,1]
    # A placeholder to feed number of tracks in the previous frames of objects 
    # to have valid rollout in the LSTM layers 
    pf_seq_len = tf.placeholder(dtype=tf.float32,shape=(n_batch),name='tau_length_batch_ph')
    
    # Expand dimension of the feature vector tensor to have the same shape with the prev. features 
    exp_lstm_input2 = tf.expand_dims(feature_vec,axis=1)
    # Concatenate feature vec of objects in the batch with their features from the prev time steps
    # To keep the correct order in the lstm layers -> concat: batch x tau-1 x feat_len + batch x 1 x feat_len
    # feature order in tau dim: [-2,-1,0]

    concat_inputs = tf.concat((lstm_input1,exp_lstm_input2),axis=1)
    if lstm_params['flags']['preprocess_lstm_input'] :
        exp_concat = tf.expand_dims(concat_inputs,axis=3) # batch x tau x feat_len x 1
        preprocessed = tf_util.conv2d(exp_concat, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='preprocess1', bn_decay=lstm_params['bn_decay'])
        preprocessed2 = tf_util.conv2d(preprocessed, 1, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='preprocess2', bn_decay=lstm_params['bn_decay'],activation_fn=None)
        preprocessed3 = tf.nn.relu( preprocessed2 + exp_concat)
        concat_inputs = tf.squeeze(preprocessed3)
    if lstm_params['flags']['time_indication']:
        time_indicator = tf.constant([tau-1-i for i in range(tau)],dtype=concat_inputs.dtype)
        time_indicator = tf.reshape(tf.tile(time_indicator,[tf.shape(concat_inputs)[0]]),[-1,tau]) 
        time_indicator = tf.expand_dims(time_indicator,axis=2)
        concat_inputs = tf.concat((concat_inputs,time_indicator),axis=2)
        zeros = tf.zeros((int(np.shape(feature_vec)[0]),1),dtype=feature_vec.dtype)
        feature_vec = tf.concat((feature_vec,zeros),axis=1) 
        vec_len += 1
    
    if lstm_params['cell_type'] == 'lstm':
        rnn_cell = tf.nn.rnn_cell.LSTMCell
    elif lstm_params['cell_type'] == 'gru':
        rnn_cell = tf.nn.rnn_cell.GRUCell
    elif lstm_params['cell_type'] == 'norm_lstm':
        rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
    else:
        rnn_cell = tf.nn.rnn_cell.LSTMCell
    
    layers = [rnn_cell(size) for size in lstm_params['layer_sizes']]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

    
    outputs, states = tf.nn.dynamic_rnn(cell = multi_rnn_cell,inputs=concat_inputs,\
                                        sequence_length=None,\
                                        time_major=False,dtype=tf.float32)
    
    loss_tens = None
    lstm_return_index = -1
    lstm_output = tf_util.fully_connected(outputs[:,lstm_return_index,:],\
                                          vec_len,scope='lstm-track')
    
    
    if cos_loss:
        if tau==2:
            loss_tens = tf_cosine_loss(feature_vec,lstm_input1[:,-1,:],axis=1)
        else:
            loss_average = 0
            for ii in range(cos_loss_prop):
                loss_average += tf_cosine_loss(feature_vec,lstm_input1[:,-ii-1,:],axis=1)
            #loss2 = tf_cosine_loss(feature_vec,lstm_input1[:,-2,:],axis=1)
            loss_tens = loss_average/cos_loss_prop
        if lstm_params is not None:
            # This is to use the cosine similarity loss only after some batch training
            offset = lstm_params['global_step'] - lstm_params['cos_loss_batch_thr']
            mult = tf.cond(offset>0,true_fn= lambda: 1.0, false_fn= lambda: 0.0)
            loss_tens = loss_tens * mult
    if end_points is None:
        return [outputs[:,lstm_return_index,:],states],[lstm_input1,pf_seq_len],loss_tens
    elif type(end_points) == dict:
        end_points['lstm_layer'] = {}
        end_points['lstm_layer']['outputs'] = lstm_output
        end_points['lstm_layer']['states'] = states
        end_points['lstm_layer']['feat_input'] = lstm_input1
        end_points['lstm_layer']['pf_seq_len'] = pf_seq_len
        end_points['lstm_layer']['loss'] = loss_tens
        return lstm_output
    else:
        return [lstm_output,states],[lstm_input1,pf_seq_len],loss_tens

def lstm_layer_box_est(feature_vec,n_batch,tau,vec_len,end_points=None,cos_loss=True,
                       cos_loss_prop=1,lstm_params=None):
    '''
    LSTM layer to process feature vectors of v1 Amodal 3D Box Estimation PointNet
    
    feature_vec : Tensor of feature vec in the PointNet (batch x feature_length)
    n_batch     : Number of objects in a batch
    tau         : Number of time steps for the recurrent layers
    vec_len     : Length of feature vector tensor
    end_points  : If None the method returns the outputs of last step and states 
        with the lstm_input placeholder to feed in the features from the prev. time steps and 
        a placeholder to feed in the number of time steps for each entry in the batch. 
        If a dictionary is fed, the placeholders and the outputs are directly entered in the 
        dictionary and only the output of the last tau from lstm_layers is returned
        If None, returns [outputs[:,-1,:],states],[lstm_input1,pf_seq_len],loss_tens
        if a dict provided, returns outputs[:,-1,:]
        else returns [outputs[:,-1,:],states],[lstm_input1,pf_seq_len],loss_tens
    cos_loss    : If true, the loss is calculated between the features of the current frame and last two frames.
        The loss tensor is added into the end_points or returned 
        end_points['lstm_layer']['loss']
    cos_loss_prop: The number of previous steps that the cosine loss will be applied. 
        If 1, cosine loss will be calculated between the features of the current frame and 
        that of one step previous. If 2, then the cosine loss will be calculated with 
        two step previous and the average will be taken.
    lstm_params : A dictionary to be able to use other lstm parameters from the user. 
    Returns the output of recurrent layers from the last time step and all states
    '''
    # Placeholder to feed previous features of the objects in the batch
    lstm_input1 = tf.placeholder(dtype=tf.float32,shape=(n_batch,tau-1,vec_len),name='feature_placeholder')
    
    dropout = lstm_params['flags']['dropout']
    if dropout:
        lstm_input1 = tf_util.dropout(lstm_input1,lstm_params['is_training'], 'lstm_dropout',
                                      keep_prob=0.5, noise_shape=[n_batch,1,vec_len])

    # A placeholder to feed number of tracks in the previous frames of objects 
    # to have valid rollout in the LSTM layers 
    pf_seq_len = tf.placeholder(dtype=tf.float32,shape=(n_batch),name='tau_length_batch_ph')
    
    # Expand dimension of the feature vector tensor to have the same shape with the prev. features 
    exp_lstm_input2 = tf.expand_dims(feature_vec,axis=1)
    # Concatenate feature vec of objects in the batch with their features from the prev time steps
    # To keep the correct order in the lstm layers -> concat: batch x tau-1 x feat_len + batch x 1 x feat_len
    # feature order in tau dim: [-2,-1,0]
    #concat_inputs = tf.concat((exp_lstm_input2,lstm_input1),axis=1)
    #IPython.embed()
    #h = tf.slice(sizes, [0, 2], [-1, 1]) 
    concat_inputs = tf.concat((lstm_input1,exp_lstm_input2),axis=1) # batch x tau x feat_len
    #rev_lstm_input1 = tf.reverse(lstm_input1,[1])
    #concat_inputs = tf.concat((exp_lstm_input2,rev_lstm_input1),axis=1)
    #pf_seq_len = tf.add(pf_seq_len,1)
    batch_norm_before_lstm = False
    if batch_norm_before_lstm:
        concat_inputs = tf.expand_dims(concat_inputs,axis=2)
        concat_inputs = tf_util.batch_norm_for_conv2d(concat_inputs, lstm_params['is_training'],
                                        bn_decay=lstm_params['bn_decay'], scope='lstm-bn',
                                        data_format='NHWC')
        concat_inputs = tf.squeeze(concat_inputs)

    instance_norm_before_lstm =False
    if instance_norm_before_lstm:
        concat_inputs = tf.expand_dims(concat_inputs,axis=3)
        concat_inputs = tf.contrib.layers.instance_norm(concat_inputs,data_format='NCHW')
        concat_inputs = tf.squeeze(concat_inputs)
    
    loss_preprocess = None
    if lstm_params['flags']['preprocess_lstm_input'] :
        """
        exp_concat = tf.expand_dims(concat_inputs,axis=3) # batch x tau x feat_len x 1
        preprocessed = tf_util.conv2d(exp_concat, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='preprocess1', bn_decay=lstm_params['bn_decay'])
        preprocessed2 = tf_util.conv2d(preprocessed, 1, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='preprocess2', bn_decay=lstm_params['bn_decay'],activation_fn=None)
        preprocessed3 = tf.nn.relu( preprocessed2 + exp_concat)
        concat_inputs = tf.squeeze(preprocessed3)
        """
        
        exp_concat = tf.expand_dims(concat_inputs,axis=2) # batch x tau x 1 x feat_len     
        tr_exp_concat = tf.transpose(exp_concat,perm=[0,3,2,1]) # batch x feat_len x 1 x tau      
        deepfuse1 = tf_util.conv2d(tr_exp_concat, 2*tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='deepfuse1', bn_decay=lstm_params['bn_decay']) # deep fusion batch x feat_len x 1 x 2*tau
        deepfuse2 = tf_util.conv2d(deepfuse1, 4*tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='deepfuse2', bn_decay=lstm_params['bn_decay']) # deep fusion batch x feat_len x 1 x 4*tau
        deepfuse3 = tf_util.conv2d(deepfuse2, tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='deepfuse3', bn_decay=lstm_params['bn_decay'],activation_fn=None) # Go back to the original feature length to create org feature vector. batch x feat_len x 1 x tau
        deepfuse4 = tf.nn.relu( tr_exp_concat + deepfuse3)
        
        deepfuse5 = tf_util.conv2d(deepfuse4, tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='deepfuse5', bn_decay=lstm_params['bn_decay']) # To regenerate original feature vectors batch x feat_len x 1 x tau
        deepfuse6 = tf_util.conv2d(deepfuse5, tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=lstm_params['is_training'],
                             scope='deepfuse6', bn_decay=lstm_params['bn_decay']) # To regenerate original feature vectors batch x feat_len x 1 x tau

        tr_deepfuse = tf.transpose(deepfuse4,perm=[0,3,2,1]) # batch x tau x 1 x feat_len 
        org_deepfuse = tf.transpose(deepfuse6,perm=[0,3,2,1]) # batch x tau x 1 x feat_len 
        
        pred_concat = tf.squeeze(org_deepfuse) # to apply cosine distance to recreate the org tensors from the reduced ones. batch x tau x feat_len (vec_len)
        
        # loss to regenerate original feature vectors from the reduced feature vectors
        loss_preprocess = 0
        for i_tau in range(tau):
            loss_preprocess += tf_cosine_loss(concat_inputs[:,i_tau,:],pred_concat[:,i_tau,:],axis=1)
        concat_inputs = tf.squeeze(tr_deepfuse) # To continue with further layers  
    
    if lstm_params['flags']['time_indication']:
        time_indicator = tf.constant([tau-1-i for i in range(tau)],dtype=concat_inputs.dtype)
        time_indicator = tf.reshape(tf.tile(time_indicator,[tf.shape(concat_inputs)[0]]),[-1,tau]) 
        time_indicator = tf.expand_dims(time_indicator,axis=2)
        concat_inputs = tf.concat((concat_inputs,time_indicator),axis=2)
        zeros = tf.zeros((int(np.shape(feature_vec)[0]),1),dtype=feature_vec.dtype)
        feature_vec = tf.concat((feature_vec,zeros),axis=1) 
        vec_len += 1
    
    # To change number of units in rnn 
    if lstm_params['layer_sizes'] is not None:
        vec_len_org = vec_len + 0
        vec_len = lstm_params['layer_sizes'][0]
        
    if lstm_params['cell_type'] == 'lstm':
        lstm_layer = tf.nn.rnn_cell.LSTMCell(vec_len,name="lstm_cell")
    elif lstm_params['cell_type'] == 'gru':
        lstm_layer = tf.nn.rnn_cell.GRUCell(vec_len,activation=None,name='gru_cell')
    else:
        lstm_layer = tf.nn.rnn_cell.LSTMCell(vec_len,name="lstm_cell")
    
    
    
    outputs, states = tf.nn.dynamic_rnn(cell = lstm_layer,inputs=concat_inputs,\
                                        sequence_length=None,\
                                        time_major=False,dtype=tf.float32)
    
    loss_tens = None
    lstm_return_index = -1
    if lstm_params['layer_sizes'] is not None:
        lstm_output = tf_util.fully_connected(outputs[:,lstm_return_index,:],\
                                          vec_len_org,scope='lstm-track')
    
    if cos_loss:
        if tau==2:
            loss_tens = tf_cosine_loss(feature_vec,lstm_input1[:,-1,:],axis=1)
        else:
            loss_average = 0
            for ii in range(cos_loss_prop):
                loss_average += tf_cosine_loss(feature_vec,lstm_input1[:,-ii-1,:],axis=1)
            #loss2 = tf_cosine_loss(feature_vec,lstm_input1[:,-2,:],axis=1)
            loss_tens = loss_average/cos_loss_prop
        if lstm_params is not None:
            # This is to use the cosine similarity loss only after some batch training
            offset = lstm_params['global_step'] - lstm_params['cos_loss_batch_thr']
            mult = tf.cond(offset>0,true_fn= lambda: 1.0, false_fn= lambda: 0.0)
            loss_tens = loss_tens * mult
    if end_points is None:
        return [outputs[:,lstm_return_index,:],states],[lstm_input1,pf_seq_len],loss_tens
    elif type(end_points) == dict:
        end_points['lstm_layer'] = {}
        end_points['lstm_layer']['states'] = states
        end_points['lstm_layer']['feat_input'] = lstm_input1
        end_points['lstm_layer']['pf_seq_len'] = pf_seq_len
        if loss_tens is not None:
            if loss_preprocess is not None:
                loss_tens = loss_tens + loss_preprocess
        else:
            if loss_preprocess is not None:
                loss_tens = loss_preprocess
             
        end_points['lstm_layer']['loss'] = loss_tens
        if lstm_params['layer_sizes'] is not None:
            end_points['lstm_layer']['outputs'] = lstm_output#outputs
            return lstm_output#outputs[:,lstm_return_index,:]
        else:
            end_points['lstm_layer']['outputs'] = outputs
            return outputs[:,lstm_return_index,:]
    else:
        return [outputs[:,lstm_return_index,:],states],[lstm_input1,pf_seq_len],loss_tens

def multilayer_conv_box_est(feature_vec,n_batch,tau,vec_len,bn_decay,is_training,\
                            end_points=None,cos_loss=True,\
                            cos_loss_prop=1,lstm_params=None):
    '''
    Convolutional layer to process feature vectors of v1 Amodal 3D Box Estimation PointNet in time
    
    feature_vec : Tensor of feature vec in the PointNet (batch x feature_length)
    n_batch     : Number of objects in a batch
    tau         : Number of time steps for the recurrent layers
    vec_len     : Length of feature vector tensor
    end_points  : If None the method returns the output tensor 
        with the lstm_input placeholder to feed in the features from the prev. time steps and 
        a placeholder to feed in the number of time steps for each entry in the batch. 
        If a dictionary is fed, the placeholders and the outputs are directly entered in the 
        dictionary and only the output of the last tau from lstm_layers is returned
        If None, returns [s_conv_concat],[lstm_input1,pf_seq_len], loss_tens
        if a dict provided, returns s_conv_concat
        else returns [s_conv_concat],[lstm_input1,pf_seq_len], loss_tens
    cos_loss    : If true, the loss is calculated between the features of the current frame and last two frames.
        The loss tensor is added into the end_points or returned 
    cos_loss_prop: The number of previous steps that the cosine loss will be applied. 
        If 1, cosine loss will be calculated between the features of the current frame and 
        that of one step previous. If 2, then the cosine loss will be calculated with 
        two step previous and the average will be taken.
    lstm_params : A dictionary to be able to use other lstm parameters from the user. 
    Returns the output of recurrent layers from the last time step and all states
    '''
    # Placeholder to feed previous features of the objects in the batch
    lstm_input1 = tf.placeholder(dtype=tf.float32,shape=(n_batch,tau-1,vec_len),name='feature_placeholder')
    dropout = lstm_params['flags']['dropout']
    if dropout:
        lstm_input1 = tf_util.dropout(lstm_input1,lstm_params['is_training'], 'lstm_dropout',
                                      keep_prob=0.5, noise_shape=[n_batch,tau-1,1])
    # A placeholder to feed number of tracks in the previous frames of objects 
    # to have valid rollout in the LSTM layers 
    pf_seq_len = tf.placeholder(dtype=tf.float32,shape=(n_batch),name='tau_length_batch_ph')
    
    # Expand dimension of the feature vector tensor to have the same shape with the prev. features 
    exp_lstm_input2 = tf.expand_dims(feature_vec,axis=1)
    # Concatenate feature vec of objects in the batch with their features from the prev time steps
    # To keep the correct order in the lstm layers -> concat: batch x tau-1 x feat_len + batch x 1 x feat_len
    # feature order in tau dim: [-2,-1,0]
    concat_inputs = tf.concat((lstm_input1,exp_lstm_input2),axis=1)
    # Expand the last dimension and transpose to have a relevant shape for convolutions (BxHxWxC)
    exp_concat = tf.expand_dims(concat_inputs,axis=3)
    tr_exp_concat = tf.transpose(exp_concat,perm=[0,2,3,1])
    conv_layers = [tr_exp_concat]
    
    ### Multi-layer fusion part
    for i_layer,layer_size in enumerate(lstm_params['layer_sizes']):
        conv_layers.append(tf_util.conv2d(conv_layers[-1], layer_size, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-track{}'.format(i_layer), bn_decay=bn_decay))
    
    tr_exp_concat = conv_layers[-1]
    ### To apply time indication or temp attention, the num of channels should be tau
    ### The following convolution operation ensures this
    if lstm_params['flags']['time_indication'] or lstm_params['flags']['temp_attention']:
        tr_exp_concat = tf_util.conv2d(tr_exp_concat, tau, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv-track-final', bn_decay=bn_decay)
    
    ### Add time-step indicating indices to the feature vectors
    if lstm_params['flags']['time_indication']:
        time_indicator = tf.constant([tau-1-i for i in range(tau)],dtype=tr_exp_concat.dtype)
        time_indicator = tf.reshape(tf.tile(time_indicator,[tf.shape(tr_exp_concat)[0]]),[-1,tau]) 
        time_indicator = tf.expand_dims(tf.expand_dims(time_indicator,axis=1),axis=2)
        tr_exp_concat = tf.concat((tr_exp_concat,time_indicator),axis=1)
        zeros = tf.zeros((int(np.shape(feature_vec)[0]),1),dtype=feature_vec.dtype)
        feature_vec = tf.concat((feature_vec,zeros),axis=1)
    
    ### Add temporal attention parts
    if lstm_params['flags']['temp_attention']:

        attention_softmax = tf.nn.softmax(tr_exp_concat,axis=3)
        attention_modulated = attention_softmax*tr_exp_concat

        tr_conv_concat_reduce_sum = tf.math.reduce_sum(attention_modulated,axis=3)
        s_conv_concat = tf.squeeze(tr_conv_concat_reduce_sum) + feature_vec
        try:
            end_points['lstm_layer_summary']['temp_attention_wei']=attention_softmax
        except:
            end_points['lstm_layer_summary'] = dict()
            end_points['lstm_layer_summary']['temp_attention_wei']=attention_softmax
    else:
        conv_concat = tf_util.conv2d(tr_exp_concat, 1, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv-track_l2', bn_decay=bn_decay)
        s_conv_concat = tf.squeeze(conv_concat)
    
    
    loss_tens = None
    if cos_loss:
        if tau==2:
            loss_tens = tf_cosine_loss(feature_vec,lstm_input1[:,-1,:],axis=1)
        else:
            loss_average = 0
            for ii in range(cos_loss_prop):
                loss_average += tf_cosine_loss(feature_vec,lstm_input1[:,-ii-1,:],axis=1)
            #loss2 = tf_cosine_loss(feature_vec,lstm_input1[:,-2,:],axis=1)
            loss_tens = loss_average/cos_loss_prop
        if lstm_params is not None:
            # This is to use the cosine similarity loss only after some batch training
            offset = lstm_params['global_step'] - lstm_params['cos_loss_batch_thr']
            mult = tf.cond(offset>0,true_fn= lambda: 1.0, false_fn= lambda: 0.0)
            loss_tens = loss_tens * mult
    if end_points is None:
        return [s_conv_concat],[lstm_input1,pf_seq_len],loss_tens
    elif type(end_points) == dict:
        end_points['lstm_layer'] = {}
        end_points['lstm_layer']['outputs'] = s_conv_concat
        end_points['lstm_layer']['feat_input'] = lstm_input1
        end_points['lstm_layer']['pf_seq_len'] = pf_seq_len
        end_points['lstm_layer']['loss'] = loss_tens
        return s_conv_concat
    else:
        return [s_conv_concat],[lstm_input1,pf_seq_len],loss_tens

def max_pool2d_ext(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 3 ints -> HxWxC
    stride: a list of 3 ints -> HxWxC
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w, kernel_c = kernel_size
    stride_h, stride_w, stride_c = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, kernel_c],
                             strides=[1, stride_h, stride_w, stride_c],
                             padding=padding,
                             name=sc.name)
    return outputs


def conv_layer_box_est(feature_vec,n_batch,tau,vec_len,bn_decay,is_training,end_points=None,cos_loss=True,
                       cos_loss_prop=1,lstm_params=None):
    '''
    Convolutional layer to process feature vectors of v1 Amodal 3D Box Estimation PointNet in time
    
    feature_vec : Tensor of feature vec in the PointNet (batch x feature_length)
    n_batch     : Number of objects in a batch
    tau         : Number of time steps for the recurrent layers
    vec_len     : Length of feature vector tensor
    end_points  : If None the method returns the output tensor 
        with the lstm_input placeholder to feed in the features from the prev. time steps and 
        a placeholder to feed in the number of time steps for each entry in the batch. 
        If a dictionary is fed, the placeholders and the outputs are directly entered in the 
        dictionary and only the output of the last tau from lstm_layers is returned
        If None, returns [s_conv_concat],[lstm_input1,pf_seq_len], loss_tens
        if a dict provided, returns s_conv_concat
        else returns [s_conv_concat],[lstm_input1,pf_seq_len], loss_tens
    cos_loss    : If true, the loss is calculated between the features of the current frame and last two frames.
        The loss tensor is added into the end_points or returned 
    cos_loss_prop: The number of previous steps that the cosine loss will be applied. 
        If 1, cosine loss will be calculated between the features of the current frame and 
        that of one step previous. If 2, then the cosine loss will be calculated with 
        two step previous and the average will be taken.
    lstm_params : A dictionary to be able to use other lstm parameters from the user. 
    Returns the output of recurrent layers from the last time step and all states
    '''
    # Placeholder to feed previous features of the objects in the batch
    lstm_input1 = tf.placeholder(dtype=tf.float32,shape=(n_batch,tau-1,vec_len),name='feature_placeholder')
    dropout = lstm_params['flags']['dropout']
    if dropout:
        lstm_input1 = tf_util.dropout(lstm_input1,lstm_params['is_training'], 'lstm_dropout',
                                      keep_prob=0.5, noise_shape=[n_batch,tau-1,1])
    
    # A placeholder to feed number of tracks in the previous frames of objects 
    # to have valid rollout in the LSTM layers 
    pf_seq_len = tf.placeholder(dtype=tf.float32,shape=(n_batch),name='tau_length_batch_ph')
    
    # Expand dimension of the feature vector tensor to have the same shape with the prev. features 
    exp_lstm_input2 = tf.expand_dims(feature_vec,axis=1)
    
    # Concatenate feature vec of objects in the batch with their features from the prev time steps
    # To keep the correct order in the lstm layers -> concat: batch x tau-1 x feat_len + batch x 1 x feat_len
    # feature order in tau dim: [-2,-1,0]
    concat_inputs = tf.concat((lstm_input1,exp_lstm_input2),axis=1)
    # Expand the last dimension and transpose to have a relevant shape for convolutions (BxHxWxC)
    exp_concat = tf.expand_dims(concat_inputs,axis=3)
    tr_exp_concat = tf.transpose(exp_concat,perm=[0,2,3,1])
    tr_exp_concat_init = tr_exp_concat
    tr_exp_concat = tf_util.conv2d(tr_exp_concat, 100, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='temp_fusion1', bn_decay=bn_decay)
    tr_exp_concat = tf_util.conv2d(tr_exp_concat, tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='temp_fusion2', bn_decay=bn_decay)
    tr_exp_concat = tr_exp_concat_init + tr_exp_concat
    if lstm_params['flags']['time_indication']:
        time_indicator = tf.constant([tau-1-i for i in range(tau)],dtype=tr_exp_concat.dtype)
        time_indicator = tf.reshape(tf.tile(time_indicator,[tf.shape(tr_exp_concat)[0]]),[-1,tau]) 
        time_indicator = tf.expand_dims(tf.expand_dims(time_indicator,axis=1),axis=2)
        tr_exp_concat = tf.concat((tr_exp_concat,time_indicator),axis=1)
        zeros = tf.zeros((int(np.shape(feature_vec)[0]),1),dtype=feature_vec.dtype)
        feature_vec = tf.concat((feature_vec,zeros),axis=1)
    
    if lstm_params['flags']['temp_attention']:
        # Non-local attention 
        num_channel = 128
        # T = tau, F= feature length
        a1 = tf.transpose(tr_exp_concat, perm=[0,3,1,2]) #  BxFx1xT-> BxTxFx1
        a_conv1 = tf_util.conv2d(a1, num_channel, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='att1', bn_decay=bn_decay) # B x T x F x num_channel
        
        a_conv2 = tf_util.conv2d(a1, num_channel, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='att2', bn_decay=bn_decay) # B x T x F x num_channel
        
        a_conv3 = tf_util.conv2d(a1, num_channel, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='att3', bn_decay=bn_decay) # B x T x F x num_channel
        sh_a_conv1 = [ int(i) for i in np.shape(a_conv1)] # sh_a_conv1 = tf.shape(a_conv1)       
        a_conv1_rs = tf.reshape(a_conv1, [sh_a_conv1[0],sh_a_conv1[1]*sh_a_conv1[2],sh_a_conv1[3]]) # B x TF x num_channel
        a_conv2_rs = tf.reshape(a_conv2, [sh_a_conv1[0],sh_a_conv1[1]*sh_a_conv1[2],sh_a_conv1[3]]) # B x TF x num_channel
        a_conv2_rs_tr = tf.transpose(a_conv2_rs, perm=[0,2,1]) # B x num_channel x TF
        a_conv1_a = tf.matmul(a_conv1_rs,a_conv2_rs_tr) # B x TF x TF
        a_conv1_sm = tf.nn.softmax(a_conv1_a,axis=2)
        a_conv3_rs = tf.reshape(a_conv3, [sh_a_conv1[0],sh_a_conv1[1]*sh_a_conv1[2],sh_a_conv1[3]]) # B x TF x num_channel
        a_att = tf.matmul(a_conv1_sm,a_conv3_rs) # B x TF x num_channel 
        a_att_rs = tf.reshape(a_att, [sh_a_conv1[0], sh_a_conv1[1], sh_a_conv1[2],sh_a_conv1[3]])
        a_att_conv = tf_util.conv2d(a_att_rs, 1, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='att4', bn_decay=bn_decay) # B x T x F x 1
        att_fin = a_att_conv + a1
        att_fin_tr = tf.transpose(att_fin, perm=[0,2,1,3]) # B x F x T x 1
        att_fin_tr_mp = tf_util.max_pool2d(att_fin_tr,kernel_size=[1,tau],scope='att_maxpool',stride=[1,1]) # B x F x 1 x 1
        s_conv_concat = tf.squeeze(att_fin_tr_mp) # B x F 
        
        """
        '''
        # Channel-based attention weights (B x Feature_len x 1 x Tau)
        attention_outp = tf_util.conv2d(tr_exp_concat, tau, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='concat-attention', bn_decay=bn_decay)
        '''
        attention_softmax = tf.nn.softmax(tr_exp_concat,axis=3)
        attention_modulated = attention_softmax*tr_exp_concat
        '''
        # Apply max pool in channel axis. To do that a permutation is necessary since there is no 3D max pool implemented
        tr_conv_concat = tf.transpose(attention_modulated,perm=[0,1,3,2])
        conv_concat_max_pool = tf_util.max_pool2d(tr_conv_concat,kernel_size=[1,tau],
                   scope='conv_track_max_pool',
                   stride=[1, 1])
        tr_conv_concat_max_pool = tf.transpose(conv_concat_max_pool,perm=[0,1,3,2])
        '''
        tr_conv_concat_reduce_sum = tf.math.reduce_sum(attention_modulated,axis=3)
        s_conv_concat = tf.squeeze(tr_conv_concat_reduce_sum) + feature_vec
        try:
            end_points['lstm_layer_summary']['temp_attention_wei']=attention_softmax
        except:
            end_points['lstm_layer_summary'] = dict()
            end_points['lstm_layer_summary']['temp_attention_wei']=attention_softmax
        """
    else:
        conv_concat = tf_util.conv2d(tr_exp_concat, 1, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv-track_l2', bn_decay=bn_decay)
        s_conv_concat = tf.squeeze(conv_concat)
        
        
    
    loss_tens = None
    if cos_loss:
        if tau==2:
            loss_tens = tf_cosine_loss(feature_vec,lstm_input1[:,-1,:],axis=1)
        else:
            loss_average = 0
            for ii in range(cos_loss_prop):
                loss_average += tf_cosine_loss(feature_vec,lstm_input1[:,-ii-1,:],axis=1)
            #loss2 = tf_cosine_loss(feature_vec,lstm_input1[:,-2,:],axis=1)
            loss_tens = loss_average/cos_loss_prop
        if lstm_params is not None:
            # This is to use the cosine similarity loss only after some batch training
            offset = lstm_params['global_step'] - lstm_params['cos_loss_batch_thr']
            mult = tf.cond(offset>0,true_fn= lambda: 1.0, false_fn= lambda: 0.0)
            loss_tens = loss_tens * mult
    if end_points is None:
        return [s_conv_concat],[lstm_input1,pf_seq_len],loss_tens
    elif type(end_points) == dict:
        end_points['lstm_layer'] = {}
        end_points['lstm_layer']['outputs'] = s_conv_concat
        end_points['lstm_layer']['feat_input'] = lstm_input1
        end_points['lstm_layer']['pf_seq_len'] = pf_seq_len
        end_points['lstm_layer']['loss'] = loss_tens
        return s_conv_concat
    else:
        return [s_conv_concat],[lstm_input1,pf_seq_len],loss_tens
    
def batch_track_num(feature_dict,wfts):
    '''
    To return number of the same objects in the previous frames (number of tracks inside tau steps backwards)
    
    feature_dict: A dictionary that contains world ids, inside these frame ids, and inside 'feature' and 'track' dictionaries
    wfts        : List of tuples that contain (world_id, frame_id, track_id) of the objects in the batch 
    
    Returns a 1D array showing number of tracks for each entry in batch
    '''
    seq_len = []
    for wft in wfts:
        w,f,t = wft
        tracks = feature_dict[w][f][t]['track']
        seq_count = 0
        for trk in tracks:
            if trk is not None:
                seq_count+=1
        seq_len.append(seq_count)
    return np.array(seq_len)

def tf_cosine_loss(tens1,tens2,axis):
    '''
    tens1 : A tensor with shape (batch x length of feature vector)
    tens2 : A tensor with shape (batch x length of feature vector)
    axis  : Shows the axs of feature vectors. Here mostly 1. 
    
    Returns the reduced loss, reducing the batch losses. 
    '''
    
    pp1 = tf.nn.l2_normalize(tens1,axis=axis)
    pp2 = tf.nn.l2_normalize(tens2,axis=axis)

    lcos = tf.losses.cosine_distance(pp1,pp2,axis=axis)
    return lcos
