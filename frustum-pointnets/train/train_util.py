''' Util functions for training and evaluation.

Author: Charles R. Qi
Date: September 2017
'''

import numpy as np
from model_util import g_mean_size_arr
from model_util import NUM_HEADING_BIN, CAR_OBJECT_CLASS, PADDING_OBJECT_CLASS
from box_util import box3d_iou

NR_CLASSES_FOR_KITTI_ONEHOT = 3
NR_CLASSES_FOR_VKITTI_ONEHOT = 2

import IPython
def rotate_pc_along_y_by_angle(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def rotate_pc_along_y(pc, rot_angles):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    for i, angle in enumerate(rot_angles):
        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotmat = np.array([[cosval, -sinval], [sinval, cosval]])

        pc[i, :, 0:2] = np.dot(pc[i, :, 0:2], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)

    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)

    class_id = (shifted_angle / angle_per_class).astype(int, copy=False)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)

    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual

    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi

    return angle


def size2class(size, object_types):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = object_types
    mean_sizes = g_mean_size_arr[object_types]
    size_residual = size - np.asarray(mean_sizes)
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_mean_size_arr[pred_cls]
    return mean_size + residual


def get_center_view_rot_angle(frustum_angle):
    ''' Get the frustum rotation angle, it is shifted by pi/2 so that it
    can be directly used to adjust GT heading angle '''
    return np.pi / 2.0 + frustum_angle


def get_box3d_center(batch_box_3d):
    ''' Get the center (XYZ) of 3D bounding box. '''
    box3d_center = (batch_box_3d[:, 0, :] + batch_box_3d[:, 6, :]) / 2.0
    return box3d_center


def get_center_view_box3d_center(box3d_center, rot_angle):
    ''' Frustum rotation of 3D bounding box center. '''
    return rotate_pc_along_y(np.expand_dims(box3d_center, 1), rot_angle).squeeze()


def get_center_view_point_set(ps, center_view_rot_angle):
    ''' Frustum rotation of point clouds.
    NxC points with first 3 channels as XYZ
    z is facing forward, x is left ward, y is downward
    '''
    point_set = np.copy(ps)
    return rotate_pc_along_y(point_set, center_view_rot_angle)


def sparse_tensor_to_dense_array(sparse_tensor):
    dense_array = np.zeros(sparse_tensor.dense_shape)
    dense_array[sparse_tensor.indices[:, 0], sparse_tensor.indices[:, 1]] = sparse_tensor.values
    return dense_array


def parse_batch(dataset, idxs, start_idx, end_idx, num_point, num_channel, from_rgb_detection=False):
    if from_rgb_detection:
        return parse_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel)

    bsize = end_idx - start_idx

    batch_identifier = []
    batch_frame_id = np.zeros((bsize,))
    batch_object_id = np.zeros((bsize,))
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize, NR_CLASSES_FOR_KITTI_ONEHOT))  # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            identifier, frame_id, object_id, ps, seg, center, hclass, hres, sclass, sres, rotangle, onehotvec = dataset[idxs[i + start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            identifier, frame_id, object_id, ps, seg, center, hclass, hres, sclass, sres, rotangle = dataset[idxs[i + start_idx]]

        batch_identifier.append(identifier)
        batch_frame_id[i] = frame_id
        batch_object_id[i] = object_id
        batch_data[i, ...] = ps[:, 0:num_channel]
        batch_label[i, :] = seg
        batch_center[i, :] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
    if dataset.one_hot:
        return batch_identifier, batch_frame_id, batch_object_id, batch_data, batch_label, batch_center, batch_heading_class, batch_heading_residual, batch_size_class, batch_size_residual, batch_rot_angle, batch_one_hot_vec
    else:
        return batch_identifier, batch_frame_id, batch_object_id, batch_data, batch_label, batch_center, batch_heading_class, batch_heading_residual, batch_size_class, batch_size_residual, batch_rot_angle


def parse_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel):
    # world_ids = batch['world_id']
    # variation_ids = batch['variation_id']
    # frame_ids = batch['frame_id']
    # object_ids = batch['object_id']
    # batch_id_strings = []
    # for i, world_id in enumerate(world_ids):
    #     batch_id_strings.append(str(world_id) + '-' + str(variation_ids[i]) + '-' + str(frame_ids[i]) + '-' + str(object_ids[i]))

    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize, NR_CLASSES_FOR_KITTI_ONEHOT))  # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps, rotangle, prob, onehotvec = dataset[idxs[i + start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps, rotangle, prob = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps[:, 0:num_channel]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec
    else:
        return batch_data, batch_rot_angle, batch_prob


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box corners
    '''
    R = roty(heading_angle)
    l, w, h = box_size

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)

    return corners_3d


def compute_box3d_iou(center_pred, heading_logits, heading_residuals, size_logits, size_residuals,
                      center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)

    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)


def compute_box3d_iou_with_padding(center_pred, heading_logits, heading_residuals, size_logits, size_residuals,
                      actual_nr_objects, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(actual_nr_objects):
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)

    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)

def from_prediction_to_label_format(center, angle_class, angle_res, size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y_by_angle(np.expand_dims(center, 0), -rot_angle).squeeze()
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


def get_batch(dataset, idxs, start_idx, end_idx, num_point, num_channel, num_classes=3, from_rgb_detection=False,tracks=False):
    ''' Prepare batch data for training/evaluation.
    batch size is determined by start_idx-end_idx

    Input:
        dataset: an instance of FrustumDataset class
        idxs: a list of data element indices
        start_idx: int scalar, start position in idxs
        end_idx: int scalar, end position in idxs
        num_point: int scalar
        num_channel: int scalar
        from_rgb_detection: bool
        tracks: bool : to return world,frame,track ids of objects in the batch
    Output:
        batched data and label
    '''
    num_point = dataset.npoints
    if from_rgb_detection:
        return get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx,
                                            num_point, num_channel,  num_classes,tracks=tracks)

    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_label = np.zeros((bsize, num_point), dtype=np.int32)
    batch_center = np.zeros((bsize, 3))
    batch_heading_class = np.zeros((bsize,), dtype=np.int32)
    batch_heading_residual = np.zeros((bsize,))
    batch_size_class = np.zeros((bsize,), dtype=np.int32)
    batch_size_residual = np.zeros((bsize, 3))
    batch_rot_angle = np.zeros((bsize,))
    # Emec added this line here
    batch_indices=[]
    
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize, num_classes))  # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps, seg, center, hclass, hres, sclass, sres, rotangle, onehotvec = \
                dataset[idxs[i + start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps, seg, center, hclass, hres, sclass, sres, rotangle = \
                dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps[:, 0:num_channel]
        batch_label[i, :] = seg
        batch_center[i, :] = center
        batch_heading_class[i] = hclass
        batch_heading_residual[i] = hres
        batch_size_class[i] = sclass
        batch_size_residual[i] = sres
        batch_rot_angle[i] = rotangle
        if tracks:
            idx_obj = idxs[i + start_idx]
            batch_indices.append((dataset.w_l[idx_obj],dataset.f_l[idx_obj],dataset.t_l[idx_obj]))
    if dataset.one_hot:
        return batch_data, batch_label, batch_center, \
               batch_heading_class, batch_heading_residual, \
               batch_size_class, batch_size_residual, \
               batch_rot_angle, batch_one_hot_vec,batch_indices
    else:
        return batch_data, batch_label, batch_center, \
               batch_heading_class, batch_heading_residual, \
               batch_size_class, batch_size_residual, batch_rot_angle,batch_indices


def get_batch_from_rgb_detection(dataset, idxs, start_idx, end_idx, num_point, num_channel, num_classes,tracks=False):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, num_point, num_channel))
    batch_rot_angle = np.zeros((bsize,))
    batch_prob = np.zeros((bsize,))
    batch_indices=[]
    if dataset.one_hot:
        batch_one_hot_vec = np.zeros((bsize, num_classes))  # for car,ped,cyc
    for i in range(bsize):
        if dataset.one_hot:
            ps, rotangle, prob, onehotvec = dataset[idxs[i + start_idx]]
            batch_one_hot_vec[i] = onehotvec
        else:
            ps, rotangle, prob = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps[:, 0:num_channel]
        batch_rot_angle[i] = rotangle
        batch_prob[i] = prob
        if tracks:
            idx_obj = idxs[i + start_idx]
            batch_indices.append((dataset.w_l[idx_obj],dataset.f_l[idx_obj],dataset.t_l[idx_obj]))
    if dataset.one_hot:
        return batch_data, batch_rot_angle, batch_prob, batch_one_hot_vec, batch_indices
    else:
        return batch_data, batch_rot_angle, batch_prob, batch_indices
