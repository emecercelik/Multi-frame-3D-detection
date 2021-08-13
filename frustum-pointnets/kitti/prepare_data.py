''' Prepare KITTI data for 3D object detection.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
try:
    import cloudpickle as pickle
except:
    import pickle
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))

import kitti_util as utils
from kitti_object import *

import vkitti_utils
#from vkitti_object import *


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def get_center_view_rot_angle(frustum_angle):
    ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
    can be directly used to adjust GT heading angle '''
    return np.pi / 2.0 + frustum_angle


def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4, 2))
    box2d_corners[0, :] = [box2d[0], box2d[1]]
    box2d_corners[1, :] = [box2d[2], box2d[1]]
    box2d_corners[2, :] = [box2d[2], box2d[3]]
    box2d_corners[3, :] = [box2d[0], box2d[3]]
    box2d_roi_inds = in_hull(pc[:, 0:2], box2d_corners)
    return pc[box2d_roi_inds, :], box2d_roi_inds

def rotate_pc_along_y(pc, rot_angle):
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

def demo_tracking(dir_tracking_root,drive_id_list=None):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset_tracking = kitti_tracking_object(dir_tracking_root, 'training',drive_list=drive_id_list)
    #dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir))
    data_idx = 10
    drive_idx = 4
    # Load data from dataset
    objects = dataset_tracking.get_label_objects(data_idx,drive_idx)
    objects[0].print_object()
    img = dataset_tracking.get_image(data_idx,drive_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    # pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
    calib =dataset_tracking.get_calibration(data_idx,drive_idx)

    pc_velo = dataset_tracking.get_lidar(data_idx,drive_idx)
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_rect[:, 3] = pc_velo[:, 3]
    img = dataset_tracking.get_image(data_idx,drive_idx)
    img_height, img_width, img_channel = img.shape
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                             calib, 0, 0, img_width, img_height, True)

    object = objects[0]
    xmin, ymin, xmax, ymax = object.box2d

    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                   (pc_image_coord[:, 0] >= xmin) & \
                   (pc_image_coord[:, 1] < ymax) & \
                   (pc_image_coord[:, 1] >= ymin)
    box_fov_inds = box_fov_inds & img_fov_inds
    pc_in_box_fov = pc_rect[box_fov_inds, :]

    # Get frustum angle (according to center pixel in 2D BOX)
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                    box2d_center_rect[0, 0])

    # pc_in_box_fov = rotate_pc_along_y(pc_in_box_fov, get_center_view_rot_angle(frustum_angle))

    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    box3d_pts_3d = rotate_pc_along_y(box3d_pts_3d, get_center_view_rot_angle(frustum_angle))
    box3droi_pc_velo = rotate_pc_along_y(box3droi_pc_velo, get_center_view_rot_angle(frustum_angle))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d], fig=fig)
    mlab.show(1)
    raw_input()

    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d], fig=fig)
    mlab.show(1)
    raw_input()

    # Draw lidar in rect camera coord
    print(' -------- LiDAR points in rect camera coordination --------')
    pc_rect = calib.project_velo_to_rect(pc_velo[:, 0:3])
    fig = draw_lidar_simple(pc_rect)
    raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # show_lidar_with_boxes(pc_velo, objects, calib)
    # raw_input()
    show_lidar_with_boxes(pc_velo[:,0:3], objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)
    raw_input()

    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:, 0:2] = imgfov_pts_2d
    cameraUVDepth[:, 2] = imgfov_pc_rect[:, 2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin, ymin, xmax, ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo[:,0:3], calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()






def demo(kitti_dir):
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir))
    data_idx = 0

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    print(('Image shape: ', img.shape))
    # pc_velo = dataset.get_lidar(data_idx)[:, 0:3]
    calib = dataset.get_calibration(data_idx)

    pc_velo = dataset.get_lidar(data_idx)
    pc_rect = np.zeros_like(pc_velo)
    pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
    pc_rect[:, 3] = pc_velo[:, 3]
    img = dataset.get_image(data_idx)
    img_height, img_width, img_channel = img.shape
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                             calib, 0, 0, img_width, img_height, True)

    object = objects[0]
    xmin, ymin, xmax, ymax = object.box2d

    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                   (pc_image_coord[:, 0] >= xmin) & \
                   (pc_image_coord[:, 1] < ymax) & \
                   (pc_image_coord[:, 1] >= ymin)
    box_fov_inds = box_fov_inds & img_fov_inds
    pc_in_box_fov = pc_rect[box_fov_inds, :]

    # Get frustum angle (according to center pixel in 2D BOX)
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_rect = calib.project_image_to_rect(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                    box2d_center_rect[0, 0])

    # pc_in_box_fov = rotate_pc_along_y(pc_in_box_fov, get_center_view_rot_angle(frustum_angle))

    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    box3d_pts_3d = rotate_pc_along_y(box3d_pts_3d, get_center_view_rot_angle(frustum_angle))
    box3droi_pc_velo = rotate_pc_along_y(box3droi_pc_velo, get_center_view_rot_angle(frustum_angle))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d], fig=fig)
    mlab.show(1)
    raw_input()

    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d], fig=fig)
    mlab.show(1)
    raw_input()

    # Draw lidar in rect camera coord
    print(' -------- LiDAR points in rect camera coordination --------')
    pc_rect = calib.project_velo_to_rect(pc_velo[:, 0:3])
    fig = draw_lidar_simple(pc_rect)
    raw_input()

    # Draw 2d and 3d boxes on image
    print(' -------- 2D/3D bounding boxes in images --------')
    show_image_with_boxes(img, objects, calib)
    raw_input()

    # Show all LiDAR points. Draw 3d box in LiDAR point cloud
    print(' -------- LiDAR points and 3D boxes in velodyne coordinate --------')
    # show_lidar_with_boxes(pc_velo, objects, calib)
    # raw_input()
    show_lidar_with_boxes(pc_velo[:, 0:3], objects, calib, True, img_width, img_height)
    raw_input()

    # Visualize LiDAR points on images
    print(' -------- LiDAR points projected to image plane --------')
    show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)
    raw_input()

    # Show LiDAR points that are in the 3d box
    print(' -------- LiDAR points in a 3D bounding box --------')
    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(objects[0], calib.P)
    box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
    box3droi_pc_velo, _ = extract_pc_in_box3d(pc_velo, box3d_pts_3d_velo)
    print(('Number of points in 3d box: ', box3droi_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(box3droi_pc_velo, fig=fig)
    draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
    mlab.show(1)
    raw_input()

    # UVDepth Image and its backprojection to point clouds
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cameraUVDepth = np.zeros_like(imgfov_pc_rect)
    cameraUVDepth[:, 0:2] = imgfov_pts_2d
    cameraUVDepth[:, 2] = imgfov_pc_rect[:, 2]

    # Show that the points are exactly the same
    backprojected_pc_velo = calib.project_image_to_velo(cameraUVDepth)
    print(imgfov_pc_velo[0:20])
    print(backprojected_pc_velo[0:20])

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(backprojected_pc_velo, fig=fig)
    raw_input()

    # Only display those points that fall into 2d box
    print(' -------- LiDAR points in a frustum from a 2D box --------')
    xmin, ymin, xmax, ymax = \
        objects[0].xmin, objects[0].ymin, objects[0].xmax, objects[0].ymax
    boxfov_pc_velo = \
        get_lidar_in_image_fov(pc_velo[:, 0:3], calib, xmin, ymin, xmax, ymax)
    print(('2d box FOV point num: ', boxfov_pc_velo.shape[0]))

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(boxfov_pc_velo, fig=fig)
    mlab.show(1)
    raw_input()


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    perturbed_xmin = cx2 - w2 / 2.0
    perturbed_ymin = cy2 - h2 / 2.0
    perturbed_xmax = cx2 + w2 / 2.0
    perturbed_ymax = cy2 + h2 / 2.0

    if perturbed_xmin < 0:
        perturbed_xmin = 0.0

    if perturbed_ymin < 0:
        perturbed_ymin = 0.0

    return [perturbed_xmin, perturbed_ymin, perturbed_xmax, perturbed_ymax]


def random_shift_box2d_vkitti(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio

    xmin, ymin, xmax, ymax = box2d

    h = ymax - ymin
    w = xmax - xmin

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)

    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1

    perturbed_xmin = cx2 - w2 / 2.0
    perturbed_ymin = cy2 - h2 / 2.0
    perturbed_xmax = cx2 + w2 / 2.0
    perturbed_ymax = cy2 + h2 / 2.0

    if perturbed_xmin < 0:
        perturbed_xmin = 0.0

    if perturbed_ymin < 0:
        perturbed_ymin = 0.0

    return np.array([perturbed_xmin, perturbed_ymin, perturbed_xmax, perturbed_ymax])


def all_frustum_data(perturb_box2d=False, augmentX=1, type_whitelist=['Car'],kitti_dir='dataset/KITTI/object'):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''

    idx_filename = os.path.join(BASE_DIR, 'image_sets/trainval.txt')
    split = 'training'

    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    pos_cnt = 0
    all_cnt = 0
    ignored = 0
    counter = 0
    for data_idx in data_idx_list:
        # print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):
                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    # print(box2d)
                    # print(xmin, ymin, xmax, ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax - ymin < 25 or np.sum(label) == 0:
                    ignored += 1
                    continue

                counter += 1

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]

    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / counter))
    print('Ignored / All: %d / %d' % (ignored, counter))
    print('Average objects per frame %f' % (counter / float(len(data_idx_list))))

def extract_frustum_data(idx_filename, split, output_filename,
                         perturb_box2d=False, augmentX=1, type_whitelist=['Car'],
                         kitti_dir='dataset/KITTI/object'):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0
    ignored = 0
    #size_pc_byte = 0
    for data_idx in data_idx_list:
        # print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist: continue

            # 2D BOX: Get pts rect backprojected
            box2d = objects[obj_idx].box2d
            for _ in range(augmentX):

                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                    # print(box2d)
                    # print(xmin, ymin, xmax, ymax)
                else:
                    xmin, ymin, xmax, ymax = box2d
                box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                               (pc_image_coord[:, 0] >= xmin) & \
                               (pc_image_coord[:, 1] < ymax) & \
                               (pc_image_coord[:, 1] >= ymin)
                box_fov_inds = box_fov_inds & img_fov_inds
                pc_in_box_fov = pc_rect[box_fov_inds, :]
                # Get frustum angle (according to center pixel in 2D BOX)
                box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                uvdepth = np.zeros((1, 3))
                uvdepth[0, 0:2] = box2d_center
                uvdepth[0, 2] = 20  # some random depth
                box2d_center_rect = calib.project_image_to_rect(uvdepth)
                frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                box2d_center_rect[0, 0])
                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                label = np.zeros((pc_in_box_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if ymax - ymin < 25 or np.sum(label) == 0:
                    ignored += 1
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_box_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)
                #ssize_pc_byte+=sys.getsizeof(pc_in_box_fov)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_box_fov.shape[0]
        #print("Size of size_pc_byte is {:.4f} MB".format(size_pc_byte/1024./1024.))
    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / len(id_list)))
    print('Ignored / All: %d / %d' % (ignored, all_cnt))

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(box3d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)


def extract_frustum_data_tracking(dir_tracking_root,split,output_filename,
                                  perturb_box2d=False, augmentX=1, type_whitelist=['Car'],
                                  drive_id_list=None,other_args=None):
    '''
    To extract frustum data from KITTI tracking dataset
    
    dir_tracking_root: Root directory of KITTI tracking data. Should contain below folders
        data_tracking_image_2/<split>/image_02/
        data_tracking_label_2/<split>/label_02/
        data_tracking_velodyne/<split>/velodyne/
        data_tracking_calib/<split>/calib/
    split            : 'training' or 'test'
    output_filename  : Name of the output .pickle file 
    perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
    augmentX: scalar, how many augmentations to have for each 2D box.
    type_whitelist: a list of strings, object types we are interested in.
    drive_id_list : A list of indices of drives in KITTI tracking dataset that will be used to extract 
        frustum data
        
    
    '''
    dataset_tracking = kitti_tracking_object(dir_tracking_root, split,drive_list=drive_id_list)
    #data_idx_list = [int(line.rstrip()) for line in open(idx_filename)] # indices of frames that will be used either for training or validation


    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    image_id_list = [] # Frame Ind of each frustum
    world_id_list = [] # World Ind of each frustum
    track_id_list = [] # 2D Track ID of each frustum

    pos_cnt = 0
    all_cnt = 0
    ignored = 0

    counter_obj = 0
    #size_pc_byte = 0
    #size_pc_velo = 0
    for drive_ind in dataset_tracking.drive_id_list:
        image_list = dataset_tracking.get_image_idx(drive_ind)
        for image_ind in image_list:
            print('------------- ', drive_ind,image_ind)
            calib = dataset_tracking.get_calibration(image_ind,drive_ind)  # 3 by 4 matrix
            objects = dataset_tracking.get_label_objects(image_ind,drive_ind)
            pc_velo = dataset_tracking.get_lidar(image_ind,drive_ind)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = dataset_tracking.get_image(image_ind,drive_ind)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                     calib, 0, 0, img_width, img_height, True)
            #size_pc_velo+= sys.getsizeof(pc_velo)
            for obj_idx in range(len(objects)):
                if objects[obj_idx].type not in type_whitelist: continue

                # 2D BOX: Get pts rect backprojected
                box2d = objects[obj_idx].box2d
                for _ in range(augmentX):

                    # Augment data by box2d perturbation
                    if perturb_box2d:
                        xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                        # print(box2d)
                        # print(xmin, ymin, xmax, ymax)
                    else:
                        xmin, ymin, xmax, ymax = box2d
                    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                                   (pc_image_coord[:, 0] >= xmin) & \
                                   (pc_image_coord[:, 1] < ymax) & \
                                   (pc_image_coord[:, 1] >= ymin)
                    box_fov_inds = box_fov_inds & img_fov_inds
                    pc_in_box_fov = pc_rect[box_fov_inds, :]
                    # Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                    uvdepth = np.zeros((1, 3))
                    uvdepth[0, 0:2] = box2d_center
                    uvdepth[0, 2] = 20  # some random depth
                    box2d_center_rect = calib.project_image_to_rect(uvdepth)
                    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                    box2d_center_rect[0, 0])
                    # 3D BOX: Get pts velo in 3d box
                    obj = objects[obj_idx]
                    box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                    _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                    label = np.zeros((pc_in_box_fov.shape[0]))
                    label[inds] = 1
                    # Get 3D BOX heading
                    heading_angle = obj.ry
                    # Get 3D BOX size
                    box3d_size = np.array([obj.l, obj.w, obj.h])

                    # Reject too far away object or object without points
                    if ymax - ymin < 25 or np.sum(label) == 0:
                        ignored += 1
                        continue
                    if other_args is not None:
                        if other_args.apply_num_point_thr:
                            if np.sum(label)<other_args.min_num_point_thr or np.sum(label)>other_args.max_num_point_thr:
                                ignored+=1
                                continue

                    id_list.append(counter_obj)
                    box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                    box3d_list.append(box3d_pts_3d)
                    input_list.append(pc_in_box_fov)
                    label_list.append(label)
                    type_list.append(objects[obj_idx].type)
                    heading_list.append(heading_angle)
                    box3d_size_list.append(box3d_size)
                    frustum_angle_list.append(frustum_angle)
                    world_id_list.append(drive_ind)
                    image_id_list.append(image_ind)
                    track_id_list.append(objects[obj_idx].track_id)
                    #size_pc_byte += sys.getsizeof(pc_in_box_fov)
                    # collect statistics
                    pos_cnt += np.sum(label)
                    all_cnt += pc_in_box_fov.shape[0]
                    counter_obj +=1 
                    
            #print(sys.getsizeof(input_list)/1024./1024.,'MB')
            #print("Size of input_list is {:.4f} MB".format(sys.getsizeof(input_list)/1024./1024.))
            #print("Size of id_list is {:.4f} MB".format(sys.getsizeof(id_list)/1024./1024.))
            #print("Size of pc_velo is {:.4f} MB".format(sys.getsizeof(pc_velo)/1024./1024.))
            #print("Size of pc_rect is {:.4f} MB".format(sys.getsizeof(pc_rect)/1024./1024.))
            #print("Size of box2d_list is {:.4f} MB".format(sys.getsizeof(box2d_list)/1024./1024.))
            #print("Size of box3d_list is {:.4f} MB".format(sys.getsizeof(box3d_list)/1024./1024.))
            #print("Size of label_list is {:.4f} MB".format(sys.getsizeof(label_list)/1024./1024.))
            #print("Size of type_list is {:.4f} MB".format(sys.getsizeof(type_list)/1024./1024.))
            #print("Size of heading_list is {:.4f} MB".format(sys.getsizeof(heading_list)/1024./1024.))
            #print("Size of world_id_list is {:.4f} MB".format(sys.getsizeof(world_id_list)/1024./1024.))
            #print("Size of image_id_list is {:.4f} MB".format(sys.getsizeof(image_id_list)/1024./1024.))
            #print("Size of track_id_list is {:.4f} MB".format(sys.getsizeof(track_id_list)/1024./1024.))
            #print("Size of dataset_tracking is {:.4f} MB".format(sys.getsizeof(dataset_tracking)/1024./1024.))
            #print("Size of size_pc_byte is {:.4f} MB".format(size_pc_byte/1024./1024.))
            #print("Size of size_pc_velo is {:.4f} MB".format(size_pc_velo/1024./1024.))
            
        print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
        print('Average npoints: %f' % (float(all_cnt) / len(id_list)))
        print('Ignored / All: %d / %d' % (ignored, all_cnt))
    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(box3d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)
        # Emec added the following three lines
        pickle.dump(image_id_list, fp)
        pickle.dump(world_id_list,fp)
        pickle.dump(track_id_list,fp)

def extract_frustum_data_tracking_rgb_detection(dir_tracking_root,split,output_filename,
                                  perturb_box2d=False, augmentX=1, type_whitelist=['Car'],
                                  drive_id_list=None,det_file=None, img_height_threshold=25,
                                       lidar_point_threshold=5,other_args=None):
    '''
    To extract frustum data from KITTI tracking dataset using the detections from RGB image
    
    dir_tracking_root: Root directory of KITTI tracking data. Should contain below folders
        data_tracking_image_2/<split>/image_02/
        data_tracking_label_2/<split>/label_02/
        data_tracking_velodyne/<split>/velodyne/
        data_tracking_calib/<split>/calib/
    split            : 'training' or 'test'
    output_filename  : Name of the output .pickle file 
    perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
    augmentX: scalar, how many augmentations to have for each 2D box.
    type_whitelist: a list of strings, object types we are interested in.
    drive_id_list : A list of indices of drives in KITTI tracking dataset that will be used to extract 
        frustum data
    det_file      : Path to the RGB detection file. Inside should be <drive_name>/rgb_detection.txt files according to drives stated in drive_id_list
    img_height_threshold: int, neglect image with height lower than that.
    lidar_point_threshold: int, neglect frustum with too few points.
    
    '''
    dataset_tracking = kitti_tracking_object(dir_tracking_root, split,drive_list=drive_id_list)
    #data_idx_list = [int(line.rstrip()) for line in open(idx_filename)] # indices of frames that will be used either for training or validation
    id_list, type_list, box2d_list, prob_list, drive_id_list, track_id_list,\
        label_dict = read_det_file_tracking(det_file,drive_id_list)

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    prob_list = []
    
    
    image_id_list = [] # Frame Ind of each frustum
    world_id_list = [] # World Ind of each frustum
    track_id_list = [] # 2D Track ID of each frustum

    pos_cnt = 0
    all_cnt = 0
    ignored = 0

    counter_obj = 0
    #size_pc_byte = 0
    #size_pc_velo = 0
    for drive_ind in dataset_tracking.drive_id_list:
        image_list = dataset_tracking.get_image_idx(drive_ind)
        for image_ind in image_list:
            print('------------- ', drive_ind,image_ind)
            calib = dataset_tracking.get_calibration(image_ind,drive_ind)  # 3 by 4 matrix
            #objects = dataset_tracking.get_label_objects(image_ind,drive_ind)
            # Object list from RGB detection dictionary: label_dict
            try:
                objects = label_dict[drive_ind][image_ind]
            except:
                print('No predicted objects in drive {}, image {}'.format(drive_ind,image_ind))
                objects = []
            pc_velo = dataset_tracking.get_lidar(image_ind,drive_ind)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = dataset_tracking.get_image(image_ind,drive_ind)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                     calib, 0, 0, img_width, img_height, True)
            #size_pc_velo+= sys.getsizeof(pc_velo)
            for obj_idx in range(len(objects)):
                if objects[obj_idx].type not in type_whitelist: continue

                # 2D BOX: Get pts rect backprojected
                box2d = objects[obj_idx].box2d
                for _ in range(augmentX):

                    # Augment data by box2d perturbation
                    if perturb_box2d:
                        xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                        # print(box2d)
                        # print(xmin, ymin, xmax, ymax)
                    else:
                        xmin, ymin, xmax, ymax = box2d
                    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                                   (pc_image_coord[:, 0] >= xmin) & \
                                   (pc_image_coord[:, 1] < ymax) & \
                                   (pc_image_coord[:, 1] >= ymin)
                    box_fov_inds = box_fov_inds & img_fov_inds
                    pc_in_box_fov = pc_rect[box_fov_inds, :]
                    # Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                    uvdepth = np.zeros((1, 3))
                    uvdepth[0, 0:2] = box2d_center
                    uvdepth[0, 2] = 20  # some random depth
                    box2d_center_rect = calib.project_image_to_rect(uvdepth)
                    frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                                    box2d_center_rect[0, 0])
                    

                    # Pass objects that are too small
                    if ymax - ymin < img_height_threshold or \
                            len(pc_in_box_fov) < lidar_point_threshold:
                        continue

                    id_list.append(counter_obj)
                    type_list.append(objects[obj_idx].type)
                    box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                    prob_list.append(objects[obj_idx].scr)
                    input_list.append(pc_in_box_fov)
                    frustum_angle_list.append(frustum_angle)
                    world_id_list.append(drive_ind)
                    image_id_list.append(image_ind)
                    track_id_list.append(objects[obj_idx].track_id)
                    
                    counter_obj +=1 
                    all_cnt += pc_in_box_fov.shape[0]
                    
            
        print('Average npoints: %f' % (float(all_cnt) / len(id_list)))
        print('Ignored / All: %d / %d' % (ignored, all_cnt))
    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)
        # Emec added the following three lines
        pickle.dump(image_id_list, fp)
        pickle.dump(world_id_list,fp)
        pickle.dump(track_id_list,fp)



def extract_vkitti_frustum_data_rgb_detection(world_ids, variations, output_filename, append=False, perturb_box2d=False,
                                              perturb_box3d=False, lower_perturbation=False, augmentX=1, type_whitelist=['Car'],
                                              img_height_threshold=25, lidar_point_threshold=5, nr_points=1024,dataset_dir='dataset/VKITTI/object'):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    #dataset_dir = '/media/HDD/tuba/dataset/virtual'
    dataset = vkitti_object(dataset_dir)

    id_list = []  # int number
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    type_list = []  # string e.g. Car
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axis
    image_id_list = [] # Frame Ind of each frustum
    world_id_list = [] # World Ind of each frustum
    track_id_list = [] # 2D Track ID of each frustum

    pos_cnt = 0
    all_cnt = 0
    counter = 0
    ignored_objects = 0
    nr_frames = 0
    for world_id in world_ids:
        for variation_id, variation in enumerate(variations):
            print(world_id, ' ', variation)
            calibrations = dataset.get_calibrations(world_id, variation)  # 3 by 4 matrix
            objects = dataset.get_ground_truth_objects(world_id, variation)

            for frame_id in range(dataset.get_nr_frames(world_id, variation)):
                nr_frames += 1

                calibration = calibrations[frame_id]
                depth_map = dataset.get_depth(world_id, variation, frame_id)

                objects_in_frame = []
                if frame_id in objects:
                    objects_in_frame = objects[frame_id]

                for object in objects_in_frame:
                    if object.type not in type_whitelist: continue

                    for _ in range(augmentX):
                        # 2D BOX: Get pts rect backprojected
                        box2d = object.box2d
                        # Augment data by box2d perturbation
                        if lower_perturbation and perturb_box2d:
                            xmin, ymin, xmax, ymax = random_shift_box2d_vkitti(box2d)
                        elif perturb_box2d:
                            xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                        else:
                            xmin, ymin, xmax, ymax = box2d

                        pc_in_box_fov = get_point_cloud_in_frustum(depth_map, calibration, xmin, ymin, xmax, ymax)
                        if pc_in_box_fov.shape[0] == 0: continue

                        choice = np.random.choice(pc_in_box_fov.shape[0], nr_points, replace=True)
                        pc_in_box_fov = pc_in_box_fov[choice, :]

                        # Get frustum angle (according to center pixel in 2D BOX)
                        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
                        uv_depth = np.zeros((1, 3))
                        uv_depth[0, 0:2] = box2d_center
                        uv_depth[0, 2] = 20  # some random depth
                        box2d_center_rect = calibration.project_image_to_camera_with_depth(uv_depth)[0]
                        frustum_angle = -1 * np.arctan2(box2d_center_rect[2], box2d_center_rect[0])

                        _, box3d_pts_3d = vkitti_utils.compute_box_3d(object, calibration.K, perturb_box3d)
                        _, inds = extract_pc_in_box3d(pc_in_box_fov, box3d_pts_3d)
                        label = np.zeros((pc_in_box_fov.shape[0]), dtype=np.int32)
                        label[inds] = 1

                        box3d_size = np.array([object.l, object.w, object.h])

                        # Pass objects that are too small
                        if ymax - ymin < img_height_threshold or np.sum(label) < lidar_point_threshold:
                            ignored_objects += 1
                            continue

                        id_list.append(str(counter))
                        box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
                        box3d_list.append(box3d_pts_3d)
                        input_list.append(pc_in_box_fov)
                        label_list.append(label)
                        type_list.append('Car')
                        heading_list.append(object.ry)
                        box3d_size_list.append(box3d_size)
                        frustum_angle_list.append(frustum_angle)
                        image_id_list.append(object.frame)
                        world_id_list.append(world_id)
                        track_id_list.append(object.tid)


                        pos_cnt += np.sum(label)
                        all_cnt += pc_in_box_fov.shape[0]
                        counter += 1

    print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
    print('Average npoints: %f' % (float(all_cnt) / counter))
    print('Ignored / All: %d / %d' % (ignored_objects, counter))
    print('Average objects per frame %f' % (counter / float(nr_frames)))

    if append:
        with open(output_filename, 'ab') as fp:
            pickle.dump(id_list, fp)
            pickle.dump(box2d_list, fp)
            pickle.dump(box3d_list, fp)
            pickle.dump(input_list, fp)
            pickle.dump(label_list, fp)
            pickle.dump(type_list, fp)
            pickle.dump(heading_list, fp)
            pickle.dump(box3d_size_list, fp)
            pickle.dump(frustum_angle_list, fp)
            # Emec added the following three lines
            pickle.dump(image_id_list, fp)
            pickle.dump(world_id_list,fp)
            pickle.dump(track_id_list,fp)
    else:
        with open(output_filename, 'wb') as fp:
            pickle.dump(id_list, fp)
            pickle.dump(box2d_list, fp)
            pickle.dump(box3d_list, fp)
            pickle.dump(input_list, fp)
            pickle.dump(label_list, fp)
            pickle.dump(type_list, fp)
            pickle.dump(heading_list, fp)
            pickle.dump(box3d_size_list, fp)
            pickle.dump(frustum_angle_list, fp)
            # Emec added the following three lines
            pickle.dump(image_id_list, fp)
            pickle.dump(world_id_list,fp)
            pickle.dump(track_id_list,fp)


def get_box3d_dim_statistics(idx_filename,kitti_dir='dataset/KITTI/object'):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type == 'DontCare': continue
            dimension_list.append(np.array([obj.l, obj.w, obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle', 'wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)


def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3, 7)]))
    return id_list, type_list, box2d_list, prob_list

def read_det_file_tracking(det_filename,drive_list):
    ''' 
    Parse lines in KITTI tracking 2D detection output files
    det_filename : Path to the rgb_detections folder that contain following: <drive_id>/rgb_detection.txt: 0001/rgb_detection.txt, 0011/rgb_detection.txt
    The format of one line:
        path_to_the_image_file/000000.png drive_id track_id class_id det_score x1 y1 x2 y2
        
        Note: Image name (000000.png) is the frame_id at the same time
    
    Returns:
        id_list, type_list, box2d_list, prob_list, drive_id_list, track_id_list, label_dict
        Note: All the lists are in object order. Each list entry is an object. The label_dict,however,
            is oriented in drive_id and frame_id order. Please see below for the dictionary format.
        Format of label_dict:
            label_dict[frame_id][drive_id] contains list of objects
            Each list entry is an object class with the attr. below:
                frame_id [int], drive_id [int], track_id [int], type [str], scr [float], box2d [list of 4 floats: x1,y1,x2,y2]
                
    '''
    class kitti_object_tracking_rgb_det():
        '''Load and parse object data into a usable format.'''

        def __init__(self, frame_id,drive_id,track_id,cls_name,scr,bbox):
            self.frame_id = frame_id
            self.drive_id = drive_id
            self.track_id = track_id
            self.type = cls_name
            self.scr = scr
            self.box2d = bbox
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    id_list = []
    drive_id_list = []
    track_id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    label_dict = {}
    for drive_idx in drive_list:
        rgb_file_name = os.path.join(det_filename,'{:04d}'.format(drive_idx),'rgb_detection.txt')
        for line in open(rgb_file_name, 'r'):
            t = line.rstrip().split(" ")
            # Get information from the line
            frame_id = int(os.path.basename(t[0]).rstrip('.png'))
            drive_id = int(t[1])
            track_id = int(t[2])
            cls_id = det_id2str[int(t[3])]
            scr = float(t[4])
            bbox = np.array([float(t[i]) for i in range(5, 9)])
            # Generate the dictionary to keep information per 
            if drive_id not in label_dict.keys():
                label_dict[drive_id] = {}
            if frame_id not in label_dict[drive_id].keys():
                label_dict[drive_id][frame_id] = []
            
            # Create object specific dictionary
            '''
            obj_dict = {}
            obj_dict['frame_id'] = frame_id
            obj_dict['drive_id'] = drive_id
            obj_dict['track_id'] = track_id
            obj_dict['cls_id'] = cls_id
            obj_dict['scr'] = scr
            obj_dict['bbox'] = bbox
            '''
            #Create object specific class
            obj_dict = kitti_object_tracking_rgb_det(frame_id,drive_id,track_id,cls_id,scr,bbox)
            # Append the object-specific dictionary to the frame object list
            label_dict[drive_id][frame_id].append(obj_dict)
            
            # Create also lists to keep information in the object order 
            id_list.append(frame_id)
            drive_id_list.append(drive_id)
            track_id_list.append(track_id)
            type_list.append(cls_id)
            prob_list.append(scr)
            box2d_list.append(bbox)
    return id_list, type_list, box2d_list, prob_list, drive_id_list, track_id_list, label_dict

def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5,kitti_dir='dataset/KITTI/object'):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = []  # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % \
              (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
            pc_rect[:, 3] = pc_velo[:, 3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov( \
                pc_velo[:, 0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib, pc_rect, pc_image_coord, img_fov_inds]
            cache_id = data_idx
        else:
            calib, pc_rect, pc_image_coord, img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected
        xmin, ymin, xmax, ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])

        # Pass objects that are too small
        if ymax - ymin < img_height_threshold or \
                len(pc_in_box_fov) < lidar_point_threshold:
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

    with open(output_filename, 'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list, fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)


def write_2d_rgb_detection(det_filename, split, result_dir,kitti_dir='dataset/KITTI/object'):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format.

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, kitti_dir), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--all', action='store_true', help='Process all to get stats.')
    parser.add_argument('--all_train', action='store_true', help='Process all to get stats.')
    parser.add_argument('--gen_train', action='store_true',
                        help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--vkitti_all', action='store_true', help='Generate vkitti data')
    parser.add_argument('--gen_vkitti_train', action='store_true', help='Generate vkitti data')
    parser.add_argument('--gen_vkitti_val', action='store_true', help='Generate vkitti data')
    parser.add_argument('--ground_vkitti_train', action='store_true', help='Generate vkitti train data')
    parser.add_argument('--ground_vkitti_val', action='store_true', help='Generate vkitti val data')
    parser.add_argument('--gen_val_rgb_detection', action='store_true',
                        help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    parser.add_argument('--perturb2d', action='store_true', help='To apply 2D perturbation')
    parser.add_argument('--perturb3d', action='store_true', help='To apply 3D perturbation')
    parser.add_argument('--video_train', metavar='N', type=int, nargs='+',help='Indices of vkitti video to be used for train data')
    parser.add_argument('--video_val', metavar='N', type=int, nargs='+',help='Indices of vkitti video to be used for val data')
    parser.add_argument('--name_train', default='', help='an extension to the name of saved training data file')
    parser.add_argument('--name_val', default='', help='an extension to the name of saved validation data file')
    parser.add_argument('--num_point', type=int, default=1024, help='Number of points per frustum [default: 1024]')
    parser.add_argument('--vkitti_path',default=None,help='path to the vkitti dataset')
    parser.add_argument('--kitti_path', default = 'dataset/KITTI/object', help='Path to the KITTI dataset. Inside should be training/-calib,-image_2,-label_2,-velodyne')
    parser.add_argument('--gen_tracking_train', action='store_true', help='Generate KITTI Tracking data for training')
    parser.add_argument('--gen_tracking_val', action='store_true', help='Generate KITTI Tracking data for validation')
    parser.add_argument('--tracking_path', default = '', help='Path to the KITTI tracking dataset. Inside should be data_tracking_<name>/<split>/<name>. <name>:calib,image_02,label_02,velodyne. <split>:training,testing')
    parser.add_argument('--rgb_detection_path', default = '', help='Path to the KITTI tracking rgb_detections. Inside should be <drive_name>/rgb_detection.txt files according to drives stated in video_val')
    parser.add_argument('--apply_num_point_thr', action='store_true', help='To exclude frustums that have less than minimum and more than maximum number of points given with the flags')
    parser.add_argument('--max_num_point_thr', type=int, default=8196, help='Maximum number of points per frustum if apply_num_point_thr is set')
    parser.add_argument('--min_num_point_thr', type=int, default=1, help='Minimum number of points per frustum if apply_num_point_thr is set')
    parser.add_argument('--augmentX', type=int, default=1, help='Shows how many times a 2D box will be augmented')
    args = parser.parse_args()
    # vkitti and kitti tracking consumes in different formats
    if args.video_train is not None:
        if args.gen_tracking_train:
            args.video_train = [int(i) for i in args.video_train]
        else:
            args.video_train = ['{:04d}'.format(i) for i in args.video_train]
    if args.video_val is not None:
        if args.gen_tracking_val:
            args.video_val = [int(i) for i in args.video_val]
        else:
            args.video_val = ['{:04d}'.format(i) for i in args.video_val]
    
    if args.demo and not args.gen_tracking_train:
        demo(args.kitti_path)
        exit()
        
    if args.demo and args.gen_tracking_train:
        demo_tracking(args.tracking_path,drive_id_list=args.video_train)
        exit()
    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.all:
        all_frustum_data(perturb_box2d=False, augmentX=1, type_whitelist=type_whitelist,kitti_dir=args.kitti_path)

    if args.gen_train:
        extract_frustum_data(os.path.join(BASE_DIR, 'image_sets/train.txt'), 'training',
                             os.path.join(BASE_DIR, output_prefix + 'train.pickle'), perturb_box2d=args.perturb2d, augmentX=1,
                             type_whitelist=type_whitelist,kitti_dir=args.kitti_path)

    if args.gen_val:
        extract_frustum_data(
            os.path.join(BASE_DIR, 'image_sets/val.txt'), 'training',
            os.path.join(BASE_DIR, output_prefix + 'val.pickle'), perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist,kitti_dir=args.kitti_path)

    if args.gen_val_rgb_detection and not args.gen_tracking_val:
        extract_frustum_data_rgb_detection(
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'), 'training',
            os.path.join(BASE_DIR, output_prefix + 'val_rgb_detection.pickle'), type_whitelist=type_whitelist,kitti_dir=args.kitti_path)

    if args.all_train:
        filename = os.path.join(BASE_DIR, output_prefix + 'train.pickle')
        extract_frustum_data(os.path.join(BASE_DIR, 'image_sets/train.txt'), 'training',
                             filename, perturb_box2d=True, augmentX=3,
                             type_whitelist=type_whitelist,kitti_dir=args.kitti_path)

        extract_vkitti_frustum_data_rgb_detection(world_ids=['0001', '0006', '0020', '0002', '0018'],
                                                  variations=['15-deg-left', '30-deg-left', 'clone', 'morning', 'rain',
                                                              '15-deg-right', '30-deg-right', 'fog',
                                                              'overcast', 'sunset'],
                                                  append=True,
                                                  output_filename=filename,
                                                  perturb_box2d=True,
                                                  augmentX=3,dataset_dir=args.vkitti_path)

    variations = ['15-deg-left', '30-deg-left', 'clone', 'morning', 'rain', '15-deg-right', '30-deg-right', 'fog',
                  'overcast', 'sunset']
    # variations = ['clone']

    if args.vkitti_all:
        
        extract_vkitti_frustum_data_rgb_detection(world_ids=['0001', '0006', '0020', '0002', '0018'],
                                                  variations=variations,
                                                  output_filename=os.path.join(BASE_DIR,
                                                                               'frustum_caronly_' + 'vkitti_train.pickle'),
                                                  perturb_box2d=args.perturb2d, perturb_box3d=args.perturb3d,
                                                  augmentX=3,dataset_dir=args.vkitti_path)

    if args.ground_vkitti_train:
        # world_ids=['0001', '0006', '0020'] , perturb2d False, perturb3d False
        extract_vkitti_frustum_data_rgb_detection(world_ids=args.video_train, variations=variations,
                                                  output_filename=os.path.join(BASE_DIR,
                                                                               'ground_caronly_' + 'vkitti_train.pickle'),
                                                  perturb_box2d=args.perturb2d, perturb_box3d=args.perturb3d,
                                                  augmentX=1,dataset_dir=args.vkitti_path)

    if args.ground_vkitti_val:
        extract_vkitti_frustum_data_rgb_detection(world_ids=args.video_val, variations=variations,
                                                  output_filename=os.path.join(BASE_DIR,
                                                                               'ground_caronly_' + 'vkitti_val.pickle'),
                                                  perturb_box2d=False, perturb_box3d=False,
                                                  augmentX=1,dataset_dir=args.vkitti_path)

    if args.gen_vkitti_train:
        #print(os.path.join(BASE_DIR,'frustum_caronly_2D3D' + 'vkitti_train.pickle'))
        #world_ids=['0001', '0006', '0020'] perturb2d True perturb3d True
        extract_vkitti_frustum_data_rgb_detection(world_ids=args.video_train, variations=variations,
                                                  output_filename=os.path.join(BASE_DIR,
                                                                               'frustum_caronly_' + 'vkitti_train' + args.name_train +'.pickle'),
                                                  perturb_box2d=args.perturb2d, perturb_box3d=args.perturb3d,
                                                  augmentX=1, nr_points=args.num_point,dataset_dir=args.vkitti_path)

    if args.gen_vkitti_val:
        #world_ids=['0002', '0018'], perturb2d False, perturb3d False
        extract_vkitti_frustum_data_rgb_detection(world_ids=args.video_val, variations=variations,
                                                  output_filename=os.path.join(BASE_DIR,
                                                                               'frustum_caronly_' + 'vkitti_val' + args.name_val + '.pickle'),
                                                  perturb_box2d=False, perturb_box3d=False,
                                                  augmentX=1, nr_points=args.num_point,dataset_dir=args.vkitti_path)
    if args.gen_tracking_train:
        extract_frustum_data_tracking(dir_tracking_root=args.tracking_path,split='training',
                              output_filename=os.path.join(BASE_DIR,output_prefix+'tracking'+args.name_train+'_train'+'.pickle'),
                              perturb_box2d=args.perturb2d, augmentX=args.augmentX,
                              type_whitelist=type_whitelist,drive_id_list=args.video_train,other_args=args)
    if args.gen_tracking_val and not args.gen_val_rgb_detection:
        extract_frustum_data_tracking(dir_tracking_root=args.tracking_path,split='training',
                              output_filename=os.path.join(BASE_DIR,output_prefix+'tracking'+args.name_val+'_val'+'.pickle'),
                              perturb_box2d=False, augmentX=1,
                              type_whitelist=type_whitelist,drive_id_list=args.video_val,other_args=args)
    
    if args.gen_tracking_val and args.gen_val_rgb_detection:
        extract_frustum_data_tracking_rgb_detection(dir_tracking_root=args.tracking_path,split='training',
                              output_filename=os.path.join(BASE_DIR, output_prefix + 'tracking'+args.name_val+'_rgb_detection'+'.pickle'),
                              perturb_box2d=False, augmentX=1,
                              type_whitelist=type_whitelist,drive_id_list=args.video_val,
                              det_file=args.rgb_detection_path,other_args=args)
        
    
