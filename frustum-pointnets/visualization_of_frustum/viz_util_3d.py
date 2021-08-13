''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''

import numpy as np
import mayavi.mlab as mlab
import argparse

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)
    
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov=np.array([  # 45 degree
        [20., 20., 0.,0.],
        [20.,-20., 0.,0.],
    ],dtype=np.float64)
    
    mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
   
    # draw square region
    TOP_Y_MIN=-20
    TOP_Y_MAX=20
    TOP_X_MIN=0
    TOP_X_MAX=40
    TOP_Z_MIN=-2.0
    TOP_Z_MAX=0.4
    
    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=0.1, line_width=1, figure=fig)
    
    #mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def read_label(addr):
    """
    Reads the labels (objects) line by line. Returns the lines that are read 
    from the file with the index as a string.
    
    Opens the label file indicated by the ind(index) and the label address. Reads all 
    the lines. Extracts index as a text from the address.        
    """
    f = open(addr,"r")
    lines = f.readlines() 
    f.close()
    return lines

def get_objects(label_addr):
    """
    ind : the index of the image (or label) that the objects will be read from
    filter_out_classes: To filter out the objects that belong to the given classes
    
    Extracts object ground truth that are read from the label file indicated with the ind (index).
    Creates a kitti object and assigns all the features as object features.
    
    Loops through the lines and splites the string of each line with " " (space). Casts the splitted 
    data into an appropriate type. Sets a kitti object for each line.
    
    Returns the list of Kitti objects.
    """
    lines = read_label(label_addr)
    object_list = []
    for line in lines:
        try:      
            cl_n,tr,occ,alp,x1,y1,x2,y2,h,w,l,x,y,z,rot =  line.split(" ")
        except:
            cl_n,tr,occ,alp,x1,y1,x2,y2,h,w,l,x,y,z,rot,_ =  line.split(" ")
            
        obj_dict ={}
        obj_dict['class'] = cl_n
        obj_dict['truncation'] = float(tr)
        obj_dict['occlusion'] = int(occ)
        obj_dict['alpha'] = float(alp)
        obj_dict['x1'] = float(x1)
        obj_dict['y1'] = float(y1)
        obj_dict['x2'] = float(x2)
        obj_dict['y2'] = float(y2)
        obj_dict['h'] = float(h)
        obj_dict['w'] = float(w)
        obj_dict['l'] = float(l)
        obj_dict['x'] = float(x)
        obj_dict['y'] = float(y)
        obj_dict['z'] = float(z)
        obj_dict['rotation_y'] = float(rot)

        object_list.append(obj_dict)
    return object_list

def get3dbbox(obj_dict):
    '''
    Dict should contain h,w,l,x,y,z, rotation_y keys
    
    Returns corners with 3x8 dims
    '''
    l = obj_dict['l']
    h = obj_dict['h']
    w = obj_dict['w']
    ry = obj_dict['rotation_y']
    x = obj_dict['x']
    y = obj_dict['y']
    z = obj_dict['z']
    
    corners = np.array([[l/2., l/2., -l/2., -l/2., l/2., l/2., -l/2., -l/2.],
                        [0, 0, 0, 0, -h, -h, -h, -h],
                        [w/2., -w/2., -w/2., w/2., w/2., -w/2., -w/2., w/2.]])
    
    R = np.array([[np.cos(ry),0,np.sin(ry)],
                  [0,1,0],
                  [-np.sin(ry),0,np.cos(ry)]])
    
    corners_new = np.dot(R,corners)+np.array([[x],[y],[z]])
    
    #corners_new = [corners_new[2],corners_new[0],corners_new[1]]
    return corners_new

def rect_to_velo(corners,calib):
    
    corners_3d_ref = np.dot(calib['inv_R0_rect'],corners)
    n_col = np.shape(corners_3d_ref)[1]
    ones = np.ones((1,n_col))
    corners_3d_ref_ext = np.vstack((corners_3d_ref,ones))
    corners_velo = np.dot(calib['Tr_cam_to_velo'],corners_3d_ref_ext)
    return np.transpose(corners_velo)


def get_calib_dict(calib_file_path):
    def inverse_rigid_transform(Tr_velo_to_cam):
        inv_mat = np.zeros_like(Tr_velo_to_cam)
        inv_mat[0:3,0:3] = np.transpose(Tr_velo_to_cam[0:3,0:3])
        inv_mat[0:3,3] = np.dot(-np.transpose(Tr_velo_to_cam[0:3,0:3]),Tr_velo_to_cam[0:3,3])
        return inv_mat
    f = open(calib_file_path,"r")
    lines = f.readlines() 
    cal_dict={}
    for line in lines[:-1]:
        split_line = line.split(': ')
        values = split_line[1].split(' ')
        asd = []

        for val in values:
            asd.append(float(val))
        if split_line[0][0] == 'P':
            calib_mat = np.reshape(asd,(3,4))
        elif split_line[0] == 'R0_rect':
            calib_mat = np.reshape(asd,(3,3))
        else:
            calib_mat = np.reshape(asd,(3,4))

        cal_dict[split_line[0]]=calib_mat
    cal_dict['Tr_cam_to_velo'] = inverse_rigid_transform(cal_dict['Tr_velo_to_cam'])
    cal_dict['inv_R0_rect'] = np.linalg.inv(cal_dict['R0_rect'])
    return cal_dict
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualization with Lidar Points')
    parser.add_argument('--label_path',dest = 'label_path', help='Path to the labels of 3d bounding boxes in KITTI format',default=None)
    parser.add_argument('--lidar_path',dest = 'lidar_path', help='Path to the lidar point clouds',required=True)
    parser.add_argument('--calib_path',dest = 'calib_path', help='Path to calibration txt files',default=None)
    parser.add_argument('--gt_label_path',dest = 'gt_label_path', help='Path to ground-truth labels',default=None)
    parser.add_argument('--lidar_ind',dest = 'lidar_ind', help='Index of the lidar',required=True,type=int)
    parser.add_argument('--fig_name',dest = 'fig_name', help='Name to save the figure',default=None)
    
    args = parser.parse_args()
    
    ## Lidar point cloud 
    lidar_path = args.lidar_path+"{:06d}.bin".format(args.lidar_ind)
    lidar_points = np.fromfile(lidar_path,dtype=np.float32)
    lidar_points = lidar_points.reshape((-1,4))
    pc = lidar_points[:,0:3] # 3d lidar points excluding reflectance
    fig = draw_lidar(pc)
    ## Get calibration
    if args.calib_path is not None:
        calib_mat = get_calib_dict(args.calib_path+"{:06d}.txt".format(args.lidar_ind))
    
        if args.label_path is not None:
            pred_objects = get_objects(args.label_path+"{:06d}.txt".format(args.lidar_ind))
            pred_corners = []
            #pred_color = (1, 0, 0)
            for obj in pred_objects:
                corners_rect = get3dbbox(obj)
                corners_velo = rect_to_velo(corners_rect,calib_mat)
                pred_corners.append(corners_velo)
            draw_gt_boxes3d(pred_corners,fig)
            print('Pred boxes drawn in white')
        else:
            print('Prediction labels are not given.')
            
        if args.gt_label_path is not None:
            gt_objects = get_objects(args.gt_label_path+"{:06d}.txt".format(args.lidar_ind))
            gt_corners = []
            gt_color = (0, 1, 0)
            for obj in gt_objects:
                if obj['class']!='DontCare':
                    corners_rect = get3dbbox(obj)
                    corners_velo = rect_to_velo(corners_rect,calib_mat)
                    gt_corners.append(corners_velo)
            draw_gt_boxes3d(gt_corners,fig,color=gt_color)
            print(gt_corners)
            print('Gt boxes drawn in green!')
        else:
            print('Ground-truth label path is not given.')
    else:
        print('Calibration path is not given.')
    
    
    
   
    
    
    
    
    
    #pc = np.loadtxt('mayavi/kitti_sample_scan.txt')
    
    if args.fig_name is not None:
        mlab.savefig(args.fig_name, figure=fig)
    raw_input()
    

