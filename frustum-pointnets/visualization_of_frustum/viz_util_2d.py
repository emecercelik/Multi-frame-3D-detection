''' Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017

Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
'''

import numpy as np
import argparse
from PIL import Image, ImageDraw,ImageFont
import os

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3




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
        
        #split_line = line.split(': ')
        split_line = line.split(' ')
        # Calibration files for tracking and object detection datasets are different
        # therefore I need to check if there is a clolon after calib matrix name (P0: or P0)
        if split_line[0][-1]==':':
            calib_type = split_line[0][:-1]
        else:
            calib_type = split_line[0]
        
        values = split_line[1:] 
        #values = split_line[1].split(' ')
        float_values = []
        print(values)
        for val in values:
            print(val)
            if val != '' and val != '\n':
                float_values.append(float(val))
        if calib_type[0] == 'P':
            calib_mat = np.reshape(float_values,(3,4))
        elif calib_type == 'R0_rect':
            calib_mat = np.reshape(float_values,(3,3))
        elif calib_type == 'R_rect':
            calib_type = 'R0_rect'
            calib_mat = np.reshape(float_values,(3,3))
            
        elif calib_type == 'Tr_velo_cam':
            calib_type = 'Tr_velo_to_cam'
            calib_mat = np.reshape(float_values,(3,4))
            
        elif calib_type == 'Tr_imu_velo':
            calib_type = 'Tr_imu_to_velo'
            calib_mat = np.reshape(float_values,(3,4))
        else:
            calib_mat = np.reshape(float_values,(3,4))

        cal_dict[calib_type]=calib_mat
    cal_dict['Tr_cam_to_velo'] = inverse_rigid_transform(cal_dict['Tr_velo_to_cam'])
    cal_dict['inv_R0_rect'] = np.linalg.inv(cal_dict['R0_rect'])
    return cal_dict

def from_point_to_edges(point3D):

    p = point3D
    qs = np.transpose(p)
    edges = []
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        edges.append([qs[i, 0], qs[i, 1], qs[j, 0], qs[j, 1]])
        i, j = k + 4, (k + 1) % 4 + 4
        edges.append([qs[i, 0], qs[i, 1], qs[j, 0], qs[j, 1]])

        i, j = k, k + 4
        edges.append([qs[i, 0], qs[i, 1],qs[j, 0], qs[j, 1]])
    
    
    '''
    edges3D = [ [p[0][0],p[1][0],p[0][1],p[1][1]],[p[0][0],p[1][0],p[0][2],p[1][2]],
                [p[0][0],p[1][0],p[0][4],p[1][4]],[p[0][1],p[1][1],p[0][3],p[1][3]],
                [p[0][1],p[1][1],p[0][5],p[1][5]],[p[0][2],p[1][2],p[0][3],p[1][3]],
                [p[0][2],p[1][2],p[0][6],p[1][6]],[p[0][3],p[1][3],p[0][7],p[1][7]],
                [p[0][4],p[1][4],p[0][5],p[1][5]],[p[0][4],p[1][4],p[0][6],p[1][6]],
                [p[0][5],p[1][5],p[0][7],p[1][7]],[p[0][6],p[1][6],p[0][7],p[1][7]]]
    '''
    return edges

def rect_to_cam(corners,calib):
    '''
    From corners in 3d rectified image coordinates to the 3d boxes on left color image plane 
    
    Returns the pixels of the corners on the image plane
    '''    
    n_col = np.shape(corners)[1]
    ones = np.ones((1,n_col))
    corners_rect_ext = np.vstack((corners,ones))
    corners_2d = np.dot(calib['P2'],corners_rect_ext)
    corners_2d[0,:] /=corners[2,:]
    corners_2d[1,:] /=corners[2,:]
    
    return corners_2d[0:2,:]

def draw_3d_boxes_on_image(multi_corners,draw,color='blue'):
    for corners in multi_corners:
        edges = from_point_to_edges(corners)
        for edge in edges:
            draw.line(edge,fill=color)
   
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualization with Lidar Points')
    parser.add_argument('--label_path',dest = 'label_path', help='Path to the labels of 3d bounding boxes in KITTI format',default=None)
    parser.add_argument('--image_path',dest = 'image_path', help='Path to the image folder',required=True)
    parser.add_argument('--calib_path',dest = 'calib_path', help='Path to calibration txt files',default=None)
    parser.add_argument('--gt_label_path',dest = 'gt_label_path', help='Path to ground-truth labels',default=None)
    parser.add_argument('--image_ind',dest = 'image_ind', help='Index of the image',required=True,type=int)
    parser.add_argument('--fig_name',dest = 'fig_name', help='Name to save the figure',default=None)
    
    args = parser.parse_args()
    img_name =os.path.join(args.image_path,"{:06d}.png".format(args.image_ind))
    img = Image.open(img_name).convert("RGB")
    draw = ImageDraw.Draw(img)
    ## Get calibration
    if args.calib_path is not None:
        try:
            calib_name = os.path.join(args.calib_path,"{:06d}.txt".format(args.image_ind))
            calib_mat = get_calib_dict(calib_name)
        except:
            track_id = os.path.basename(args.image_path)
            calib_name = os.path.join(args.calib_path,"{}.txt".format(track_id))
            calib_mat = get_calib_dict(calib_name)
    
        if args.label_path is not None:
            label_name = os.path.join(args.label_path,"{:06d}.txt".format(args.image_ind)) 
            pred_objects = get_objects(label_name)
            pred_corners = []
            for obj in pred_objects:
                corners_rect = get3dbbox(obj)
                corners_cam = rect_to_cam(corners_rect,calib_mat)
                pred_corners.append(corners_cam)
            draw_3d_boxes_on_image(pred_corners,draw,color='blue')
            print('Pred boxes drawn in blue')
        else:
            print('Prediction labels are not given.')
            
        if args.gt_label_path is not None:
            gt_label_name = os.path.join(args.gt_label_path,"{:06d}.txt".format(args.image_ind))
            gt_objects = get_objects(gt_label_name)
            gt_corners = []
            for obj in gt_objects:
                if obj['class']!='DontCare':
                    corners_rect = get3dbbox(obj)
                    corners_cam = rect_to_cam(corners_rect,calib_mat)
                    gt_corners.append(corners_cam)
            draw_3d_boxes_on_image(gt_corners,draw,color='green')
            print(gt_corners)
            print('Gt boxes drawn in green!')
        else:
            print('Ground-truth label path is not given.')
    else:
        print('Calibration path is not given.')
    
    
    if args.fig_name is not None:
        img.save(args.fig_name,'PNG')
    img.show(title='{:06d}.png'.format(args.image_ind))
    

