import tensorflow as tf
import argparse
import numpy as np
import glob
import os
from PIL import Image
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import np_box_list,np_box_list_ops

from misc import filter_threshold
import IPython

def write_predictions_to_file_in_kitti_format(pred_folder,gt_folder,pred_scr_thr,pred_boxes,pred_scores,\
                                              pred_cls,gt_boxes,gt_cls,img_name=None,with_score=False,\
                                              category_index=None):
    """
    This method is to write predictions of the detection network into text files.
    This methods creates a folder for predictions and ground_truths. Objects of each image are written in a txt file whose name is the ID of the image. 
    The predictions are added into the prediction file only if the prediction score is above the pred_scr_thr value.
    Each entry is separated by a space (" ") and each line of a text file corresponds to a predicted object.
    
    pred_folder : name and path of the folder that the predicted objects will be written into. Ex: "../Object-Detection-Metrics/p_rcnn2_v23p0_0.6"
    gt_folder   : name and path of the folder that the ground-truth objects will be written into. Ex: "../Object-Detection-Metrics/gt_rcnn2_v23p0_0.6"
    pred_scr_thr: A floating point number threshold or a list of float thresholds. Only objects with scores above this threshold will be written as predictions. If this is a list, then the score threshold will be called with the index of predicted class.
    pred_boxes  : Predicted bounding boxes in the format of (n,4) list. n is the number of predicted objects. 
    pred_scores : Prediction scores of objects in the format of a list of n entries. 
    pred_cls    : Class indicator (scalar, float or int) of predicted objects as a list. Cls 0: Car, cls 1: Pedestrian, cls 2: Cyclist
    gt_boxes    : Ground-truth bounding boxes in the format of (n,4) list. 
    gt_cls      : Class indicator of ground-truth bounding boxes, a list of n scalar entries (float or int). 
    img_name    : Name of the image file (must be a number). Usually the index of image. if not provided, the default of the prediction text files will be "1.txt". The given number of string will be casted into integer. Ex: "1" or 1. Output: 000001.txt
    with_score: If true, the last entry of each line is the score. Otherwise, score is not written in the txt file.
    category_index : This is a dictionary that keeps name of classes for every index used in detection. Format: {1: {'id': 1, 'name': 'car'}, 2: {'id': 2, 'name': 'pedestrian'}}. If this is None, the format below will be used instead.
        Cls ind 1: "Car", cls ind 2: "Pedestrian", cls ind 3: "Cyclist"
    
    
    """
    
    ## Create a folder to write the predictions
    if os.path.isdir(pred_folder):
        print("Prediction folder exists!")
    else:
        os.makedirs(pred_folder)
    
    ## Create a folder to write the ground-truths
    if gt_folder is not None:
        if os.path.isdir(gt_folder):
            print("Ground-truth folder exists!")
        else:
            os.makedirs(gt_folder)
    
    ## Determine the file name
    if img_name is None:
        img_name='1.txt'
    else:
        img_name = '{:06d}'.format(int(img_name)) + '.txt'
    
    with open(pred_folder+'/'+img_name,'w') as file_name:
        for bbox,cls,pred_scr in zip(pred_boxes,pred_cls,pred_scores):
            if category_index is not None:
                cls = int(cls)
                cls_name = category_index[cls]['name'].capitalize()
            else:
                cls = int(cls)
                if cls == 1:
                    cls_name = "Car"
                elif cls == 2:
                    cls_name = "Pedestrian"
                elif cls == 3:
                    cls_name = "Cyclist"
                else:
                    cls_name = "unknown"
            if type(pred_scr_thr)==float:
                scr_thr = pred_scr_thr
            else:
                scr_thr = pred_scr_thr[cls]
            if pred_scr>=scr_thr:
                ## Write in the order below
                ## Type (1), truncation (1), occlusion(1), alpha (1), bbox (4), dimensions (3), location(3), rotation_y(1), score (1: optional) 
                ## Values other than type, bbox and score are dummy
                file_name.write(cls_name+' '+str(0)+' '+str(0)+' '+str(0)+' ')
                for box in bbox:
                    file_name.write(str(box)+' ')
                file_name.write(str(0)+' '+str(0)+' '+str(0)+' '+str(0)+' '+str(0)+' '+str(0)+' '+str(0))
                ## This adds the prediction score to the end
                if with_score:
                    file_name.write(' '+str(pred_scr))
                file_name.write('\n')
    if gt_folder is not None:
        with open(gt_folder+'/'+img_name,'w') as file_name:
            for gt_box,cls in zip(gt_boxes,gt_cls):
                if cls == 3:
                    cls = 1
                elif cls == 5:
                    cls = 2

                file_name.write(str(cls)+' ')
                for box in gt_box:
                    file_name.write(str(box)+' ')
                file_name.write('\n')


def run_inference_for_single_image(image, sess):
    #with graph.as_default():
    #    cfg = tf.ConfigProto() 
    #    cfg.gpu_options.allow_growth = True
    #    cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
    #    with tf.Session(config=cfg) as sess:
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def load_image_into_numpy_array(image):
    '''
    To load image into a numpy array 
    '''
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, -1)).astype(np.uint8)[:,:,0:3]

def inference(args):
    ## Load the frozen Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    ## Load label map 
    category_index = label_map_util.create_category_index_from_labelmap(args.path_to_labels, use_display_name=True)
    
    ## Image paths
    image_paths = glob.glob(os.path.join(args.path_to_images,'*.{}'.format(args.image_format)))
    image_paths.sort()
    
    ## Check if the path exists
    try:
        os.makedirs(args.output_path)
    except Exception as e:
        print(e)
    
    results_arr = []
    image_arr = []
    with detection_graph.as_default():
        cfg = tf.ConfigProto() 
        cfg.gpu_options.allow_growth = True
        cfg.gpu_options.per_process_gpu_memory_fraction = 0.9
        ### augmentation graph
        resize_scales = [0.5,0.7,2.0,4.0]
        augment_input = tf.placeholder(dtype=tf.uint8,shape=(None,None,None,3))
        shape_ph = tf.placeholder(dtype=tf.float32,shape=(2))
        resized_tensors = []
        resized_shapes = []
        for scale in resize_scales:
            resized_shapes.append([tf.cast(shape_ph[0]*scale,tf.int32),tf.cast(shape_ph[1]*scale,tf.int32)])
            resized_tensors.append(tf.image.resize(augment_input,\
                                                   resized_shapes[-1]))
        flipped_tensor = tf.image.flip_left_right(augment_input)
        bright_tensor = tf.image.random_brightness(augment_input,0.1)
        #aug_image_tensors = [flipped_tensor,bright_tensor]+resized_tensors
        #aug_type=['flip','brightness']+resize_scales+['original']
        aug_image_tensors = resized_tensors
        aug_type=resize_scales+['original']

        with tf.Session(config=cfg) as sess:
            for image_i,image_path in enumerate(image_paths):
                print('**** Inference {}/{}'.format(image_i,len(image_paths)))
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Get augmented images
                augmented_np_images = sess.run(aug_image_tensors,\
                                        feed_dict={augment_input:image_np_expanded,shape_ph:np.shape(image_np_expanded)[1:3]})
                
                
                augmented_np_images.append(image_np_expanded)
                # Actual detection.
                category_inds = category_index.keys()
                max_ind = max(category_inds)+1
                all_boxes = np.array([[0.,0.,0.,0.]])
                all_scores = np.zeros((1,max_ind))
                
                for i_aug,img in enumerate(augmented_np_images):
                    
                    print(i_aug)
                    output_dict=run_inference_for_single_image(img,sess)
                    
                    ## Filter detections with the given thresholds
                    category_inds = category_index.keys()
                    filter_ind = np.array([])
                    for cat_ind in category_inds:
                        ind_cls = filter_threshold(args.scr_thr[cat_ind],cat_ind,output_dict['detection_classes'],output_dict['detection_scores'])
                        filter_ind = np.concatenate((filter_ind,ind_cls))
                    filter_ind = filter_ind.astype(int)
                    #IPython.embed()
                    # Visualization of the results of a detection.
                    
                    im = augmented_np_images[-1][0].copy()
                    boxes = output_dict['detection_boxes'][filter_ind]
                    if aug_type[i_aug] == 'flip':
                        x2 = 1.0 - output_dict['detection_boxes'][filter_ind][:,1:2]
                        x1 = 1.0 - output_dict['detection_boxes'][filter_ind][:,3:4]
                        y1 = output_dict['detection_boxes'][filter_ind][:,0:1] 
                        y2 = output_dict['detection_boxes'][filter_ind][:,2:3] 
                        boxes = np.hstack((y1,x1,y2,x2))
                    all_boxes = np.vstack((all_boxes,boxes))
                    nms_scores = np.zeros((len(boxes),max_ind))
                    for scr_l,clss,scr in zip(nms_scores,
                                              output_dict['detection_classes'][filter_ind],
                                              output_dict['detection_scores'][filter_ind]):
                        scr_l[clss] = scr
                    all_scores = np.vstack((all_scores,nms_scores))    
                    #IPython.embed() 
                    '''
                    vis_util.visualize_boxes_and_labels_on_image_array(
                      im,
                      boxes,
                      output_dict['detection_classes'][filter_ind],
                      output_dict['detection_scores'][filter_ind],
                      category_index,
                      instance_masks=output_dict.get('detection_masks'),
                      use_normalized_coordinates=True,
                      line_thickness=8)
                    
                    # Get image ID to save the image and predictions
                    try:
                        img_id = int(os.path.basename(image_path).split('.')[0])
                    except Exception as e:
                        print(e)
                        img_id = image_i
                        
                    if args.save_images:
                        save_image_path = os.path.join(args.output_path,'images_aug')
                        if not os.path.isdir(save_image_path):
                            os.mkdir(save_image_path)
                        plt.imsave(os.path.join(save_image_path,os.path.basename(image_path)+'_{}.png'.format(i_aug)),im/255.0)
                    '''
                try:
                    img_id = int(os.path.basename(image_path).split('.')[0])
                except Exception as e:
                    print(e)
                    img_id = image_i
                im = augmented_np_images[-1][0].copy()        
                boxlist = np_box_list.BoxList(all_boxes[1:,:])        
                boxlist.add_field('scores',all_scores[1:,:])
                boxlist_clean = np_box_list_ops.multi_class_non_max_suppression(
                            boxlist, score_thresh=0.5, iou_thresh=0.45, max_output_size=20)
                scores_clean = boxlist_clean.get_field('scores')
                classes_clean = boxlist_clean.get_field('classes')
                boxes = boxlist_clean.get()
                #IPython.embed()
                vis_util.visualize_boxes_and_labels_on_image_array(
                      im,
                      boxes,
                      classes_clean.astype(np.int),
                      scores_clean,
                      category_index,
                      instance_masks=output_dict.get('detection_masks'),
                      use_normalized_coordinates=True,
                      line_thickness=8)
                if args.save_images:
                    save_image_path = os.path.join(args.output_path,'images_aug')
                    if not os.path.isdir(save_image_path):
                        os.mkdir(save_image_path)
                    plt.imsave(os.path.join(save_image_path,'nms'+os.path.basename(image_path)),im/255.0)
                    
                # The ouputs are in different format ([y1,x1,y2,x2] and scaled into 0-1). Converts into KITTI ([x1,y1,x2,y2] and pixel values)
                y1 = boxes[:,0] * image.size[1]
                x1 = boxes[:,1] * image.size[0]
                y2 = boxes[:,2] * image.size[1]
                x2 = boxes[:,3] * image.size[0]
                boxes[:,0] = x1
                boxes[:,1] = y1
                boxes[:,2] = x2
                boxes[:,3] = y2
                
                write_predictions_to_file_in_kitti_format(args.output_path,None,args.scr_thr,boxes,\
                                                     scores_clean,classes_clean,\
                                                     None,None,img_id,with_score=args.with_score,category_index=category_index)
                #num_det = output_dict['num_detections']
                ### Filter for tracking
                #car_indices = filter_threshold(args.scr_thr[1],1,output_dict['detection_classes'],output_dict['detection_scores'])
                #ped_indices = filter_threshold(args.scr_thr[2],2,output_dict['detection_classes'],output_dict['detection_scores'])
                #filter_ind = np.concatenate((car_indices,ped_indices))
                #num_det = np.max(np.argwhere(output_dict['detection_scores']>trck_scr_thr))
                #results_arr.append([output_dict['detection_boxes'][filter_ind],output_dict['detection_scores'][filter_ind],\
                #                   output_dict['detection_classes'][filter_ind]])
                #image_arr.append([np.array(image)])
                if args.save_images:
                    save_image_path = os.path.join(args.output_path,'images')
                    if not os.path.isdir(save_image_path):
                        os.mkdir(save_image_path)
                    plt.imsave(os.path.join(save_image_path,os.path.basename(image_path)),image_np)
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen_graph_path',help='Path to the exported frozen graph of the network that will be used for inference.')
    parser.add_argument('--path_to_labels',help='Path to the label map of the trained network.')
    parser.add_argument('--path_to_images',help='Path to the images that will be tested. All images in this directory will be tested.')
    parser.add_argument('--image_format',default='png',help='Format of the images that will be tested. By default they are considered as png. It can be jpg, jpeg, etc. ')
    parser.add_argument('--output_path',help='Path that the predictions will be recorded in KITTI format. Names of the text files will be indices that the images have.')
    parser.add_argument('--scr_thr', metavar='N', type=float, nargs='+',help='A list of score thresholds for each class in the label map. The thresholds of classes will be called using the indices in label maps. Only the predictions above this threshold will be written in the predictions. ')
    parser.add_argument('--with_score', action='store_true', help='To save predictions into txt files with the prediction score in KITTI format.')
    parser.add_argument('--save_images', action='store_true', help='To save images with the predicted boxes drawn. Images are saved in the images folder of output_path')
    
    args = parser.parse_args()
    
    inference(args)

