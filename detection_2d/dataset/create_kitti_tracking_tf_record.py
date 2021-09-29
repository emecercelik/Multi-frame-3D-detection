# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw KITTI detection dataset to TFRecord for object_detection.

Converts KITTI detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. The raw dataset can be
  downloaded from:
  http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip.
  http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
  Permission can be requested at the main website.

  KITTI detection dataset contains 7481 training images. Using this code with
  the default settings will set aside the first 500 images as a validation set.
  This can be altered using the flags, see details below.

Example usage:
    python object_detection/dataset_tools/create_kitti_tf_record.py \
        --data_dir=/home/user/kitti \
        --output_path=/home/user/kitti.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou,area

import IPython

import matplotlib.pyplot as plt

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/training/label_2 (annotations) and'
                           '<data_dir>/data_object_image_2/training/image_2'
                           '(images).')
tf.app.flags.DEFINE_list('drive_ids',[],'Drive IDs that the tracking \
                                  dataset of KITTI will be read from. ')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('classes_to_use', 'car,pedestrian,dontcare',
                           'Comma separated list of class names that will be'
                           'used. Adding the dontcare class will remove all'
                           'bboxs in the dontcare regions.')
tf.app.flags.DEFINE_string('label_map_path', 'data/kitti_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_bool('is_training',False,'if True, the training tfrecords will be prepared. Otherwise validation')
tf.app.flags.DEFINE_bool('min_area',False,'if True, boxes that have smaller area than a square, whose edge is defined with min_area_one_edge, will be discarded.')
tf.app.flags.DEFINE_float('min_area_one_edge',32.0,'if min_area flag is True, this defines one edge of a sqaure box, boxes smaller than whose area will be discarded.')
tf.app.flags.DEFINE_string('statistics_path', None, 'Path where plots of the data statistics will be saved.')

FLAGS = tf.app.flags.FLAGS

def convert_kitti_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, is_training,
                               drive_ids,min_area,min_area_one_edge,
                               statistics_path):
  """Convert the KITTI detection dataset to TFRecords.

  Args:
    data_dir: The full path to the unzipped folder containing the unzipped data
      from data_object_image_2 and data_object_label_2.zip.
      Folder structure is assumed to be: data_dir/data_object_label_2/training/label_2/<drive_id>.txt (annotations)
      and data_dir/data_object_image_2/training/image_2/<drive_id>/ (images).
    drive_ids: The drive IDs that will be used while generating tfrecords.
    output_path: The path to which TFRecord files will be written. The TFRecord
      with the training set will be located at: <output_path>_train.tfrecord
      And the TFRecord with the validation set will be located at:
      <output_path>_val.tfrecord
    classes_to_use: List of strings naming the classes for which data should be
      converted. Use the same names as presented in the KIITI README file.
      Adding dontcare class will remove all other bounding boxes that overlap
      with areas marked as dontcare regions.
    label_map_path: Path to label map proto
    validation_set_size: How many images should be left as the validation set.
      (Ffirst `validation_set_size` examples are selected to be in the
      validation set).
  """
  
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  sample_count = 0

  annotation_dir = os.path.join(data_dir,
                                'data_tracking_label_2',
                                'training',
                                'label_02')

  image_dir = os.path.join(data_dir,
                           'data_tracking_image_2',
                           'training',
                           'image_02')
  if is_training:
      tfrecords_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  else:
      tfrecords_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)
  drive_ids = [int(drive_id) for drive_id in drive_ids[0].split(' ')]
  print(drive_ids)
  statistics = {}
  statistics['3d_width'] = {}
  statistics['3d_height'] = {}
  statistics['3d_length'] = {}
  statistics['num_frames'] = {}
  for cls_name in label_map_dict.keys():
     statistics['3d_width'][cls_name] = []
     statistics['3d_height'][cls_name] = []
     statistics['3d_length'][cls_name] = []
  for drive_id in drive_ids:
      statistics[drive_id] = {}
      statistics[drive_id]['track_count']={}
      statistics[drive_id]['area']=[]
      for key in label_map_dict.keys():
          statistics[drive_id][key]=[]
      drive_image_dir = os.path.join(image_dir,'{:04d}'.format(drive_id))
      drive_annotation_dir = os.path.join(annotation_dir,'{:04d}.txt'.format(drive_id))
      images = sorted(tf.gfile.ListDirectory(drive_image_dir))
      statistics['num_frames'][drive_id] = len(images)
      for img_name in images:
        img_num = int(img_name.split('.')[0])
        for key in label_map_dict.keys():
            statistics[drive_id][key].append(0)
        
        print("Drive:{:04d}, Img:{:06d}, Training: {}".format(drive_id,img_num, is_training))
        
        img_anno = read_annotation_file(drive_annotation_dir,img_num)
        for trck_id in img_anno['track_id']:
            try:
                statistics[drive_id]['track_count'][trck_id]+=1
            except:
                statistics[drive_id]['track_count'][trck_id]=1
        for obj_cls,h_3d,w_3d,l_3d in zip(img_anno['type'],img_anno['3d_bbox_height'],img_anno['3d_bbox_width'],img_anno['3d_bbox_length']):
            if obj_cls in label_map_dict.keys():
                statistics['3d_width'][obj_cls].append(h_3d)
                statistics['3d_height'][obj_cls].append(w_3d)
                statistics['3d_length'][obj_cls].append(l_3d)
            
        for obj_cls in img_anno['type']:
            if obj_cls in label_map_dict.keys():
                statistics[drive_id][obj_cls][-1]+=1
                
        image_path = os.path.join(drive_image_dir, img_name)
        
        # Filter all bounding boxes of this frame that are of a legal class, and
        # don't overlap with a dontcare region.
        # TODO(talremez) filter out targets that are truncated or heavily occluded.
        annotation_for_image = filter_annotations(img_anno, classes_to_use,min_area,min_area_one_edge)
        
        example,image_shape = prepare_example(image_path, annotation_for_image, label_map_dict)
        tfrecords_writer.write(example.SerializeToString())
        sample_count+=1
        
        for l,t,r,b in zip(img_anno['2d_bbox_left'],img_anno['2d_bbox_top'],\
                           img_anno['2d_bbox_right'],img_anno['2d_bbox_bottom']):
            width = min(np.abs(r-l),image_shape[1])
            height= min(np.abs(b-t),image_shape[0])
            area = width*height/(image_shape[0]*image_shape[1])
            statistics[drive_id]['area'].append(area)
      if -1 in statistics[drive_id]['track_count'].keys(): del statistics[drive_id]['track_count'][-1]
      
      if statistics_path is not None:
          #### Save Object frequency plot
          num_different_objects = len(list(statistics[drive_id]['track_count'].keys()))
          track_count_list = [statistics[drive_id]['track_count'][key] for key in statistics[drive_id]['track_count'].keys()]
          cropped_list = [c if c<=10 else 11 for c in track_count_list]
          track_bins = [0,1,2,3,4,5,6,7,8,9,10,11]
          plt.Figure()
          res1 = plt.hist(cropped_list, bins=track_bins,stacked=True)
          ax = plt.gca()
          for count, patch in zip(res1[0],res1[2]):
                ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
          #plt.ylim(-1,2500)
          plt.title('Object Frequency\n Drive:{}, # of different objects:{}'.format(drive_id,num_different_objects))
          plt.xlabel('Number of Frames')
          plt.ylabel('Number of Objects')
          drive_sta_path = os.path.join(statistics_path,'{:04d}'.format(drive_id))
          if not os.path.isdir(drive_sta_path):
              os.makedirs(drive_sta_path)
          plt.savefig(os.path.join(drive_sta_path,'track_hist_{}.png'.format(drive_id)))
          plt.clf()
          #### Save Object Area Plot 1 (Finer plot)
          area_bins = [0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
          plt.Figure()
          res1 = plt.hist(statistics[drive_id]['area'], bins=area_bins)
          ax = plt.gca()
          for count, patch in zip(res1[0],res1[2]):
              ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
          #plt.ylim(-1,2500)
          plt.title('Object Area (Normalized with Image Area)\n Img W:{}, H:{}, Drive:{}'.format(image_shape[1],image_shape[0],drive_id))
          plt.xlabel('Normalized Area')
          plt.ylabel('Number of Objects')
          
          plt.savefig(os.path.join(drive_sta_path,'area_fine_{}.png'.format(drive_id)))
          plt.clf()
          #### Save Object Area Plot 2 (broader plot)
          area_bins = [0,0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1]
          plt.Figure()
          res1 = plt.hist(statistics[drive_id]['area'], bins=area_bins)
          ax = plt.gca()
          for count, patch in zip(res1[0],res1[2]):
              ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
          #plt.ylim(-1,2500)
          plt.title('Object Area (Normalized with Image Area)\n Img W:{}, H:{}, Drive:{}'.format(image_shape[1],image_shape[0],drive_id))
          plt.xlabel('Normalized Area')
          plt.ylabel('Number of Objects')
          
          plt.savefig(os.path.join(drive_sta_path,'area_rough_{}.png'.format(drive_id)))
          plt.clf()
          #### Save number of objects per frame statistics
          for obj_cls in label_map_dict.keys():
              statistics[drive_id][obj_cls]
              cls_bins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
              plt.Figure()
              res1 = plt.hist(statistics[drive_id][obj_cls], bins=cls_bins)
              ax = plt.gca()
              for count, patch in zip(res1[0],res1[2]):
                  ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()))
              #plt.ylim(-1,2500)
              plt.title('Number of Objects per Frame\n Class:{}, Drive:{}'.format(obj_cls,drive_id))
              plt.xlabel('Number of Objects per Frame')
              plt.ylabel('Number of Frames')
              
              plt.savefig(os.path.join(drive_sta_path,'num_obj_per_frame_{}_{}.png'.format(obj_cls,drive_id)))
              plt.clf()
          
  #### Save object size statistics in a txt file
  if is_training:
      txt_file_name = 'statistics_train.txt'
  else:
      txt_file_name = 'statistics_val.txt'
  with open(os.path.join(statistics_path,txt_file_name),'w') as file_name:
      file_name.write('3D object size statistics:\n')
      drives_name=''
      num_frames_per_drive='\nNumber of Frames per Drive (Session)\n'
      num_objs_per_drive='\nNumber of Objects per Drive (Session)\n'
      for drive_id in drive_ids:
          drives_name+='{} '.format(drive_id)
          num_frames_per_drive+='* Drive {}: {}\n'.format(drive_id,statistics['num_frames'][drive_id])
          cls_keys = label_map_dict.keys()
          num_objs_per_drive+='* Drive:{} '.format(drive_id)
          for cls_key in cls_keys:
              num_cls_obj = np.sum(statistics[drive_id][cls_key])
              num_objs_per_drive+=' {}: {} '.format(cls_key,num_cls_obj)
          num_objs_per_drive+='\n'
      file_name.write(drives_name+'\n')
      
      sta_txt = ''
      for obj_cls in label_map_dict.keys():
          mean_3d_width = np.mean(statistics['3d_width'][obj_cls])
          mean_3d_height = np.mean(statistics['3d_height'][obj_cls])
          mean_3d_length = np.mean(statistics['3d_length'][obj_cls])
          sta_txt+='Cls: {}\n Mean 3D Width: {}\n Mean 3D Height: {}\n Mean 3D Length: {}\n'.format(\
                         obj_cls,mean_3d_width,mean_3d_height,mean_3d_length)
      
      
      file_name.write(sta_txt) 
      file_name.write(num_frames_per_drive)
      file_name.write(num_objs_per_drive)
        
  tfrecords_writer.close()
  
  #IPython.embed()
  ## Plot statistics
  '''
  for drive_id in drive_ids:
      track_count_list = [statistics[drive_id]['track_count'][x] for key in statistics[drive_id]['track_count'].keys()]
      for d_id in statistics.keys():
          statistics[]
          plt.Figure()
          res1 = plt.hist(stat_results[cls], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
          ax = plt.gca()
          for count, patch in zip(res1[0],res1[2]):
            ax.annotate(str(int(count)), xy=(patch.get_x()+0.025, patch.get_height()))
          #plt.ylim(-1,2500)
          plt.title('IoU values between predictions and ground-truth labels\nCls:{}, Drive:{}, Log:{}, Total # of Gt:{}'.format(cls,drive_id,os.path.basename(args.path_to_log),int(np.sum(res1[0]))))
          plt.xlabel('IoU with Gt')
          plt.ylabel('Num of predictions')
          plt.savefig(os.path.join(os.path.dirname(pred_folders[-1]),'hist.png'))
  '''
def prepare_example(image_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.

  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.

  Returns:
    example: The converted tf.Example.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(image.shape[1])
  height = int(image.shape[0])

  xmin_norm = np.clip(annotations['2d_bbox_left'] / float(width),0.0,1.09)
  ymin_norm = np.clip(annotations['2d_bbox_top'] / float(height),0.0,1.09)
  xmax_norm = np.clip(annotations['2d_bbox_right'] / float(width),0.0,1.09)
  ymax_norm = np.clip(annotations['2d_bbox_bottom'] / float(height),0.0,1.09)
  
  difficult_obj = [0]*len(xmin_norm)
  
  
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [x.encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(
          [label_map_dict[x] for x in annotations['type']]),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.float_list_feature(
          annotations['truncated']),
      'image/object/alpha': dataset_util.float_list_feature(
          annotations['alpha']),
      'image/object/3d_bbox/height': dataset_util.float_list_feature(
          annotations['3d_bbox_height']),
      'image/object/3d_bbox/width': dataset_util.float_list_feature(
          annotations['3d_bbox_width']),
      'image/object/3d_bbox/length': dataset_util.float_list_feature(
          annotations['3d_bbox_length']),
      'image/object/3d_bbox/x': dataset_util.float_list_feature(
          annotations['3d_bbox_x']),
      'image/object/3d_bbox/y': dataset_util.float_list_feature(
          annotations['3d_bbox_y']),
      'image/object/3d_bbox/z': dataset_util.float_list_feature(
          annotations['3d_bbox_z']),
      'image/object/3d_bbox/rot_y': dataset_util.float_list_feature(
          annotations['3d_bbox_rot_y']),
  }))

  return example,image.shape


def filter_annotations(img_all_annotations, used_classes,min_area,min_area_one_edge):
  """Filters out annotations from the unused classes and dontcare regions.

  Filters out the annotations that belong to classes we do now wish to use and
  (optionally) also removes all boxes that overlap with dontcare regions.

  Args:
    img_all_annotations: A list of annotation dictionaries. See documentation of
      read_annotation_file for more details about the format of the annotations.
    used_classes: A list of strings listing the classes we want to keep, if the
    list contains "dontcare", all bounding boxes with overlapping with dont
    care regions will also be filtered out.

  Returns:
    img_filtered_annotations: A list of annotation dictionaries that have passed
      the filtering.
  """

  img_filtered_annotations = {}

  # Filter the type of the objects.
  relevant_annotation_indices = [
      i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
  ]

  for key in img_all_annotations.keys():
    img_filtered_annotations[key] = (
        img_all_annotations[key][relevant_annotation_indices])

  if 'dontcare' in used_classes:
    dont_care_indices = [i for i,
                         x in enumerate(img_filtered_annotations['type'])
                         if x == 'dontcare']

    # bounding box format [y_min, x_min, y_max, x_max]
    all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                          img_filtered_annotations['2d_bbox_left'],
                          img_filtered_annotations['2d_bbox_bottom'],
                          img_filtered_annotations['2d_bbox_right']],
                         axis=1)

    ious = iou(boxes1=all_boxes,
               boxes2=all_boxes[dont_care_indices])

    # Remove all bounding boxes that overlap with a dontcare region.
    if ious.size > 0:
      boxes_to_remove = np.amax(ious, axis=1) > 0.0
      for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_filtered_annotations[key][np.logical_not(boxes_to_remove)])
  if min_area:
      # bounding box format [y_min, x_min, y_max, x_max]
      all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                              img_filtered_annotations['2d_bbox_left'],
                              img_filtered_annotations['2d_bbox_bottom'],
                              img_filtered_annotations['2d_bbox_right']],
                             axis=1)
      areas = area(all_boxes)
      boxes_to_remove = areas < np.square(min_area_one_edge)
      for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_filtered_annotations[key][np.logical_not(boxes_to_remove)])
      

  return img_filtered_annotations


def read_annotation_file(filename,img_id):
  """Reads a KITTI annotation file.

  Converts a KITTI annotation file into a dictionary containing all the
  relevant information.

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip().split(' ') for x in content if int(x.strip().split(' ')[0])==img_id]

  anno = {}
  anno['frame_id'] = np.array([int(x[0]) for x in content])
  anno['track_id'] = np.array([int(x[1]) for x in content])
  anno['type'] = np.array([x[2].lower() for x in content])
  anno['truncated'] = np.array([float(x[3]) for x in content])
  anno['occluded'] = np.array([int(x[4]) for x in content])
  anno['alpha'] = np.array([float(x[5]) for x in content])

  anno['2d_bbox_left'] = np.array([float(x[6]) for x in content])
  anno['2d_bbox_top'] = np.array([float(x[7]) for x in content])
  anno['2d_bbox_right'] = np.array([float(x[8]) for x in content])
  anno['2d_bbox_bottom'] = np.array([float(x[9]) for x in content])

  anno['3d_bbox_height'] = np.array([float(x[10]) for x in content])
  anno['3d_bbox_width'] = np.array([float(x[11]) for x in content])
  anno['3d_bbox_length'] = np.array([float(x[12]) for x in content])
  anno['3d_bbox_x'] = np.array([float(x[13]) for x in content])
  anno['3d_bbox_y'] = np.array([float(x[14]) for x in content])
  anno['3d_bbox_z'] = np.array([float(x[15]) for x in content])
  anno['3d_bbox_rot_y'] = np.array([float(x[16]) for x in content])

  return anno


def main(_):
  convert_kitti_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_path=FLAGS.output_path,
      classes_to_use=FLAGS.classes_to_use.split(','),
      label_map_path=FLAGS.label_map_path,
      is_training=FLAGS.is_training,
      drive_ids=FLAGS.drive_ids,
      min_area=FLAGS.min_area,
      min_area_one_edge=FLAGS.min_area_one_edge,
      statistics_path=FLAGS.statistics_path)

if __name__ == '__main__':
  tf.app.run()
