from __future__ import print_function

import os, pickle
import sys
import vkitti_utils
# import tensorflow as tf
from vkitti_object import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

dataset = vkitti_object(os.path.join(ROOT_DIR, 'dataset/virtual'))

world_ids = ['0001', '0002', '0006', '0018', '0020']
variations = ['15-deg-left', '30-deg-left', 'clone', 'morning', 'rain', '15-deg-right', '30-deg-right', 'fog',
              'overcast', 'sunset']

all_objects = 0
all_frames = 0

min_depth = 0

total_l=0
total_w=0
total_h=0

max_nr_objects_in_frame = 0

for world_id in world_ids:
    for variation_id, variation in enumerate(variations):
        print(world_id + ' ' + variation)
        for frame_id in range(dataset.get_nr_frames(world_id, variation)):

            objects = dataset.get_ground_truth_objects(world_id, variation)
            objects_in_frame = []

            if frame_id in objects:
                all_frames += 1
                objects_in_frame = objects[frame_id]

            counter = 0
            for object in objects_in_frame:
                if object.type not in ['Car']: continue
                counter += 1
                total_l += object.l
                total_w += object.w
                total_h += object.h

                all_objects += 1

            if counter > max_nr_objects_in_frame:
                max_nr_objects_in_frame = counter

print(all_objects / all_frames)
print("Average object size ==> l = ", total_l/all_objects, ' w = ', total_w/all_objects, ' h = ', total_h/all_objects)

print("max_nr_objects_in_frame = ",max_nr_objects_in_frame)
