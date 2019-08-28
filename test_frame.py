import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
import pdb

os.environ['PYTHONPATH']='/env/python:/home/sjang/research/data/tmp/waymo-od'
m=imp.find_module('waymo_open_dataset', ['.'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

tf.enable_eager_execution()


def image_show(data, name, layout, cmap=None):
    """Show an image."""
    plt.subplot(*layout)
    img_jpeg = tf.image.decode_jpeg(data)
    plt.imshow(img_jpeg, cmap=cmap)
    pdb.set_trace()
    plt.title(name)
    plt.grid(False)
    plt.axis('off')

def save_image(data, name):
    fname = tf.constant(name)
    fwrite = tf.write_file(fname, data)

def parse_range_image_and_camera_projection(frame):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto
  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:
      range_image_str_tensor = tf.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == open_dataset.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = open_dataset.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      camera_projection_str_tensor = tf.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]
    if len(laser.ri_return2.range_image_compressed) > 0:
      range_image_str_tensor = tf.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      camera_projection_str_tensor = tf.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)
  return range_images, camera_projections, range_image_top_pose 


if __name__ == '__main__':

    
    FILENAME = '../../waymo_open_dataset/tfrecords/train/segment-15533468984793020049_800_000_820_000_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, 
                compression_type='')
    n_frame = 0
    for data in dataset:
    	frame = open_dataset.Frame()
    	frame.ParseFromString(bytearray(data.numpy()))
    
    	(range_images, 
    	camera_projections,
    	range_image_top_pose) = parse_range_image_and_camera_projection(frame)
    
    	for index, image in enumerate(frame.images):
            img_name = open_dataset.CameraName.Name.Name(image.name) + '_%d.jpg'%n_frame
            img_name = 'images/' + img_name
            save_image(image.image, img_name)
            print('saved {}'.format(img_name))

    		#image_show(image.image, 
    		#	open_dataset.CameraName.Name.Name(image.name), 
    		#	[3,3,index+1])
    	#plt.show()
    
    	n_frame += 1
    	print('frame: %d' % n_frame)	
