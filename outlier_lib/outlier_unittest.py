"""
description: Outlier Removal Example

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

## Load the cuda function
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
path_grid = tf.resource_loader.get_path_to_datafile(
	os.path.join(path, 'outlier.so'))

outlier_lib = tf.load_op_library(path_grid)
outlier = outlier_lib.outlier

@ops.RegisterGradient('Outlier')
def _outlier(op, grad):
	return [None, None, None] # no grad

## outlier removal unittest
def load_exr(filename, datatype=np.float16):
	import OpenEXR
	import Imath
	HALF  = Imath.PixelType(Imath.PixelType.HALF)
	if not OpenEXR.isOpenExrFile(filename):
		raise Exception("File '%s' is not an EXR file." % filename)
	infile = OpenEXR.InputFile(filename)

	header = infile.header()
	dw = header['dataWindow']
	width = dw.max.x - dw.min.x + 1
	height = dw.max.y - dw.min.y + 1

	return_matrix_ch_B = np.fromstring(infile.channels('B')[0], 
		dtype=datatype).reshape(height, width)
	return_matrix_ch_G = np.fromstring(infile.channels('G')[0], 
		dtype=datatype).reshape(height, width)
	return_matrix_ch_R = np.fromstring(infile.channels('R')[0], 
		dtype=datatype).reshape(height, width)
	matrix_new = np.stack((return_matrix_ch_R, 
		return_matrix_ch_G, return_matrix_ch_B), axis=-1)
	return matrix_new

def tone_mapping(input_image):
	x = np.maximum(0., input_image - 0.004)
	tone_mapped_color = (x*(6.2*x + 0.5))/(x*(6.2*x + 1.7) + 0.06)
	where_are_NaNs = np.isnan(tone_mapped_color)
	tone_mapped_color[where_are_NaNs] = 0
	return tone_mapped_color

if __name__ == "__main__":
	data_type = np.float32
	## Change load_dir to your data folder
	load_dir = '/mnt/ext_toshiba/denoising/tungsten/data/xiaoxu_denoising_data/dining-room/spp_64_data/0'
	input_image = load_exr(os.path.join(load_dir, 'color.exr'), datatype=np.float16).astype(data_type)
	input_albedo = plt.imread(os.path.join(load_dir, 'albedo.png')).astype(data_type) / 255.0
	input_normal = load_exr(os.path.join(load_dir, 'normal.exr'), datatype=np.float16).astype(data_type)
	input_normal = (input_normal + 1.0) * 0.5

	tf_input_image = tf.get_variable("tf_input_image", initializer=np.expand_dims(input_image, axis=0), dtype=data_type)
	tf_input_albedo = tf.get_variable("tf_input_albedo", initializer=np.expand_dims(input_albedo, axis=0), dtype=data_type)
	tf_input_normal = tf.get_variable("tf_input_normal", initializer=np.expand_dims(input_normal, axis=0), dtype=data_type)
	
	with tf.device("/gpu:0"):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			tf_output_image = outlier(tf_input_image, tf_input_albedo, tf_input_normal)
			output_image = np.squeeze(sess.run(tf_output_image))

	plt.imsave('example_input.png', tone_mapping(input_image))
	plt.imsave('example_output.png', tone_mapping(output_image))