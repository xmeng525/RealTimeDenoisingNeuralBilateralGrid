"""
description: DataLoader

@author: Xiaoxu Meng
@author: QZheng
"""

from __future__ import division
import os
import numpy as np
import scipy.misc
import math
import PIL.Image
import array
import tensorflow as tf
from image_utils import load_exr

class dataLoader(object):

	def __init__(self,
				 data_dir,
				 subset,
				 image_start_idx,
				 img_per_scene,
				 scene_list,
				 patch_per_img=50,
				 patch_width=128,
				 patch_height=128):
		"""
		- data_dir is location
		- subset use train|test
		- batch_size is int
		"""
		self.data_dir = data_dir
		self.subset = subset
		self.patch_width = patch_width
		self.patch_height = patch_height
		self.scene_list = scene_list
		self.patch_per_img = patch_per_img
		self.image_start_idx = image_start_idx
		self.img_per_scene = img_per_scene
		self.dataset_name = self.get_dataset_name()

		self.load_dataset(subset)

	def get_dataset_name(self):
		dataset_name = 'bw_data_' + str(
			self.patch_height) + 'x' + str(self.patch_width) + '_' + str(
			len(self.scene_list)) + 'scenes_' + str(self.img_per_scene) + 'ips_' + str(
			self.patch_per_img) + 'ppi_' + self.subset + '.tfrecords'
		return os.path.join(dataset_name)

	def load_dataset(self, subset):
		if os.path.exists(self.dataset_name):
			print(self.dataset_name, ' exisits.')  # all is good
		else:
			self.encode_to_tfrecords(subset)

	def encode_to_tfrecords(self, subset):
		writer = tf.python_io.TFRecordWriter(self.dataset_name)
		print(self.subset, 'Data_dir ', self.data_dir)
		if subset == 'train' or subset == 'valid':
			for scene_name in self.scene_list:
				print('Processing scene ', scene_name)
				data_subdir = os.path.join(self.data_dir, scene_name)
				print('Visit data subdir ', data_subdir)
				for idx in range(self.image_start_idx, self.img_per_scene + self.image_start_idx):
					print("	" + str(idx))
					exr_name = str(idx) + '.exr'
					albedo_path = os.path.join(data_subdir, 'inputs', 'albedo' + exr_name)
					normal_path = os.path.join(data_subdir, 'inputs', 'shading_normal' + exr_name)
					depth_path = os.path.join(data_subdir, 'depth_normalized', str(idx) + '.png')
					noisy_path = os.path.join(data_subdir, 'radiance_accum', 'accum_color_noabd' + exr_name)
					GT_path = os.path.join(data_subdir, 'inputs', 'reference' + exr_name)

					# original albedo ranges between [0,1] ==> [0,1]
					albedo = load_exr(albedo_path, datatype=np.float32)
					# original normal ranges between [-1,1] ==> [0,1]
					normal = (load_exr(normal_path, datatype=np.float32) + 1.0) * 0.5
					# original depth ranges between [0,1] ==> [0,1]
					depth = np.expand_dims(np.asarray(PIL.Image.open(depth_path)), axis=2)/255
					# original noisy ranges between [0, infty] ==> [0,1] (tonempping)
					noisy = load_exr(noisy_path, datatype=np.float16)

					# original GT ranges between [0, infty] ==> [0,1] (tonempping)
					GT_full_image = load_exr(GT_path, datatype=np.float32)

					noisy_full_image = np.concatenate(
						(noisy, albedo, normal, depth), axis=2)
					noisy_full_image = noisy_full_image[:, :, 0:10]

					GT_full_image = GT_full_image.astype(np.float16)
					noisy_full_image = noisy_full_image.astype(np.float16)
					# crop
					for _ in range(self.patch_per_img):
						noisy_one, target_one = self.random_crop(
							noisy_full_image, GT_full_image)

						aug_idx = np.random.randint(0, 8)
						target_one = self.aug_input(target_one, aug_idx)
						noisy_one = self.aug_input(noisy_one, aug_idx)
						feature = {
							'target': tf.train.Feature(
								bytes_list = tf.train.BytesList(
									value=[target_one.tostring()])),
							'input': tf.train.Feature(
								bytes_list = tf.train.BytesList(
									value=[noisy_one.tostring()]))}
						example = tf.train.Example(
							features=tf.train.Features(feature=feature))
						writer.write(example.SerializeToString())

		else: # subset == 'test'
			for scene_name in self.scene_list:
				print('Processing scene ', scene_name)
				data_subdir = os.path.join(self.data_dir, scene_name)
				print('Visit test data subdir ', data_subdir)
				padding_w = 0
				padding_h = 0

				for idx in range(self.image_start_idx, self.img_per_scene + self.image_start_idx):
					print("	" + str(idx))
					exr_name = str(idx) + '.exr'
					albedo_path = os.path.join(data_subdir, 'inputs', 'albedo' + exr_name)
					normal_path = os.path.join(data_subdir, 'inputs', 'shading_normal' + exr_name)
					depth_path = os.path.join(data_subdir, 'depth_normalized', str(idx) + '.png')
					noisy_path = os.path.join(data_subdir, 'radiance_accum', 'accum_color_noabd' + exr_name)
					GT_path = os.path.join(data_subdir, 'inputs', 'reference' + exr_name)

					# original albedo ranges between [0,1] ==> [0,1]
					albedo = load_exr(albedo_path, datatype=np.float32)
					# original normal ranges between [-1,1] ==> [0,1]
					normal = (load_exr(normal_path, datatype=np.float32) + 1.0) * 0.5
					# original depth ranges between [0,1] ==> [0,1]
					depth = np.expand_dims(np.asarray(PIL.Image.open(depth_path)), axis=2)/255
					# original noisy ranges between [0, infty] ==> [0,1] (tonempping)
					noisy = load_exr(noisy_path, datatype=np.float16)

					# original GT ranges between [0, infty] ==> [0,1] (tonempping)
					GT_full_image = load_exr(GT_path, datatype=np.float32)

					noisy_full_image = np.concatenate(
						(noisy, albedo, normal, depth), axis=2)
					noisy_full_image = noisy_full_image[:, :, 0:10]

					resolution = noisy_full_image.shape
					noisy_one = np.zeros((resolution[0] + padding_h,
										 resolution[1] + padding_w, 10),
										 dtype = np.float16)
					noisy_one[0:resolution[0], 0:resolution[1],:] = \
						noisy_full_image

					target_one = np.zeros((resolution[0] + padding_h,
										   resolution[1] + padding_w, 3),
										  dtype=np.float16)
					target_one[0:resolution[0], 0:resolution[1],:] = \
						GT_full_image

					feature = {
						'target': tf.train.Feature(
							bytes_list=tf.train.BytesList(
								value=[target_one.tostring()])),
						'input': tf.train.Feature(
							bytes_list=tf.train.BytesList(
								value=[noisy_one.tostring()]))}
					example = tf.train.Example(
						features=tf.train.Features(feature=feature))
					writer.write(example.SerializeToString())
		writer.close()
		print(self.subset, ' data preprocess finished.')

	def with_offset_crop(self, x, y, offseth, offsetw, size=(256, 256)):
		cropped_x = x[offseth:offseth + size[0], offsetw:offsetw + size[1], :]
		cropped_y = y[offseth:offseth + size[0], offsetw:offsetw + size[1], :]
		cropped_x = cropped_x
		cropped_y = cropped_y
		return cropped_x, cropped_y

	def random_crop(self, x, y):
		cropped_x, cropped_y = self.random_crop_np(x, y, size=(
			self.patch_height, self.patch_width))
		return cropped_x, cropped_y

	def random_crop_np(self, x, y, size=(256, 256)):
		assert x.shape[0] >= size[0]
		assert x.shape[1] >= size[1]
		offseth, offsetw = self.generate_offset(
			shape=[x.shape[0], x.shape[1], x.shape[2]], size=size)
		cropped_x = x[offseth:offseth + size[0], offsetw:offsetw + size[1], :]
		cropped_y = y[offseth:offseth + size[0], offsetw:offsetw + size[1], :]
		return cropped_x, cropped_y

	def generate_offset(self, shape, size=(256, 256)):
		h, w, ch = shape
		range_h = h - size[0]
		range_w = w - size[1]
		offseth = 0 if range_h == 0 else np.random.randint(range_h)
		if range_w == 0:
			offsetw = 0
		else:
			my_rand = np.random.randint(range_w)
			offsetw = 1 if my_rand == 0 else int(np.log2(my_rand) / np.log2(
				range_w) * range_w)
		return offseth, offsetw

	def aug_input(self, img, idx=0):
		if idx == 0:
			return img
		elif idx == 1:
			return np.rot90(img)
		elif idx == 2:
			return np.rot90(img, k=2) # 180
		elif idx == 3:
			return np.rot90(img, k=3) # 270
		elif idx == 4:
			return np.flipud(img)
		elif idx == 5:
			return np.flipud(np.rot90(img))
		elif idx == 6:
			return np.flipud(np.rot90(img, k=2))
		elif idx == 7:
			return np.flipud(np.rot90(img, k=3))