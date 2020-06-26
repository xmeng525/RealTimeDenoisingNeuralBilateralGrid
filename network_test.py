"""
description: Neural Bilateral Grid Testing

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
@author: QZheng
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from data_loader import dataLoader
from network import DenoiserGuideNet
from image_utils import save_image, save_exr

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

INPUT_CHANNEL = 10
TARGET_CHANNEL = 3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset-name', type=str, 
	default="classroom")
parser.add_argument('-r', '--data-dir', type=str, 
	default='../../dataset/dataset_blockwise/BMFR-dataset')
parser.add_argument('-bs', '--batch-size', type=int, default=1)
parser.add_argument('-ts', '--test-size', type=int, default=60)
parser.add_argument('--export_exr',action='store_true')
parser.add_argument('--export_grid_output',action='store_true')
parser.add_argument('--export_guide_weight',action='store_true')
parser.add_argument('--export_grid',action='store_true')
args = parser.parse_args()

data_dir = args.data_dir
scene_test_list = args.dataset_name.split(' ')
test_batch_size = args.batch_size
test_per_scene = args.test_size

def tone_mapping(input_image):
	tone_mapped_color = np.clip(
		np.power(np.maximum(0., input_image), 0.454545), 0., 1.)
	return tone_mapped_color

def export_grid(result_dir, batch_cnt, grid, name):
	batch, height, width, depth, channel = grid.shape
	for b in range(batch):
		for d in range(depth):
			save_image(np.squeeze(grid[b, :, :, d, :]), 
				os.path.join(result_dir, name + '_%d_%d_%d.png' %(batch_cnt, b, d)), 'RGB')

def _parse_function_testdata(proto):
	features = tf.parse_single_example(
		proto, features={
			'target': tf.FixedLenFeature([], tf.string),
			'input': tf.FixedLenFeature([], tf.string)})

	train_input = tf.decode_raw(features['input'], tf.float16)
	train_input = tf.reshape(train_input, [IMAGE_HEIGHT,
										   IMAGE_WIDTH, INPUT_CHANNEL])

	train_target = tf.decode_raw(features['target'], tf.float16)
	train_target = tf.reshape(train_target, [IMAGE_HEIGHT,
											 IMAGE_WIDTH, TARGET_CHANNEL])
	return (train_input, train_target)

if __name__ == "__main__":
	model_dir = os.path.join(scene_test_list[0], 'model')
	result_dir = os.path.join(scene_test_list[0], 'result', 'test_out')
	errorlog_dir = os.path.join(scene_test_list[0], 'errorlog')
	summarylog_dir = os.path.join(scene_test_list[0], 'summarylog')

	os.makedirs(model_dir, exist_ok=True)
	os.makedirs(result_dir, exist_ok=True)
	os.makedirs(errorlog_dir, exist_ok=True)
	os.makedirs(summarylog_dir, exist_ok=True)

	test_data = dataLoader(data_dir=data_dir, subset='test',
						   image_start_idx=0,
						   img_per_scene=test_per_scene,
						   scene_list=scene_test_list)

	# Test
	test_dataset = tf.data.TFRecordDataset([test_data.dataset_name])
	test_dataset = test_dataset.map(_parse_function_testdata)
	test_dataset = test_dataset.batch(test_batch_size)

	handle_large = tf.placeholder(tf.string, shape=[])
	iterator_structure_large = tf.data.Iterator.from_string_handle(
		 handle_large, test_dataset.output_types, test_dataset.output_shapes)
	next_element_large = iterator_structure_large.get_next()
	test_iterator = test_dataset.make_initializable_iterator()

	# Model
	model = DenoiserGuideNet(input_shape=[None, None, None, INPUT_CHANNEL],
		target_shape=[None, None, None, TARGET_CHANNEL])
	with tf.device("/gpu:0"):
		guide_net = model.inference()

	saver = tf.train.Saver()
	config = tf.ConfigProto(allow_soft_placement=True, graph_options=tf.GraphOptions(
		optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	run_metadata = tf.RunMetadata()
	run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	opts1 = tf.profiler.ProfileOptionBuilder.float_operation()
	flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts1)

	opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
	params = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts2)

	sess.run(tf.global_variables_initializer())
	saver.restore(sess, os.path.join(model_dir, 'best_model'))
	
	# Test
	print('Start Testing...')
	batch_cnt = 0
	sess.run(test_iterator.initializer)
	test_handle = sess.run(test_iterator.string_handle())

	while True:
		try:
			src_hdr, tgt_hdr = sess.run(next_element_large,
				feed_dict={handle_large: test_handle})
			feed_dict = {guide_net['source']: src_hdr, guide_net['target']: tgt_hdr}
			
			guide_1, guide_2, guide_3, weight_1, weight_2, denoised_hdr, \
			from_grid_hdr_1, from_grid_hdr_2, from_grid_hdr_3, \
			grid_1, grid_2, grid_3, batch_loss = sess.run(
				[guide_net['guide_1'], guide_net['guide_2'], guide_net['guide_3'], \
				guide_net['weight_1'], guide_net['weight_2'], guide_net['denoised_hdr'],\
				guide_net['from_grid_hdr_1'], guide_net['from_grid_hdr_2'], guide_net['from_grid_hdr_3'],
				guide_net['grid_1'],guide_net['grid_2'],guide_net['grid_3'], guide_net['loss_all_L1']], feed_dict,
				options=run_options, run_metadata=run_metadata)

			tl = timeline.Timeline(run_metadata.step_stats)
			ctf = tl.generate_chrome_trace_format(show_memory=True)
			with open(os.path.join(errorlog_dir, 'timeline.json'),'w') as wd:
				wd.write(ctf)

			for k in range(0, src_hdr.shape[0]): 
				idx_all = batch_cnt * test_batch_size + k
				save_image(tone_mapping(src_hdr[k,:,:,0:3]), 
					os.path.join(result_dir, '%d_src.png'%idx_all), 'RGB')
				save_image(tone_mapping(tgt_hdr[k,:,:,:]), 
					os.path.join(result_dir, '%d_tgt.png'%idx_all), 'RGB')
				save_image(tone_mapping(denoised_hdr[k,:,:,:]), 
					os.path.join(result_dir, '%d_rcn.png'%idx_all), 'RGB')

				if args.export_exr:
					save_exr(src_hdr[k,:,:,0:3], 
						os.path.join(result_dir, '%d_src.exr'%idx_all))
					save_exr(tgt_hdr[k,:,:,:], 
						os.path.join(result_dir, '%d_tgt.exr'%idx_all))
					save_exr(denoised_hdr[k,:,:,:], 
						os.path.join(result_dir, '%d_rcn.exr'%idx_all))
				if args.export_grid_output:
					save_image(tone_mapping(from_grid_hdr_1[k,:,:,:]), 
						os.path.join(result_dir, '%d_from_grid_1.png'%idx_all), 'RGB')
					save_image(tone_mapping(from_grid_hdr_2[k,:,:,:]), 
						os.path.join(result_dir, '%d_from_grid_2.png'%idx_all), 'RGB')
					save_image(tone_mapping(from_grid_hdr_3[k,:,:,:]), 
						os.path.join(result_dir, '%d_from_grid_3.png'%idx_all), 'RGB')
				if args.export_guide_weight:
					save_image(guide_1[k, :, :], 
						os.path.join(result_dir, '%d_guide_1.png'%idx_all))
					save_image(guide_2[k, :, :],
						os.path.join(result_dir, '%d_guide_1.png'%idx_all))
					save_image(guide_3[k, :, :],
						os.path.join(result_dir, '%d_guide_1.png'%idx_all))
					save_image(weight_1[k,:, :],
						os.path.join(result_dir, '%d_weight_1.png'%idx_all))
					save_image(weight_2[k,:, :],
						os.path.join(result_dir, '%d_weight_2.png'%idx_all))
					save_image(1 - weight_1[k,:, :] - weight_2[k,:, :],
						os.path.join(result_dir, '%d_weight_3.png'%idx_all))
			if args.export_grid:
				export_grid(result_dir, batch_cnt, grid_1, 'grid_1')
				export_grid(result_dir, batch_cnt, grid_2, 'grid_2')
				export_grid(result_dir, batch_cnt, grid_3, 'grid_3')
			batch_cnt += 1
		except tf.errors.OutOfRangeError:
			print('Finish testing %d images.' % (batch_cnt * test_batch_size))
			break
	sess.close()



