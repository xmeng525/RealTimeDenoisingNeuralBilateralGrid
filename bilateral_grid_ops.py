"""
description: link to CUDA kernels of:
1. Bilateral Grid Creation
2. Bilateral Grid Slicing

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops

path = os.path.dirname(os.path.abspath(__file__))
path_grid = tf.resource_loader.get_path_to_datafile(
    os.path.join(path, 'bilateral_kernels', 'bilateral_kernels.so'))

bilateral = tf.load_op_library(path_grid)

create_bi_grid = bilateral.create_bi_grid
slice_bi_grid = bilateral.slice_bi_grid

@ops.RegisterGradient('CreateBiGrid')
def _create_bi_grid_grad(op, grad):
	image = op.inputs[0]
	guide = op.inputs[1]
	attrs = op.inputs[2]
	[grad_image, grad_guide] = bilateral.create_bi_grid_grad(image, guide, attrs, grad)
	return [grad_image, grad_guide, None]

@ops.RegisterGradient('SliceBiGrid')
def _slice_bi_grid_grad(op, grad):
	grid = op.inputs[0]
	guide = op.inputs[1]
	attrs = op.inputs[2]
	[grad_grid, grad_guide] = bilateral.slice_bi_grid_grad(grid, guide, attrs, grad)
	return [grad_grid, grad_guide, None]
