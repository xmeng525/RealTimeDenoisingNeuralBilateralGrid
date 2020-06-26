"""
description: Result Evaluation

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from image_utils import *

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset-name', type=str, default="classroom")
parser.add_argument('-ts', '--test-size', type=int, default=60)

args = parser.parse_args()
test_per_scene = args.test_size
scene_names = args.dataset_name.split(' ')

for idx_scene, scene_name in enumerate(scene_names):
	result_path = os.path.join(scene_name, 'result', 'test_out')
	save_path = os.path.join(scene_name, 'result', 'evaluations')
	os.makedirs(save_path, exist_ok=True)
	os.makedirs(os.path.join(save_path, 'rmse_maps'), exist_ok=True)
	os.makedirs(os.path.join(save_path, 'ssim_maps'), exist_ok=True)
	
	# Load the GT images
	gt_all = np.zeros((test_per_scene, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	for idx in range(test_per_scene):
		gt_all[idx,:,:,:] = plt.imread(os.path.join(result_path, '%d_tgt.png'%idx))

	# Load the denoised images
	test_all = np.zeros((test_per_scene, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
	for idx in range(test_per_scene):
		test_all[idx,:,:,:] = plt.imread(os.path.join(result_path, '%d_rcn.png'%idx))

	# Export the evaluation with different metrics
	psnr_arr, psnr = batch_psnr(gt_all, test_all)
	mse_arr, mse = batch_mse(gt_all, test_all)
	rmse_arr, rmse = batch_rmse(gt_all, test_all)
	ssim_arr, ssim = batch_ssim(gt_all, test_all)
	smape_arr, smape = batch_smape(gt_all, test_all)
	rltv_mse_arr, rltv_mse = batch_relative_mse(gt_all, test_all)

	np.savetxt(os.path.join(save_path, 'psnr.txt'), psnr_arr)
	np.savetxt(os.path.join(save_path, 'mse.txt'), mse_arr)
	np.savetxt(os.path.join(save_path, 'rmse.txt'), rmse_arr)
	np.savetxt(os.path.join(save_path, 'ssim.txt'), ssim_arr)
	np.savetxt(os.path.join(save_path, 'smape.txt'), smape_arr)
	np.savetxt(os.path.join(save_path, 'rltv_mse.txt'), rltv_mse_arr)

	print('Scene: %s\npsnr=%.4f\nssim=%.4f\nrmse=%.4f\nsmape=%.4f\nrltv_mse=%.4f\n'%
		(scene_name, psnr, ssim, rmse, smape, rltv_mse))

	# Export the errormaps (RMSE & SSIM)
	rmse_maps = rmse_map(gt_all, test_all)
	ssim_maps = ssim_map(gt_all, test_all)

	for idx in range(test_per_scene):
		save_image(rmse_maps[idx,:,:,:], 
			os.path.join(save_path, 'rmse_maps', '%d.png'%idx))
		save_image(ssim_maps[idx,:,:,:], 
			os.path.join(save_path, 'ssim_maps', '%d.png'%idx))