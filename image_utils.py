import os
import numpy as np
import PIL.Image
import scipy.misc
from skimage import metrics

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
									   dtype=datatype).reshape(height,
																 width)
	return_matrix_ch_G = np.fromstring(infile.channels('G')[0],
									   dtype=datatype).reshape(height,
																 width)
	return_matrix_ch_R = np.fromstring(infile.channels('R')[0],
									   dtype=datatype).reshape(height,
																 width)
	matrix_new = np.stack(
		(return_matrix_ch_R, return_matrix_ch_G, return_matrix_ch_B),
		axis=-1)
	return matrix_new

def save_exr(image, filename, datatype=np.float16):
	import OpenEXR
	import Imath
	HALF  = Imath.PixelType(Imath.PixelType.HALF)
	
	data = image.astype(datatype)
	channels = {}
	channel_data = {}
	channel_name = 'B'
	channels['B'] = Imath.Channel(type=HALF)
	channel_data[channel_name] = data[:, :, 2].tostring()
	channel_name = 'G'
	channels['G'] = Imath.Channel(type=HALF)
	channel_data[channel_name] = data[:, :, 1].tostring()
	channel_name = 'R'
	channels['R'] = Imath.Channel(type=HALF)
	channel_data[channel_name] = data[:, :, 0].tostring()

	new_header = OpenEXR.Header(data.shape[1], data.shape[0])
	new_header['channels'] = channels
	out = OpenEXR.OutputFile(filename, new_header)
	out.writePixels(channel_data)

def save_image(image, filename, mode=None):
	if image.dtype in [np.float16, np.float32, np.float64]:
		image = clip_to_uint8(image)
	else:
		assert image.dtype == np.uint8
		image.astype(np.uint8)
	PIL.Image.fromarray(image, mode=mode).save(filename)

def clip_to_uint8(arr):
	return np.clip((arr) * 255.0 + 0.5, 0, 255).astype(np.uint8)

def batch_psnr(img_true, img_test):
	psnr_arr = np.zeros(img_true.shape[0])
	for idx in range(img_true.shape[0]):
		psnr_arr[idx] = metrics.peak_signal_noise_ratio(\
			img_true[idx,:,:,:], img_test[idx,:,:,:])
	return psnr_arr, np.mean(psnr_arr)

def batch_ssim(img_true, img_test, mc=True):
	ssim_arr = np.zeros(img_true.shape[0])
	for idx in range(img_true.shape[0]):
		ssim_arr[idx] = metrics.structural_similarity(\
			img_true[idx,:,:,:], img_test[idx,:,:,:], multichannel=mc)
	return ssim_arr, np.mean(ssim_arr)

def batch_mse(img_true, img_test):
	mse_arr = np.zeros(img_true.shape[0])
	for idx in range(img_true.shape[0]):
		mse_arr[idx] = metrics.mean_squared_error(\
			img_true[idx,:,:,:], img_test[idx,:,:,:])
	return mse_arr, np.mean(mse_arr)

def batch_rmse(img_true, img_test):
	rmse_arr = np.zeros(img_true.shape[0])
	for idx in range(img_true.shape[0]):
		rmse_arr[idx] = np.sqrt(metrics.mean_squared_error(\
			img_true[idx,:,:,:], img_test[idx,:,:,:]))
	return rmse_arr, np.mean(rmse_arr)

def batch_smape(img_test, img_true):		
	im_size = img_true.shape[-1] * img_true.shape[-2] * img_true.shape[-3]  
	smape_arr = 100 * np.mean((2 * np.abs(img_true - img_test) / (  
		np.abs(img_true) + np.abs(img_test) + 0.0000001)), axis=(1,2,3))	
	
	return smape_arr, np.mean(smape_arr)	
	
def batch_relative_mse(img_true, img_test): 
	rmse_map = ((img_true - img_test) ** 2) / (img_true ** 2 + 0.0000001)   
	return np.mean(rmse_map, axis=(1,2,3)), np.mean(rmse_map)   
	
def rmse_map(img_true, img_test):   
	rmse_map = ((img_true - img_test) ** 2) / (img_true ** 2 + 0.0000001)   
	return rmse_map 
	# return np.mean(rmse_map, axis=3)  
	
def ssim_map(img_true, img_test, mc=True):  
	ssim_map = np.zeros_like(img_true)  
	for idx in range(img_true.shape[0]):	
		_, ssim_im = metrics.structural_similarity(	
			img_true[idx,:,:,:], img_test[idx,:,:,:], multichannel=mc, full=True)   
		ssim_map[idx,:,:,:] = ssim_im   
	return ssim_map