// outlier.cu.cc
// This program is used to remove high dynamic range spikes.
// Author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <iostream>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__device__ float tone_mapping(float r) {
  	float x = 0.0f;
  	if (r - 0.004f > x)
  		x = r - 0.004f;
    return (x*(6.2*x + 0.5))/(x*(6.2*x + 1.7) + 0.06);
}

__device__ float rgb2gray(float r, float g, float b) {
    return 0.2989f * r + 0.5870f * g + 0.1140f * b;
}

__global__ void OutlierKernel(
	int image_count,
	const float* input_image, 
	const float* input_albedo,
	const float* input_normal,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	float* output_image
) {
	
	CUDA_1D_KERNEL_LOOP(i, image_count)
	{	
		int w = i % image_width;
		int h = (i / image_width) % image_height;
		int b = (i / (image_width * image_height)) % batch_size;

		const int k_size = 3;
		const int k_half = k_size / 2;

		const int sz = channel_size;
		const int sx = channel_size * image_width;
		const int sy = channel_size * image_height * image_height;
		float curr_r = tone_mapping(input_image[i * channel_size]);
		float curr_g = tone_mapping(input_image[i * channel_size + 1]);
		float curr_b = tone_mapping(input_image[i * channel_size + 2]);
		const float gray = 0.2989f * curr_r + 0.5870f * curr_g + 0.1140f * curr_b;

		float neighbor_list[k_size * k_size];
		float neighbor_mean = 0.0f;
		
		float out_r = 0.0f;
		float out_g = 0.0f;
		float out_b = 0.0f;
		float out_count = 0.0f;
		for (int yy = 0; yy < k_size; yy++)
		{
			int y = h - k_half + yy;
			if (y < 0 || y > image_height - 1)
				continue;
			for (int xx = 0; xx < k_size; xx++)
			{
				int x = w - k_half + xx;
				if (x < 0 || x > image_width - 1)
					continue;
				if (yy == k_half && xx == k_half)
					continue;
				int idx_r = (x * sz + y * sx + b * sy);
				int idx_g = idx_r + 1;
				int idx_b = idx_r + 2;

				curr_r = tone_mapping(input_image[idx_r]);
				curr_g = tone_mapping(input_image[idx_g]);
				curr_b = tone_mapping(input_image[idx_b]);
				float nghb_gray = 0.2989f * curr_r + 0.5870f * curr_g + 0.1140f * curr_b;
				
				float albedo_dist = (input_albedo[idx_r] - input_albedo[i * channel_size]) * (input_albedo[idx_r] - input_albedo[i * channel_size]) + 
					(input_albedo[idx_g] - input_albedo[i * channel_size + 1]) * (input_albedo[idx_g] - input_albedo[i * channel_size + 1]) + 
					(input_albedo[idx_b] - input_albedo[i * channel_size + 2]) * (input_albedo[idx_b] - input_albedo[i * channel_size + 2]); 

				float normal_dist = (input_normal[idx_r] - input_normal[i * channel_size]) * (input_normal[idx_r] - input_normal[i * channel_size]) + 
					(input_normal[idx_g] - input_normal[i * channel_size + 1]) * (input_normal[idx_g] - input_normal[i * channel_size + 1]) + 
					(input_normal[idx_b] - input_normal[i * channel_size + 2]) * (input_normal[idx_b] - input_normal[i * channel_size + 2]); 
				if (albedo_dist < 0.1f && normal_dist < 0.1f)
				{
					neighbor_list[xx + yy * k_size] = nghb_gray;
					neighbor_mean += nghb_gray;
                    out_r += input_image[idx_r];
                    out_g += input_image[idx_g];
                    out_b += input_image[idx_b];
                    out_count += 1.0f;
				}
			}
		}
		if (out_count > 0.0f)
		{
			neighbor_mean /= out_count;
			float neighbor_std = 0.0f;
			for (int k = 0; k < k_size * k_size; k++)
			{
				if (neighbor_list[k] > 0.0f)
				{
					neighbor_std += (neighbor_list[k] - neighbor_mean) * (neighbor_list[k] - neighbor_mean);
				}
			}
			neighbor_std = sqrt(neighbor_std / (out_count - 1.0f + 1e-8f));
			if (abs(gray - neighbor_mean) > 3.0f * neighbor_std)
			{
                output_image[i * channel_size + 0] = out_r / out_count;
                output_image[i * channel_size + 1] = out_g / out_count;
                output_image[i * channel_size + 2] = out_b / out_count;
			}
			else
			{
				output_image[i * channel_size + 0] = input_image[i * channel_size + 0];
		        output_image[i * channel_size + 1] = input_image[i * channel_size + 1];
		        output_image[i * channel_size + 2] = input_image[i * channel_size + 2];
			}
		}
		else
		{
			output_image[i * channel_size + 0] = input_image[i * channel_size + 0];
	        output_image[i * channel_size + 1] = input_image[i * channel_size + 1];
	        output_image[i * channel_size + 2] = input_image[i * channel_size + 2];
		}
		
	}
}

bool OutlierKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const float* input_albedo,
	const float* input_normal,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	float* output_image
) {
	int64 image_count = batch_size * image_height * image_width;
	if (image_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(image_count, d);
		OutlierKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			image_count,
			input_image,
			input_albedo,
			input_normal,
			batch_size,
			image_height,
			image_width,
			channel_size,
			output_image);
	}
	return d.ok();
}
 
#endif
