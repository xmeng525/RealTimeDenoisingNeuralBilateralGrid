// create_bi_grid.cu.cc
// This program is used to create a bilateral grid from an 2D input.
// Author: Xiaoxu Meng

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <iostream>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__device__ float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x*x+eps);
}

__device__ float d_diff_abs(float x) {
  float eps = 1e-8;
  return x/sqrt(x*x+eps);
}

__device__ float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0f-abx, 0.0f);
}

__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}

__global__ void CreateBiGridKernel(
	int grid_count,
	const float* input_image, 
	const float* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* out_color
) {
	
	CUDA_1D_KERNEL_LOOP(i, grid_count)
	{	
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int w = (i / channel_size) % grid_width;
		int h = (i / (channel_size * grid_width)) % grid_height;
		int b = (i / (channel_size * grid_width * grid_height)) % batch_size;

		float f_w = w + 0.5f;
		float f_h = h + 0.5f;

		int left = static_cast<int>(ceil((w - 0.5f) * ss));
		int right = left + 2 * ss;
		int up = static_cast<int>(ceil((h - 0.5f) * ss));
		int down = up + 2 * ss;

		int sx = channel_size;
		int sy = channel_size * image_width;
		int sb = channel_size * image_width * image_height;

		float sum_color[64];
		float sum_count[64];
		for (int dd = 0; dd < grid_depth; dd++)
		{
			sum_color[dd] = 0.0f;
			sum_count[dd] = 0.0f;
		}

		for (int yy = up; yy < down; yy++)
		{
			if (yy < 0 || yy > image_height - 1)
				continue;
			float wy = weight_z((yy + 0.5f) / ss - f_h);
			for (int xx = left; xx < right; xx++)
			{
				if (xx < 0 || xx > image_width - 1)
					continue;
				float wx = weight_z((xx + 0.5f) / ss - f_w);

				int idx_guide = xx + yy * image_width + b * image_width * image_height;
				int idx_image = c + xx * sx + yy * sy + b * sb;
				
				float guide_val = (input_guide[idx_guide] + 0.5f) / sr;
				int d_lower = static_cast<int>(max(ceil(guide_val - 1.5f), 0.0f));
				int d_upper = static_cast<int>(min(floor(guide_val + 0.5f), grid_depth - 1.0f));
				for (int zz = d_lower; zz <= d_upper; zz++)
				{
					float wz = weight_z(guide_val - zz - 0.5f);
					float wt = wy * wx * wz;
					sum_color[zz] += input_image[idx_image] * wt;
					sum_count[zz] += wt;
				}
			}
		}
		int idx_base = w * channel_size * grid_depth + h * channel_size * grid_depth * grid_width + b * channel_size * grid_depth * grid_width * grid_height;
		for (int dd = 0; dd < grid_depth; dd++)
		{
			int idx_grid = c + dd * sx + idx_base;
			out_color[idx_grid] = abs(sum_count[dd]) > 0 ? sum_color[dd] / sum_count[dd]: 0.0f;
		}
	}
}

__global__ void CreateBiGridImageGradKernel(
	int image_count,
	const float* input_image, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* image_grad
) {
	CUDA_1D_KERNEL_LOOP(i, image_count)
	{
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int im_w = (i / channel_size) % image_width;
		int im_h = (i / (channel_size * image_width)) % image_height;
		int b = (i / (channel_size * image_width * image_height)) % batch_size;
		
		int idx_guide =  im_w + im_h * image_width + b * image_width * image_height;
		
		float f_gr_d = (input_guide[idx_guide] + 0.5f) / sr;
		float f_gr_w = (im_w + 0.5f) / ss;
		float f_gr_h = (im_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(im_w * 1.0 / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(im_h * 1.0 / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb = channel_size * grid_depth * grid_width * grid_height;

		float sum_color = 0.0f;
		for (int vv = up; vv <= down; vv++)
		{
			if (vv < 0 || vv > grid_height - 1)
				continue;
			float wy = weight_z((vv + 0.5f) - f_gr_h);
			for (int uu = left; uu <= right; uu++)
			{
				if (uu < 0 || uu > grid_width - 1)
					continue;
				float wx = weight_z((uu + 0.5f) - f_gr_w);
				for (int ww = front; ww <= back; ww++)
				{
					if (ww < 0 || ww > grid_depth - 1)
						continue;
					float wz = weight_z((ww + 0.5f) - f_gr_d);
					float wt = wy * wx * wz;
					
					float part2 = 0;
					int left_xx = static_cast<int>(ceil((uu - 0.5f) * ss));
					int right_xx = left_xx + 2 * ss + 2;
					int up_yy = static_cast<int>(ceil((vv - 0.5f) * ss));
					int down_yy = up_yy + 2 * ss + 2;

					for (int yy = up_yy; yy < down_yy; yy++)
					{
						if (yy < 0 || yy > image_height - 1)
							continue;
						for (int xx = left_xx; xx < right_xx; xx++)
						{
							if (xx < 0 || xx > image_width - 1)
								continue;
							int idx_guide_in = xx + yy * image_width + b * image_width * image_height;
							float f_xx = (xx + 0.5f) / ss;
							float f_yy = (yy + 0.5f) / ss;
							float f_zz = (input_guide[idx_guide_in] + 0.5f) / sr;

							float wyy = weight_z((vv + 0.5f) - f_yy);
							float wxx = weight_z((uu + 0.5f) - f_xx);
							float wzz = weight_z((ww + 0.5f) - f_zz);
							part2 += wyy * wxx * wzz;
						}
					}
					int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb;
					// printf("i=%d, vv=%d, uu=%d, ww=%d, lt=%d, rt=%d, up=%d, dwn=%d\n", i, vv, uu, ww, left_xx, right_xx, up_yy,down_yy);
					if (abs(part2) > 0)
						sum_color += backprop[idx_grid] * wt / part2;
				}
			}
		}
		if (abs(sum_color) > 100)
		{
			printf("Create::Big Image Grad = %.4f\n", sum_color);
			image_grad[i] = 0;
		}
		else
		{
			image_grad[i] = sum_color;
		}
	}
}

__global__ void CreateBiGridGuideGradKernel(
	int guide_count,
	const float* input_image, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* guide_grad
) {
	CUDA_1D_KERNEL_LOOP(i, guide_count)
	{
		int ss = image_width / grid_width;

		int gd_w = i % image_width;
		int gd_h = (i / image_width) % image_height;
		int b = (i / (image_width * image_height)) % batch_size;
		
		float f_gr_d = (input_guide[i] + 0.5f) / sr;
		float f_gr_w = (gd_w + 0.5f) / ss;
		float f_gr_h = (gd_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[i] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(gd_w * 1.0 / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(gd_h * 1.0 / ss - 0.5f));
		int down = up + 1;

		// printf("gd_h=%d,gd_w=%d,front=%d,left=%d, up=%d\n", gd_h, gd_w, front, left, up);
		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

		int sx = channel_size;
		int sy = channel_size * image_width;
		int sb_xy = channel_size * image_width * image_height;

		float sum_color = 0.0f;
		for (int cc = 0; cc < channel_size; cc++)
		{
			float chan_color = 0.0f;
			int idx_image = cc + gd_w * sx + gd_h * sy + b * sb_xy;
			for (int vv = up; vv <= down; vv++)
			{
				if (vv < 0 || vv > grid_height - 1)
					continue;
				for (int uu = left; uu <= right; uu++)
				{
					if (uu < 0 || uu > grid_width - 1)
						continue;
					for (int ww = front; ww <= back; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						int left_xx = static_cast<int>(ceil((uu - 0.5f) * ss));
						int right_xx = left_xx + 2 * ss + 1;
						int up_yy = static_cast<int>(ceil((vv - 0.5f) * ss));
						int down_yy = up_yy + 2 * ss + 1;

						// printf("%d,%d,%d =%d\n", vv, uu, ww, left_xx);
						
						float wx = weight_z((uu + 0.5f) - f_gr_w);
						float wy = weight_z((vv + 0.5f) - f_gr_h);
						float dwz = d_weight_z((ww + 0.5f) - f_gr_d) / sr;
						
						float part1 = wy * wx * dwz * input_image[idx_image];
						float part3 = wy * wx * dwz;
						float part2 = 0.0f;
						float part4 = 0.0f;

						for (int xx = left_xx; xx < right_xx; xx++)
						{
							if (xx < 0 || xx > image_width - 1)
								continue;
							for (int yy = up_yy; yy < down_yy; yy++)
							{
								if (yy < 0 || yy > image_height - 1)
									continue;
								int i_guide = xx + image_width * yy + b * image_width * image_height;
								int i_image = cc + xx * sx + yy * sy + b * sb_xy;

								float f_xx = (xx + 0.5f) / ss;
								float f_yy = (yy + 0.5f) / ss;
								float f_zz = (input_guide[i_guide] + 0.5f) / sr;

								float wyy = weight_z((vv + 0.5f) - f_yy);
								float wxx = weight_z((uu + 0.5f) - f_xx);
								float wzz = weight_z((ww + 0.5f) - f_zz);

								float wtt = wyy * wxx * wzz;
								part2 += wtt;
								part4 += wtt * input_image[i_image];
							}
						}
						int idx_grid = cc + ww * sw + uu * su + vv * sv + b * sb_uvw;

						float derv = 0;
						if (abs(part2) > 0)
							derv = (part1 * part2 - part3 * part4) / (part2 * part2);
						chan_color += backprop[idx_grid] * derv;
						// printf("%d,%d,%d,%d, %.4f\n", i, vv, uu, ww, part1);
					}
				}
			}
			sum_color += chan_color;
		}
		if (abs(sum_color) > 100)
		{
			printf("Create::Big Guide Grad = %.4f\n", sum_color);
			guide_grad[i] = 0;
		}
		else
		{
			guide_grad[i] = sum_color;
		}
	}
}

__global__ void SliceBiGridKernel(
	int image_count,
	const float* input_grid,
	const float* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* output_image
) {
	
	CUDA_1D_KERNEL_LOOP(i, image_count)
	{	
		int ss = image_width / grid_width;

		int c = i % channel_size;
		int im_w = (i / channel_size) % image_width;
		int im_h = (i / (channel_size * image_width)) % image_height;
		int b = (i / (channel_size * image_width * image_height)) % batch_size;
		
		int idx_guide =  im_w + im_h * image_width + b * image_width * image_height;
		
		float f_gr_d = (input_guide[idx_guide] + 0.5f) / sr;
		float f_gr_w = (im_w + 0.5f) / ss;
		float f_gr_h = (im_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(im_w * 1.0f / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(im_h * 1.0f / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb = channel_size * grid_depth * grid_width * grid_height;

		float sum_color = 0.0f;
		float sum_count = 0.0f;
		for (int vv = up; vv <= down; vv++)
		{
			if (vv < 0 || vv > grid_height - 1)
				continue;
			float wy = weight_z((vv + 0.5f) - f_gr_h);
			for (int uu = left; uu <= right; uu++)
			{
				if (uu < 0 || uu > grid_width - 1)
					continue;
				float wx = weight_z((uu + 0.5f) - f_gr_w);
				for (int ww = front - 2; ww < back + 2; ww++)
				{
					if (ww < 0 || ww > grid_depth - 1)
						continue;
					float wz = weight_z((ww + 0.5f) - f_gr_d);
					int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb;
					if (input_grid[idx_grid] > 0.0f)
					{
						float wt =  wy * wx * wz;
						sum_color += input_grid[idx_grid] * wt;
						sum_count += wt;
					}
				}
			}
		}
		output_image[i] = abs(sum_count) > 0.0f ? sum_color / sum_count: 0.0f;
	}
}

__global__ void SliceBiGridGridGradKernel(
	int grid_count,
	const float* input_grid,
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const grid_grad
) {
	CUDA_1D_KERNEL_LOOP(i, grid_count)
	{
		float sum_color = 0;
		if (input_grid[i] > 0.0f)
		{
			int ss = image_width / grid_width;
			int c = i % channel_size;
			int d = (i / channel_size) % grid_depth;
			int w = (i / (channel_size * grid_depth)) % grid_width;
			int h = (i / (channel_size * grid_depth * grid_width)) % grid_height;
			int b = (i / (channel_size * grid_depth * grid_width * grid_height)) % batch_size;

			float f_w = w + 0.5f;
			float f_h = h + 0.5f;
			float f_d = d + 0.5f;

			int left = static_cast<int>(ceil((w - 0.5f) * ss));
			int right = left + 2 * ss;
			int up = static_cast<int>(ceil((h - 0.5f) * ss));
			int down = up + 2 * ss;

			// printf("h=%d, w=%d, d=%d, c=%d, left=%d, right=%d, up=%d, down=%d\n",
			// 			h,w,d,c,left,right, up,down);
			int sw = channel_size;
			int su = channel_size * grid_depth;
			int sv = channel_size * grid_depth * grid_width;
			int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

			int sx = channel_size;
			int sy = channel_size * image_width;
			int sb_xy = channel_size * image_width * image_height;

			for (int yy = up; yy < down; yy++)
			{
				if (yy < 0 || yy > image_height - 1)
					continue;
				for (int xx = left; xx < right; xx++)
				{
					if (xx < 0 || xx > image_width - 1)
						continue;
					int idx_guide = xx + yy * image_width + b * image_width * image_height;
					int idx_image = c + xx * sx + yy * sy + b * sb_xy;

					float wy = weight_z((yy + 0.5f) / ss - f_h);
					float wx = weight_z((xx + 0.5f) / ss - f_w);
					float wz = weight_z((input_guide[idx_guide] + 0.5f) / sr - f_d);
					float wt =  wy * wx * wz;

					// printf("h=%d, w=%d, d=%d, c=%d, yy=%d, xx=%d, wy=%.4f, wx=%.4f, wz=%.4f\n",
					// 	h,w,d,c,yy,xx, wy,wx,wz);
					int front_ww = static_cast<int>(floor(input_guide[idx_guide] / sr - 0.5f));
					int back_ww = front_ww + 1;

					int left_uu = static_cast<int>(floor(xx * 1.0f / ss - 0.5f));
					int right_uu = left_uu + 1;

					int up_vv = static_cast<int>(floor(yy * 1.0f / ss - 0.5f));
					int down_vv = up_vv + 1;

					float part2 = 0.0f;
					for (int ww = front_ww; ww <= back_ww; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						for(int vv = up_vv; vv <= down_vv; vv++)
						{
							if (vv < 0 || vv > grid_height - 1)
								continue;
							for (int uu = left_uu; uu <= right_uu; uu++)
							{
								if (uu < 0 || uu > grid_width - 1)
									continue;
								float f_uu = uu + 0.5f;
								float f_vv = vv + 0.5f;
								float f_ww = ww + 0.5f;

								float wyy = weight_z((yy + 0.5f) / ss - f_vv);
								float wxx = weight_z((xx + 0.5f) / ss - f_uu);
								float wzz = weight_z((input_guide[idx_guide] + 0.5f) / sr - f_ww);
								
								int idx_grid = c + ww * sw + uu * su + vv * sv + b * sb_uvw;
								if (input_grid[idx_grid] > 0.0f)
									part2 += wyy * wxx * wzz;
							}
						}
					}
					// printf("h=%d, w=%d, d=%d, c=%d, yy=%d, xx=%d, p1=%.4f, p2=%.4f\n",
					// 	h,w,d,c,yy,xx, wt, part2);
					if (abs(part2) > 0)
					{
						sum_color += backprop[idx_image] * wt / part2;
					}
				}
			}
			if (abs(sum_color) > 100)
			{
				printf("Slice::Big Grid Grad = %.4f\n", sum_color);
				grid_grad[i] = 0;
			}
			else
			{
				grid_grad[i] = sum_color;
			}
		} else {
			grid_grad[i] = 0;
		}
	}
}

__global__ void SliceBiGridGuideGradKernel(
	int guide_count,
	const float* input_grid, 
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* guide_grad
) {
	CUDA_1D_KERNEL_LOOP(i, guide_count)
	{
		int ss = image_width / grid_width;

		int gd_w = i % image_width;
		int gd_h = (i / image_width) % image_height;
		int b = (i / (image_width * image_height)) % batch_size;
				
		float f_gr_d = (input_guide[i] + 0.5f) / sr;
		float f_gr_w = (gd_w + 0.5f) / ss;
		float f_gr_h = (gd_h + 0.5f) / ss;

		int front = static_cast<int>(floor(input_guide[i] / sr - 0.5f));
		int back = front + 1;

		int left = static_cast<int>(floor(gd_w * 1.0f / ss - 0.5f));
		int right = left + 1;

		int up = static_cast<int>(floor(gd_h * 1.0f / ss - 0.5f));
		int down = up + 1;

		int sw = channel_size;
		int su = channel_size * grid_depth;
		int sv = channel_size * grid_depth * grid_width;
		int sb_uvw = channel_size * grid_depth * grid_width * grid_height;

		int sx = channel_size;
		int sy = channel_size * image_width;
		int sb_xy = channel_size * image_width * image_height;
		
		float sum_color = 0.0f;
		for (int cc = 0; cc < channel_size; cc++)
		{
			float part1 = 0.0f;
			float part2 = 0.0f;
			float part3 = 0.0f;
			float part4 = 0.0f;
			for (int vv = up; vv <= down; vv++)
			{
				if (vv < 0 || vv > grid_height - 1)
					continue;
				for (int uu = left; uu <= right; uu++)
				{
					if (uu < 0 || uu > grid_width - 1)
						continue;
					for (int ww = 0; ww < grid_depth; ww++)
					{
						if (ww < 0 || ww > grid_depth - 1)
							continue;
						float wy = weight_z((vv + 0.5f) - f_gr_h);
						float wx = weight_z((uu + 0.5f) - f_gr_w);
						float wz = weight_z((ww + 0.5f) - f_gr_d);
						float dwz = d_weight_z((ww + 0.5f) - f_gr_d) / sr;

						int idx_grid = cc + ww * sw + uu * su + vv * sv + b * sb_uvw;

						if (input_grid[idx_grid] > 0.0f)
						{
							float dwt = wy * wx * dwz;
							float wt = wy * wx * wz;
							part1 += dwt * input_grid[idx_grid];
							part2 += wt;
							part3 += dwt;
							part4 += wt * input_grid[idx_grid];
						}
					}
				}
			}
			int idx_image = cc + gd_w * sx + gd_h * sy + b * sb_xy;
			float derv = 0;
			if (abs(part2) > 0)
			{
				derv = (part1 * part2 - part3 * part4) / (part2 * part2);
			}
			sum_color += derv * backprop[idx_image];
		}
		if (abs(sum_color) > 100)
		{
			printf("Slice::Big Guide Grad = %.4f\n", sum_color);
			guide_grad[i] = 0;
		}
		else
		{
			guide_grad[i] = sum_color;
		}
	}
}

bool CreateBiGridKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const float* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* out_color
) {
	int64 grid_count = batch_size * grid_height * grid_width * channel_size;
	if (grid_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
		
		CreateBiGridKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			grid_count,
			input_image,
			input_guide,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth, 
			sr,
			out_color);
	}
	return d.ok();
}

bool CreateBiGridGradKernelLauncher(
	const GPUDevice& d,
	const float* input_image,
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* image_grad,
	float* guide_grad
) {
	int64 image_count = batch_size * image_height * image_width * channel_size;
	if (image_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(image_count, d);
		CreateBiGridImageGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			image_count,
			input_image,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth, 
			sr,
			image_grad);
	}
	
	int64 guide_count = batch_size * image_height * image_width;
	if (guide_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(guide_count, d);
		CreateBiGridGuideGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			guide_count,
			input_image,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth, 
			sr,
			guide_grad);
	}
	return d.ok();
} 

bool SliceBiGridKernelLauncher(
	const GPUDevice& d,
	const float* input_grid,
	const float* input_guide,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const output_image
) {
	int64 image_count = batch_size * image_height * image_width * channel_size;
	if (image_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(image_count, d);
		SliceBiGridKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			image_count,
			input_grid,
			input_guide,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth, 
			sr,
			output_image);
	}
	return d.ok();
}

bool SliceBiGridGradKernelLauncher(
	const GPUDevice& d,
	const float* input_grid,
	const float* input_guide,
	const float* backprop,
	const int batch_size,
	const int image_height,
	const int image_width,
	const int channel_size,
	const int grid_height, 
	const int grid_width, 
	const int grid_depth,
	const int sr,
	float* const grid_grad,
	float* const guide_grad
) {
	int64 grid_count = batch_size * grid_height * grid_width * grid_depth * channel_size;
	if (grid_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(grid_count, d);
		SliceBiGridGridGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			grid_count,
			input_grid,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth, 
			sr,
			grid_grad);
	}
	
	int64 guide_count = batch_size * image_height * image_width;
	if (guide_count > 0) {
		CudaLaunchConfig config = GetCudaLaunchConfig(guide_count, d);
		SliceBiGridGuideGradKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
			guide_count,
			input_grid,
			input_guide,
			backprop,
			batch_size,
			image_height,
			image_width,
			channel_size,
			grid_height, 
			grid_width, 
			grid_depth,
			sr,
			guide_grad);
	}
	return d.ok();
} 
#endif
