// create_bi_grid.cc
// This program is used to create a bilateral grid from an 2D input.
// Author: Xiaoxu Meng

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
using namespace tensorflow;  // NOLINT(build/namespaces)

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("CreateBiGrid")
    .Input("input_image: float")
	  .Input("input_guide: float")
    .Input("input_attrs: int32")
    .Output("output_color_batch: float")
    .Doc(R"doc(
Transform a 2D tensor (float) to 3D bilateral grid (float).
)doc");

REGISTER_OP("CreateBiGridGrad")
    .Input("input_image: float")
    .Input("input_guide: float")  
    .Input("input_attrs: int32")
    .Input("backprop: float")
    .Output("output_image_grad: float")
    .Output("output_guide_grad: float");

REGISTER_OP("SliceBiGrid")
    .Input("input_grid: float")
    .Input("input_guide: float")
    .Input("input_attrs: int32")
    .Output("output_image: float")
    .Doc(R"doc(
Transform a 2D tensor (float) to 3D bilateral grid (float).
)doc");

REGISTER_OP("SliceBiGridGrad")
    .Input("input_grid: float")
    .Input("input_guide: float")
    .Input("input_attrs: int32")
    .Input("backprop: float")
    .Output("output_grid_grad: float")
    .Output("output_guide_grad: float");

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
  float* const out_color
);

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
	float* const image_grad,
	float* const guide_grad
);

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
);

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
);

class CreateBiGridOp : public OpKernel {
 public:
  explicit CreateBiGridOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_image_tensor = context->input(0);
    // Check the input dimension.
    if (input_image_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_image = input_image_tensor.flat<float>();
	
    // Grab the input guide
    const Tensor& input_guide_tensor = context->input(1);
    // Check the input guide dimension.
    if (input_guide_tensor.shape().dims() != 3)
      throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide = input_guide_tensor.flat<float>();

    // Grab ss, sr
    const Tensor& input_attrs_tensor = context->input(2);
    // Check the input guide dimension.
    if (input_attrs_tensor.shape().dims() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs_tensor.shape().dim_size(0);
    const int sr = input_attrs_tensor.shape().dim_size(1);

    const int batch_size = input_image_tensor.shape().dim_size(0);
    const int image_height = input_image_tensor.shape().dim_size(1);
    const int image_width = input_image_tensor.shape().dim_size(2);
    const int channel_size = input_image_tensor.shape().dim_size(3);

    const int grid_height = image_height / ss;
    const int grid_width = image_width / ss;
    const int grid_depth = 256 / sr;

    // Create an output tensor
    Tensor* output_color_tensor = nullptr;
    Tensor* output_count_tensor = nullptr;

    TensorShape output_shape({batch_size, grid_height, grid_width, grid_depth, channel_size});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_color_tensor));
    auto output_color = output_color_tensor->template flat<float>();

    // Call the cuda kernel launcher
    CreateBiGridKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_image.data(), 
	    input_guide.data(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      grid_height, 
      grid_width, 
      grid_depth, 
      sr,
      output_color.data()
    );
  }
};

class CreateBiGridGradOp : public OpKernel {
 public:
  explicit CreateBiGridGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_image_tensor = context->input(0);
    // Check the input dimension.
    if (input_image_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_image = input_image_tensor.flat<float>();
	
    // Grab the input guide
    const Tensor& input_guide_tensor = context->input(1);
    // Check the input guide dimension.
    if (input_guide_tensor.shape().dims() != 3)
      throw std::invalid_argument("Error: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide = input_guide_tensor.flat<float>();

    // Grab ss, sr
    const Tensor& input_attrs_tensor = context->input(2);
    // Check the input guide dimension.
    if (input_attrs_tensor.shape().dims() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs_tensor.shape().dim_size(0);
    const int sr = input_attrs_tensor.shape().dim_size(1);
    // Grab the input guide
    const Tensor& backprop_tensor = context->input(3);
    auto backprop = backprop_tensor.flat<float>();
	
    const int batch_size = input_image_tensor.shape().dim_size(0);
    const int image_height = input_image_tensor.shape().dim_size(1);
    const int image_width = input_image_tensor.shape().dim_size(2);
    const int channel_size = input_image_tensor.shape().dim_size(3);

    const int grid_height = image_height / ss;
    const int grid_width = image_width / ss;
    const int grid_depth = 256 / sr;

    // Create an output tensor
    Tensor* image_grad_tensor = nullptr;
    Tensor* guide_grad_tensor = nullptr;

    TensorShape image_grad_shape({batch_size, image_height, image_width, channel_size});
	  TensorShape guide_grad_shape({batch_size, image_height, image_width});
	
    OP_REQUIRES_OK(context, context->allocate_output(0, image_grad_shape,
                                                     &image_grad_tensor));
    auto image_grad = image_grad_tensor->template flat<float>();

    OP_REQUIRES_OK(context, context->allocate_output(1, guide_grad_shape,
                                                     &guide_grad_tensor));
    auto guide_grad = guide_grad_tensor->template flat<float>();
   
    // Call the cuda kernel launcher
    CreateBiGridGradKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_image.data(), 
      input_guide.data(), 
      backprop.data(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      grid_height, 
      grid_width, 
      grid_depth, 
      sr,
      image_grad.data(),
      guide_grad.data()
    );
  }
};

class SliceBiGridOp : public OpKernel {
 public:
  explicit SliceBiGridOp(OpKernelConstruction* context) : OpKernel(context) {  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_grid_tensor = context->input(0);
    // Check the input dimension.
    if (input_grid_tensor.shape().dims() != 5)
      throw std::invalid_argument("Error in Slicing: Input grid dimension should be 5: {batch, height, width, depth, channel}.");
    auto input_grid = input_grid_tensor.flat<float>();
  
    // Grab the input guide
    const Tensor& input_guide_tensor = context->input(1);
    // Check the input guide dimension.
    if (input_guide_tensor.shape().dims() != 3)
      throw std::invalid_argument("Error in Slicing: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide = input_guide_tensor.flat<float>();

    // Grab ss, sr
    const Tensor& input_attrs_tensor = context->input(2);
    // Check the input guide dimension.
    if (input_attrs_tensor.shape().dims() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 2: [ss, sr].");
    const int ss = input_attrs_tensor.shape().dim_size(0);
    const int sr = input_attrs_tensor.shape().dim_size(1);

    const int batch_size = input_grid_tensor.shape().dim_size(0);
    const int grid_height = input_grid_tensor.shape().dim_size(1);
    const int grid_width = input_grid_tensor.shape().dim_size(2);
    const int grid_depth = input_grid_tensor.shape().dim_size(3);
    const int channel_size = input_grid_tensor.shape().dim_size(4);

    // std::cout << "grid shape = " << batch_size <<", "<< grid_height<<", "<< grid_width<<", "<< grid_depth<<", "<< channel_size<< std::endl;
    const int image_height = input_guide_tensor.shape().dim_size(1);
    const int image_width = input_guide_tensor.shape().dim_size(2);
    // std::cout << "guide shape = " << input_guide_tensor.shape().dim_size(0) <<", "<< image_height <<", "<< image_width << std::endl;
    // Create an output tensor
    Tensor* output_image_tensor = nullptr;

    TensorShape output_shape({batch_size, image_height, image_width, channel_size});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_image_tensor));
    auto output_image = output_image_tensor->template flat<float>();
   
    // Call the cuda kernel launcher
    SliceBiGridKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_grid.data(), 
      input_guide.data(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      grid_height, 
      grid_width, 
      grid_depth, 
      sr,
      output_image.data()
    );
  }
};

class SliceBiGridGradOp : public OpKernel {
 public:
  explicit SliceBiGridGradOp(OpKernelConstruction* context) : OpKernel(context) {  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_grid_tensor = context->input(0);
    // Check the input dimension.
    if (input_grid_tensor.shape().dims() != 5)
      throw std::invalid_argument("Error in Slicing: Input grid dimension should be 5: {batch, height, width, depth, channel}.");
    auto input_grid = input_grid_tensor.flat<float>();
  
    // Grab the input guide
    const Tensor& input_guide_tensor = context->input(1);
    // Check the input guide dimension.
    if (input_guide_tensor.shape().dims() != 3)
      throw std::invalid_argument("Error in Slicing: Input guide dimension should be 3: {batch, height, width}.");
    auto input_guide = input_guide_tensor.flat<float>();

    // Grab ss, sr
    const Tensor& input_attrs_tensor = context->input(2);
    // Check the input guide dimension.
    if (input_attrs_tensor.shape().dims() != 2)
      throw std::invalid_argument("Error: Input guide dimension should be 1: [ss, sr].");
    const int ss = input_attrs_tensor.shape().dim_size(0);
    const int sr = input_attrs_tensor.shape().dim_size(1);

    // Grab the input guide
    const Tensor& backprop_tensor = context->input(3);
    auto backprop = backprop_tensor.flat<float>();
  
    const int batch_size = input_grid_tensor.shape().dim_size(0);
    const int grid_height = input_grid_tensor.shape().dim_size(1);
    const int grid_width = input_grid_tensor.shape().dim_size(2);
    const int grid_depth = input_grid_tensor.shape().dim_size(3);
    const int channel_size = input_grid_tensor.shape().dim_size(4);

    const int image_height = input_guide_tensor.shape().dim_size(1);
    const int image_width = input_guide_tensor.shape().dim_size(2);

    // Create an output tensor
    Tensor* grid_grad_tensor = nullptr;
    Tensor* guide_grad_tensor = nullptr;

    TensorShape image_grad_shape({batch_size, grid_height, grid_width, grid_depth, channel_size});
    TensorShape guide_grad_shape({batch_size, image_height, image_width});
  
    OP_REQUIRES_OK(context, context->allocate_output(0, image_grad_shape,
                                                     &grid_grad_tensor));
    auto grid_grad = grid_grad_tensor->template flat<float>();

    OP_REQUIRES_OK(context, context->allocate_output(1, guide_grad_shape,
                                                     &guide_grad_tensor));
    auto guide_grad = guide_grad_tensor->template flat<float>();
   
    // Call the cuda kernel launcher
    SliceBiGridGradKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_grid.data(), 
      input_guide.data(), 
      backprop.data(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      grid_height, 
      grid_width, 
      grid_depth, 
      sr,
      grid_grad.data(),
      guide_grad.data()
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("CreateBiGrid").Device(DEVICE_GPU), CreateBiGridOp);
REGISTER_KERNEL_BUILDER(Name("CreateBiGridGrad").Device(DEVICE_GPU), CreateBiGridGradOp);
REGISTER_KERNEL_BUILDER(Name("SliceBiGrid").Device(DEVICE_GPU), SliceBiGridOp);
REGISTER_KERNEL_BUILDER(Name("SliceBiGridGrad").Device(DEVICE_GPU), SliceBiGridGradOp);
