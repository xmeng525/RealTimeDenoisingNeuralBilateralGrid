// outlier.cc
// This program is used to remove high dynamic range spikes.
// Author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
using namespace tensorflow;  // NOLINT(build/namespaces)
using shape_inference::DimensionHandle;

typedef Eigen::GpuDevice GPUDevice;

REGISTER_OP("Outlier")
    .Input("input_image: float")
    .Input("input_albedo: float")
    .Input("input_normal: float")
    .Output("output_image: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

bool OutlierKernelLauncher(
  const GPUDevice& d,
  const float* input_image,
  const float* input_albedo,
  const float* input_normal,
  const int batch_size,
  const int image_height,
  const int image_width,
  const int channel_size,
  float* const output_image
);


class OutlierOp : public OpKernel {
 public:
  explicit OutlierOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_image_tensor = context->input(0);
    // Check the input dimension.
    if (input_image_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input image dimension should be 4: {batch, height, width, channel}.");
    auto input_image = input_image_tensor.flat<float>();
	
    // Grab the input guide
    const Tensor& input_albedo_tensor = context->input(1);
    // Check the input guide dimension.
    if (input_albedo_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input albedo dimension should be 4: {batch, height, width, channel}.");
    auto input_albedo = input_albedo_tensor.flat<float>();

    // Grab ss, sr
    const Tensor& input_normal_tensor = context->input(2);
    // Check the input guide dimension.
    if (input_normal_tensor.shape().dims() != 4)
      throw std::invalid_argument("Error: Input normal dimension should be 4: {batch, height, width, channel}.");
    auto input_normal = input_normal_tensor.flat<float>();

    const int batch_size = input_image_tensor.shape().dim_size(0);
    const int image_height = input_image_tensor.shape().dim_size(1);
    const int image_width = input_image_tensor.shape().dim_size(2);
    const int channel_size = input_image_tensor.shape().dim_size(3);

    // Create an output tensor
    Tensor* output_image_tensor = nullptr;

    TensorShape output_shape({batch_size, image_height, image_width, channel_size});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_image_tensor));
    auto output_image = output_image_tensor->template flat<float>();

    // Call the cuda kernel launcher
    OutlierKernelLauncher(
      context->eigen_device<GPUDevice>(),
      input_image.data(), 
	    input_albedo.data(),
      input_normal.data(),
      batch_size,
      image_height,
      image_width,
      channel_size,
      output_image.data()
    );
  }
};


REGISTER_KERNEL_BUILDER(Name("Outlier").Device(DEVICE_GPU), OutlierOp);