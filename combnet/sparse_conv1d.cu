#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define NUM_THREADS 1024
#define WARP_SIZE 32
#define NUM_WARPS 32

#define FULL_MASK 0xffffffff


/******************************************************************************
Convolution CUDA kernels
******************************************************************************/


__global__ void conv1d_forward_kernel(
    float* input_ptr,
    float* weight_ptr,
    float* output_ptr,
    int input_length,
    int kernel_size,
    int output_length
    int stride,
) {
    // Handle batch
    int batch_id = blockIdx.x;

    float weight;
    float input;

    for (int i=threadIdx.x; i<output_length; i+=1) {
        for (int j=threadIdx.x; j<kernel_length; j+=NUM_THREADS) {
            weight = weight_ptr[j];
            input = input_ptr[i+j];
        }
    }
}

void conv1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int: stride,
) {
    auto device = input.device()
    const int threads = NUM_THREADS;
    int batch_size = input.size(0);
    int input_channels = input.size(1);
    int input_length = input.size(2);
    int output_channels = weight.size(0);
    int kernel_size = weight.shape(2);
    const dim3 blocks(batch_size);
    int device_num = device.index();
    cudaSetDevice(device_num);

    output_length = ((input_length - kernel_size) / stride) + 1

    torch::Tensor output = torch::zeros(
        {batch_size, output_channels, output_length},
        torch::dtype(torch::kInt32).device(device));

    conv1d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        input_length,
        kernel_size,
        output_length
        stride,
    );
}