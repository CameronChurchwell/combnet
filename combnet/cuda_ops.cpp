#include <torch/extension.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>


/******************************************************************************
Forward definitions
******************************************************************************/


void conv1d_forward(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    torch::Tensor posterior,
    torch::Tensor trellis
);


/******************************************************************************
Macros
******************************************************************************/


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA torch::Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/******************************************************************************
Device-agnostic C++ API
******************************************************************************/


torch::Tensor conv1d(
    torch::Tensor observation,
    torch::Tensor batch_frames,
    torch::Tensor transition,
    torch::Tensor initial,
    int num_threads=0
) {
    
}


/******************************************************************************
Python binding
******************************************************************************/


PYBIND11_MODULE(cuda_ops, m) {
  m.def("conv1d", &conv1d);
}