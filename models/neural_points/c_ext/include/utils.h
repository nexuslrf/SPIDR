#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
// #include <torch/script.h> // for dev
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 512

#define CHECK_CUDA(x)                                          \
  do {                                                         \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                         \
  do {                                                              \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                              \
  do {                                               \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
             #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                              \
  do {                                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
             #x " must be a float tensor");            \
  } while (0)

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
  const int x_threads = opt_n_threads(x);
  const int y_threads =
      std::max(std::min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
  dim3 block_config(x_threads, y_threads, 1);

  return block_config;
}

#define CUDA_CHECK_ERRORS()                                           \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
              __FILE__);                                              \
      exit(-1);                                                       \
    }                                                                 \
  } while (0)