#pragma once

#ifndef ICHOL_CUDA_COMPAT_HPP
#define ICHOL_CUDA_COMPAT_HPP

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
using cudaStream_t = void *;
#endif
#else
#include <cuda_runtime.h>
#endif

#endif // ICHOL_CUDA_COMPAT_HPP
