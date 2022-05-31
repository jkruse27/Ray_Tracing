#pragma once

#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
//#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
//#include <stdlib.h>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define checkCudaRandErrors(val) check_cuda_rand( (val), #val, __FILE__, __LINE__ )

inline __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
    file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

inline __device__ void check_cuda_rand(curandStatus_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    printf("CUDA error = %u at %s: %d '%s'\n", static_cast<unsigned int>(result), file, line, func);
  }
}

inline __host__ void check_and_wait(){
    cudaError_t err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
      exit(-1);
    }

    checkCudaErrors(cudaDeviceSynchronize());
}

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
    const float pi = 3.1415926535897932385f;
    return degrees * pi / 180.0f;
}

__host__ inline void printProgress(float percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    std::printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    std::fflush(stdout);
}

#endif
