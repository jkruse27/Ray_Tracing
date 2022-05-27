#pragma once

#ifndef UTILITIES
#define UTILITIES

#include <iostream>
#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define checkCudaRandErrors(val) check_cuda_rand( (val), #val, __FILE__, __LINE__ )

inline __host__ void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

inline __device__ void check_cuda_rand(curandStatus_t result, char const *const func, const char *const file, int const line) {
  if (result) {

  }
}

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

__host__ inline double degrees_to_radians(double degrees) {
    const double pi = 3.1415926535897932385;
    return degrees * pi / 180.0;
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

__host__ inline void printProgress(double percentage) {
    int val = (int) (percentage * 100);
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    std::printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    std::fflush(stdout);
}

#endif
