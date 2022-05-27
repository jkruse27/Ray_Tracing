#pragma once

#include <vector>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "color.cuh"

class ray {
    public:
        point3 orig;
        vec3 dir;

    public:
        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        __device__ point3 origin() const  { return orig; }
        __device__ vec3 direction() const { return dir; }

        __device__ point3 at(double t) const {
            return orig + t*dir;
        }
};