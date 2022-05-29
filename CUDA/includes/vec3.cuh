// Codigo adaptado de https://raytracing.github.io/books/RayTracingInOneWeekend.html

#pragma once
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.cuh"

using std::sqrt;
using std::fabs;

class vec3 {
    public:
        __host__ __device__ vec3() : e{0,0,0} {}
        __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}
/*        __host__ __device__ vec3(float *e0, float *e1, float *e2){
            is_rand = true;
            ep[0] = e0;
            ep[1] = e1;
            ep[2] = e2;
            e[0] = *e0;
            e[1] = *e1;
            e[2] = *e2;
        } */


        __host__ __device__ float x() const { return e[0]; }
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ float operator[](int i) const { return e[i]; }
        __host__ __device__ float& operator[](int i) { return e[i]; }

        __host__ __device__ vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__ vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__ vec3& operator/=(const float t) {
            return *this *= 1/t;
        }
        
        __device__ static vec3 random(curandState *curand_States) {
            return vec3(curand_uniform(curand_States), 
                        curand_uniform(curand_States), 
                        curand_uniform(curand_States));
        }

        __device__ static vec3 random(curandState *curand_States, float min, float max) {
            return vec3(min + (max-min)*curand_uniform(curand_States), 
                        min + (max-min)*curand_uniform(curand_States),
                        min + (max-min)*curand_uniform(curand_States));
        }

        __host__ __device__ float length() const {
            return sqrt(length_squared());
        }

        __host__ __device__ float length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        /*__host__ __device__ ~vec3(){
            if(is_rand){
                checkCudaErrors(cudaFree(ep[0]));
                checkCudaErrors(cudaFree(ep[1]));
                checkCudaErrors(cudaFree(ep[2]));
            }
        }*/
    public:
        float e[3];
        float* ep[3];
        bool is_rand;
};


// Type aliases for vec3
using point3 = vec3;   // 3D point

// vec3 Utility Functions

__host__ __device__ inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator*(int t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, int t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_sphere(curandState *curand_States) {
    while (true) {
        auto p = vec3::random(curand_States,-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState *curand_States) {
    vec3 rand = vec3::random(curand_States);
    return rand/rand.length();
}

__device__ inline vec3 random_in_unit_disk(curandState *curand_States){
    float u = curand_uniform(curand_States);
    float x1 = curand_uniform(curand_States);
    float x2 = curand_uniform(curand_States);
    float x3 = curand_uniform(curand_States);

    float mag = sqrt(x1*x1 + x2*x2 + x3*x3);

    x1 /= mag; 
    x2 /= mag; 
    x3 /= mag;

    // Math.cbrt is cube root
    float c = cbrt(u);

    return vec3(x1*c, x2*c, x3*c);
}