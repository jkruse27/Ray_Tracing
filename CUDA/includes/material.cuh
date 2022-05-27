#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "color.cuh"
#include "material.cuh"

class Material {
    public:
        color albedo;
    public:
        __device__ virtual bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        ) = 0;
};

class Opaque : public Material {
    public:
        Opaque(color alb);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
};

class Metal : public Material {
    public:
        double fuzz;
    public:
        Metal(color alb, double f);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
};

class Glass : public Material {
    public:
        double ir;
    public:
        Glass(color alb, double ir);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
    
    private:
        __device__ static double reflectance(double cosine, double ref_idx);
};