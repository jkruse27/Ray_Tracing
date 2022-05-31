#pragma once

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "color.cuh"
#include "material.cuh"

typedef enum MATERIALS {
    OPAQUE,
    METAL,
    GLASS
} MATERIALS;

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
        __device__ Opaque(color alb);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
};

class Metal : public Material {
    public:
        float fuzz;
    public:
        __device__ Metal(color alb, float f);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
};

class Glass : public Material {
    public:
        float ir;
    public:
        __device__ Glass(color alb, float ir);

        __device__ bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States
        );
    
    private:
        __device__ static float reflectance(float cosine, float ref_idx);
};

__device__ inline Opaque::Opaque(color alb){
    this->albedo = alb;
}

__device__ inline Metal::Metal(color alb, float f){
    this->albedo = alb;
    this->fuzz = f;
}

__device__ inline Glass::Glass(color alb, float ir){
    this->albedo = alb;
    this->ir = ir;
}

__device__ inline bool Opaque::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    auto scatter_direction = normal + random_unit_vector(curand_States);

    if ((fabs(scatter_direction[0]) < 1e-8f) && (fabs(scatter_direction[1]) < 1e-8f) && (fabs(scatter_direction[2]) < 1e-8f))
        scatter_direction = normal;

    scattered = ray(p, scatter_direction);
    attenuation = albedo;
    return true;
}

__device__ inline bool Metal::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    vec3 rand = unit_vector(r_in.direction());
    vec3 reflected = rand - 2*dot(rand, normal)*normal;
    scattered = ray(p, reflected + fuzz*random_in_unit_sphere(curand_States));
    attenuation = albedo;
    return (dot(scattered.direction(), normal) > 0);    
}

__device__ inline bool Glass::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    attenuation = color(1.0f, 1.0f, 1.0f);
    float refraction_ratio = dot(r_in.direction(), normal) > 0 ? (1.0f/ir) : ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = fmin(dot(-unit_direction, normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform(curand_States)){
        direction = unit_direction - 2*dot(unit_direction, normal)*normal;;
    }
    else{
        vec3 r_out_perp =  refraction_ratio * (unit_direction + cos_theta*normal);
        vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * normal;
        direction = r_out_perp + r_out_parallel;
    }

    scattered = ray(p, direction);
    return true;        
}

__device__ inline float Glass::reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}
