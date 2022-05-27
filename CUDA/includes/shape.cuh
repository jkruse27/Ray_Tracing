#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.cuh"
#include "ray.cuh"
#include "color.cuh"
#include "material.cuh"

class Shape {
    public:
        point3 position;
        color shape_color;
        Material* obj_material;
    public:
        __device__ virtual double hit(const ray& r, float t_min, float t_max) = 0;
        __device__ virtual vec3 normal(const ray& r, point3 point) = 0;
};

class Sphere : public Shape{
    public:
        double radius;
    public:
        __host__ Sphere(point3 center, double rad, Material* m);
        __device__ double hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};

class Cube : public Shape{
    public:
        __device__ double hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};

class Plane : public Shape{
    public:
        vec3 u_dir;
        vec3 v_dir;
        vec3 n;
        double u;
        double v;
    public:
        __host__ Plane(point3 center, vec3 u_dir, vec3 v_dir, double u, double v, Material* m);
        __device__ double hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};