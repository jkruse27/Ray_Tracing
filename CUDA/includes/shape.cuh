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
        __device__ virtual float hit(const ray& r, float t_min, float t_max) = 0;
        __device__ virtual vec3 normal(const ray& r, point3 point) = 0;
};

class Sphere : public Shape{
    public:
        float radius;
    public:
        __host__ Sphere(point3 center, float rad, Material* m);
        __device__ float hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};

class Cube : public Shape{
    public:
        __device__ float hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};

class Plane : public Shape{
    public:
        vec3 u_dir;
        vec3 v_dir;
        vec3 n;
        float u;
        float v;
    public:
        __host__ Plane(point3 center, vec3 u_dir, vec3 v_dir, float u, float v, Material* m);
        __device__ float hit(const ray& r, float t_min, float t_max);
        __device__ vec3 normal(const ray& r, point3 point);
};






__host__ inline Sphere::Sphere(point3 center, float rad, Material* m){
    this->position = center;
    this->radius = rad;
    this->obj_material = m;
}

__device__ inline float Sphere::hit(const ray& r, float t_min, float t_max){
    float t = 0;
    vec3 oc = r.origin() - this->position;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0f * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;

    if(discriminant < 0)
        return -1.0f;
    
    t = (-b-sqrt(discriminant))/(2.0*a);

    if(t > t_max || t < t_min)
        return -1.0f;

    return t;
}

__device__ inline vec3 Sphere::normal(const ray& r, point3 point){
    return (point - this->position)/this->radius;
}

__host__ inline Plane::Plane(point3 center, vec3 u_dir, vec3 v_dir, float u, float v, Material* m){
    this->position = center;
    this->u_dir = u_dir;
    this->v_dir = v_dir;
    this->n = cross(u_dir, v_dir);
    this->u = u;
    this->v = v;
    this->obj_material = m;
}

__device__ inline float Plane::hit(const ray& r, float t_min, float t_max){

    float t = -dot(r.origin()-this->position, this->n)/dot(r.direction(), this->n);
    vec3 p = r.origin() + r.direction()*t;
    float u = dot(this->u_dir, p-this->position);
    float v = dot(this->v_dir, p-this->position);
    if(u > this->u/2 || u < -this->u/2 || v > this->v/2 || v < -this->v/2)
        return -1.0f;
    if(t > t_max || t < t_min)
        return -1.0f;
    return t;
}

__device__ inline vec3 Plane::normal(const ray& r, point3 point){
    return dot(r.direction(), this->n) < 0 ? this->n : -1*this->n;
}