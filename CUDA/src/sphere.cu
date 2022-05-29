#include "shape.cuh"

__host__ Sphere::Sphere(point3 center, float rad, Material* m){
    this->position = center;
    this->radius = rad;
    this->obj_material = m;
}

__device__ float Sphere::hit(const ray& r, float t_min, float t_max){
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

__device__ vec3 Sphere::normal(const ray& r, point3 point){
    return (point - this->position)/this->radius;
}