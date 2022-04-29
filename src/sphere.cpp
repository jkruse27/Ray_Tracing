#include "shape.hpp"

Sphere::Sphere(point3 center, color col, double rad){
    this->position = center;
    this->shape_color = col;
    this->radius = rad;
}

double Sphere::hit(const ray& r, float t_min, float t_max){
    vec3 oc = r.origin() - this->position;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;

    if(discriminant < 0)
        return -1.0;
    
    return (-b-sqrt(discriminant))/(2.0*a);
}

color Sphere::get_color(){
    return this->shape_color;
}

vec3 Sphere::normal(const ray& r, point3 point){
    //vec3 normal = (point - this->position)/this->radius;
    return (point - this->position)/this->radius;
    //return dot(r.direction(), normal) < 0 ? (-1.0)*normal : normal;
}