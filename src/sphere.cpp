#include "shape.hpp"

Sphere::Sphere(point3 center, color col, double rad, shared_ptr<Material> m){
    this->position = center;
    this->shape_color = col;
    this->radius = rad;
    this->obj_material = m;
}

double Sphere::hit(const ray& r, float t_min, float t_max){
    double t = 0;
    vec3 oc = r.origin() - this->position;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;

    if(discriminant < 0)
        return -1.0;
    
    t = (-b-sqrt(discriminant))/(2.0*a);

    if(t > t_max || t < t_min)
        return -1.0;

    return t;
}

color Sphere::get_color(){
    return this->shape_color;
}

vec3 Sphere::normal(const ray& r, point3 point){
    return (point - this->position)/this->radius;
}