#include "shape.hpp"

Plane::Plane(point3 center, vec3 u_dir, vec3 v_dir, double u, double v, shared_ptr<Material> m){
    this->position = center;
    this->u_dir = u_dir;
    this->v_dir = v_dir;
    this->n = cross(u_dir, v_dir);
    this->u = u;
    this->v = v;
    this->obj_material = m;
}

double Plane::hit(const ray& r, float t_min, float t_max){
    double t = -dot(r.origin()-this->position, this->n)/dot(r.direction(), this->n);
    vec3 p = r.origin() + r.direction()*t;

    double u = dot(this->u_dir, p-this->position);
    double v = dot(this->v_dir, p-this->position);
    
    if(u > this->u/2 || u < -this->u/2 || v > this->v/2 || v < -this->v/2)
        return -1.0;

    if(t > t_max || t < t_min)
        return -1.0;

    return t;
}

color Plane::get_color(){
    return this->shape_color;
}

vec3 Plane::normal(const ray& r, point3 point){
    return dot(r.direction(), this->n) < 0 ? this->n : -1*this->n;
}