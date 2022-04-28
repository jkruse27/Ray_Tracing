#pragma once

#include "vec3.hpp"
#include "ray.hpp"
#include "color.hpp"

class Shape {
    public:
        point3 position;
        color shape_color;
    public:
        virtual double hit(const ray& r, float t_min, float t_max) = 0;
        virtual color get_color() = 0;
        virtual vec3 normal(point3 point) = 0;
};

class Sphere : public Shape{
    public:
        double radius;
    public:
        Sphere(point3 center, color col, double rad);
        double hit(const ray& r, float t_min, float t_max);
        color get_color();
        vec3 normal(point3 point);
};

class Cube : public Shape{
    public:
        double hit(const ray& r, float t_min, float t_max);
        color get_color();  
        vec3 normal(point3 point);
};