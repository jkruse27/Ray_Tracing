#pragma once

#include "vec3.hpp"
#include "ray.hpp"
#include "color.hpp"
#include "material.hpp"

class Material {
    public:
        color albedo;
    public:
        virtual bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered
        ) = 0;
};

class Opaque : public Material {
    public:
        Opaque(color alb);

        bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered
        );
};

class Metal : public Material {
    public:
        double fuzz;
    public:
        Metal(color alb, double f);

        bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered
        );
};

class Glass : public Material {
    public:
        double ir;
    public:
        Glass(color alb, double ir);

        bool scatter(
            const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered
        );
};