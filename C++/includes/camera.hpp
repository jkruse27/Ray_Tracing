#pragma once

#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"

class Camera {
    public:
        double viewport_height, viewport_width;
        double focal_length;
        point3 origin;
        point3 vertical, horizontal;
        point3 lower_left_corner;
        double aperture;
        double focus_dist;
        vec3 u, v, w;
    
    public:
        Camera(double vp_height, double vp_width, double focal_l, point3 orig, point3 vert, point3 hor, point3 lll);
        Camera(point3 origin, point3 lookat, vec3 vup, double vfov, double aspect_ratio, double aperture, double focus_dist);
        ray get_ray(double x1, double x2);
};