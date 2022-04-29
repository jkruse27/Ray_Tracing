#pragma once

#include "color.hpp"
#include "ray.hpp"
#include "vec3.hpp"

class Camera {
    public:
        double viewport_height;
        double viewport_width;
        double focal_length;
        point3 origin;
        point3 vertical;
        point3 horizontal;
        point3 lower_left_corner;
    
    public:
        Camera(double vp_height, double vp_width, double focal_l, point3 orig, point3 vert, point3 hor, point3 lll)
            : viewport_height(vp_height), viewport_width(vp_width), focal_length(focal_l), 
              origin(orig), vertical(vert), horizontal(hor), lower_left_corner(lll)
        {};

        ray get_ray(double u, double v) {
            return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
        }
};