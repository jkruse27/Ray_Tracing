#pragma once

#include <vector>
#include <memory>
#include "vec3.hpp"
#include "color.hpp"

class ray {
    public:
        point3 orig;
        vec3 dir;

    public:
        ray() {}
        ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(unit_vector(direction))
        {}

        point3 origin() const  { return orig; }
        vec3 direction() const { return dir; }

        point3 at(double t) const {
            return orig + t*dir;
        }
};