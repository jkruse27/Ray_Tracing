#include "camera.hpp"

Camera::Camera( double vp_height, 
                double vp_width,
                double focal_l,
                point3 orig,
                point3 vert,
                point3 hor,
                point3 lll
                ){
    this->viewport_height = vp_height;
    this->viewport_width = vp_width;
    this->focal_length = focal_l;
    this->origin = orig;
    this->vertical = vert;
    this->horizontal = hor;
    this->lower_left_corner = lll;
}

Camera::Camera(point3 origin, 
               point3 lookat,
               vec3 vup,
               double vfov,
               double aspect_ratio,
               double aperture,
               double focus_dist)
    {
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta/2);
    this->viewport_height = 2.0 * h;
    this->viewport_width = aspect_ratio * viewport_height;
    this->aperture = aperture;
    this->focus_dist = focus_dist;

    this->w = unit_vector(origin - lookat);
    this->u = unit_vector(cross(vup, w));
    this->v = cross(w, u);

    this->origin = origin;
    this->horizontal = focus_dist * this->viewport_width * this->u;
    this->vertical = focus_dist * this->viewport_height * this->v;
    this->lower_left_corner = origin - horizontal/2 - vertical/2 - (focus_dist* this->w);
}

ray Camera::get_ray(double x1, double x2) {
    vec3 rd = (this->aperture/2) * random_in_unit_disk();
    vec3 offset = this->u * rd.x() + this->v * rd.y();
    return ray(
                origin + offset, 
                lower_left_corner + x1*horizontal + x2*vertical - origin - offset
               );
}