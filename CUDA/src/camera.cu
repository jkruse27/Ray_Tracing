#include "camera.cuh"

__host__ Camera::Camera( float vp_height, 
                float vp_width,
                float focal_l,
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

__host__ Camera::Camera(point3 origin, 
               point3 lookat,
               vec3 vup,
               float vfov,
               float aspect_ratio,
               float aperture,
               float focus_dist)
    {
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta/2);
    this->viewport_height = 2.0f * h;
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

__device__ ray Camera::get_ray(float x1, float x2, curandState *curand_States) {
    auto rd = (this->aperture/2) * random_in_unit_disk(curand_States);
    vec3 offset = this->u * rd.x() + this->v * rd.y();
    return ray(
                origin + offset, 
                lower_left_corner + x1*horizontal + x2*vertical - origin - offset
               );
}