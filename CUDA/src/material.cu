#include "material.cuh"

__host__ Opaque::Opaque(color alb){
    this->albedo = alb;
}

__host__ Metal::Metal(color alb, double f){
    this->albedo = alb;
    this->fuzz = f;
}

__host__ Glass::Glass(color alb, double ir){
    this->albedo = alb;
    this->ir = ir;
}

__device__ bool Opaque::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    auto scatter_direction = normal + random_unit_vector(curand_States);

    if ((fabs(scatter_direction[0]) < 1e-8) && (fabs(scatter_direction[1]) < 1e-8) && (fabs(scatter_direction[2]) < 1e-8))
        scatter_direction = normal;

    scattered = ray(p, scatter_direction);
    attenuation = albedo;
    return true;
}

__device__ bool Metal::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    vec3 rand = unit_vector(r_in.direction());
    vec3 reflected = rand - 2*dot(rand, normal)*normal;
    scattered = ray(p, reflected + fuzz*random_in_unit_sphere(curand_States));
    attenuation = albedo;
    return (dot(scattered.direction(), normal) > 0);    
}

__device__ bool Glass::scatter(const ray& r_in, vec3 normal, point3 p, color& attenuation, ray& scattered, curandState *curand_States){
    attenuation = color(1.0, 1.0, 1.0);
    double refraction_ratio = dot(r_in.direction(), normal) > 0 ? (1.0/ir) : ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    double cos_theta = fmin(dot(-unit_direction, normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vec3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > curand_uniform_double(curand_States)){
        direction = unit_direction - 2*dot(unit_direction, normal)*normal;;
    }
    else{
        vec3 r_out_perp =  refraction_ratio * (unit_direction + cos_theta*normal);
        vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * normal;
        direction = r_out_perp + r_out_parallel;
    }

    scattered = ray(p, direction);
    return true;        
}

__device__ double Glass::reflectance(double cosine, double ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}
