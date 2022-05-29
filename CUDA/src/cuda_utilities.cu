#include "cuda_utilities.cuh"

__device__ color ray_color(const ray& r, Shape** objects, int n_obj, float t_min, float t_max, int depth, curandState curand_St)
{
    if (depth <= 0)
        return color(0,0,0);

    bool any_hit = false;
    float min_t = 0;
    float tmp;
    Shape* closest_hit = nullptr;
    Shape* shape;

    for(int i = 0; i < n_obj; i++){
        shape = objects[i];
        tmp = shape->hit(r, t_min, t_max);
        
        if((tmp < min_t || !any_hit) && tmp >= 0){
            min_t = tmp;
            closest_hit = shape;
            any_hit = true;
        }
    }

    if(any_hit){
        point3 p = r.at(min_t);
        vec3 n = closest_hit->normal(r, p);
        point3 target = p + n + random_in_unit_sphere(&curand_St);

        ray scattered;
        color attenuation;
        if (closest_hit->obj_material->scatter(r, n, p, attenuation, scattered, &curand_St))
            return attenuation * ray_color(scattered, objects, n_obj, t_min, t_max, depth-1, curand_St);
            
        return color(0,0,0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
}

__global__ void fill_colors(
    color *matrix, int height, int width, Camera* camera, Shape** objects, int n_objs, int samples_per_pixel, float t_min, float t_max, int depth, curandState *curand_St
    ){

    curandState curand_States = curand_St[threadIdx.x + blockIdx.x * blockDim.x];

    for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < height; j++) {
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < width; ++i) {
            auto u = float(i) / (width-1);
            auto v = float(j) / (height-1);
            color pixel_color = color();

            for(int k = 0; k < samples_per_pixel; k++){
                pixel_color += ray_color(camera->get_ray(u+curand_uniform_float(&curand_States)/(width-1), 
                                                         v+curand_uniform_float(&curand_States)/(height-1),
                                                         &curand_States),
                                         objects,
                                         n_objs,
                                         t_min,
                                         t_max,
                                         depth,
                                         curand_States);
            }
            pixel_color /= samples_per_pixel;
            matrix[j*height+i] = pixel_color;
        }
    }
}

__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x+threadIdx.y + blockIdx.y * blockDim.y;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}