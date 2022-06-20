#pragma once

#include <vector>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.cuh"
#include "imagem.cuh"
#include "camera.cuh"
#include "color.cuh"
#include "shape.cuh"
#include "scene.cuh"
#include "utilities.cuh"
#include "vec3.cuh"
#include "cuda_utilities.cuh"
#include "config_reader.cuh"

class Renderer {
    public:
        std::shared_ptr<Imagem> render(SceneParams params);
};

__device__ color final_color(color* a, int n_a, color* e, int n_e){
    color composed_color = e[n_e-1];

    for(int i = n_a-1; i >= 0; i--)
        composed_color = e[i] + a[i]*composed_color;
    
    return composed_color;
}

__device__ color ray_color(const ray& r, SceneParams *params, curandState curand_St){
    ray cur_ray = r;
    //color cur_attenuation = color(1.0f,1.0f,1.0f);

    //novo
    color cur_attenuation[50];
    color emissions[50];
    int n_a = 0;
    int n_e = 0;
    // novo

    bool any_hit = false;
    float min_t = 0;
    float tmp = 0;
    Shape* closest_hit = nullptr;
    Shape* shape;

    for(int j = params->max_depth; j > 0; j--){
        any_hit = false;
        min_t = 0;
        closest_hit = nullptr;

        for(int i = 0; i < (*params->scene)->n_obj; i++){
            shape = (*params->scene)->objects[i]; 
            tmp = shape->hit(cur_ray, params->t_min, params->t_max);

            if((tmp < min_t || !any_hit) && tmp >= 0){
                min_t = tmp;
                closest_hit = shape;
                any_hit = true;
            }
        }

        if(any_hit){
            point3 p = r.at(min_t);
            vec3 n = closest_hit->normal(cur_ray, p);
            point3 target = p + n + random_in_unit_sphere(&curand_St);

            ray scattered;
            color attenuation;
            color emitted = closest_hit->obj_material->emitted();

            //novo
            emissions[params->max_depth-j] = emitted;
            n_e++; 
            //novo

            if (closest_hit->obj_material->scatter(cur_ray, n, p, attenuation, scattered, &curand_St)){
                //cur_attenuation *= attenuation;
                //novo
                emissions[params->max_depth-j] = emitted;
                cur_attenuation[params->max_depth-j] = attenuation;
                n_a++;
                n_e++;
                //novo
                cur_ray = scattered;
            }
            else{
                //novo
                return final_color(cur_attenuation, n_a, emissions, n_e);
                //novo               
                //return emitted;
            }
        }
        else{
            //novo
            emissions[params->max_depth-j] = params->background;
            n_e++;
            return final_color(cur_attenuation, n_a, emissions, n_e);
            //novo
            //return cur_attenuation*params->background;
        }
    }

    return color(0.0f,0.0f,0.0f);
}

__global__ void fill_colors(color *matrix, SceneParams params, curandState *curand_St){
    int height = (*params.scene)->image_height;
    int width = (*params.scene)->image_width;
    Camera* camera = (*params.scene)->camera;

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if((i >= width) || (j >= height)) return;
    curandState curand_States = curand_St[j*width+i];

    auto u = float(i) / (width-1);
    auto v = float(j) / (height-1);
    color pixel_color = color();

    for(int k = 0; k < params.samples_per_pixel; k++)
        pixel_color += ray_color(camera->get_ray(u+curand_uniform(&curand_States)/(width-1), 
                                                 v+curand_uniform(&curand_States)/(height-1),
                                                 &curand_States),
                                    &params,
                                    curand_States);
            

    pixel_color /= params.samples_per_pixel;
    matrix[j*width+i] = pixel_color;
    }

__global__ void setup_kernel(int width, int height, curandState *state)
{
    int id = (threadIdx.x + blockIdx.x * blockDim.x)*width+threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if((i >= width) || (j >= height)) return;

    curand_init(1234, id, 0, &state[j*width+i]);
}

__host__ std::shared_ptr<Imagem> Renderer::render(SceneParams params){
    dim3 blocks(params.width/params.TX+1,params.height/params.TY+1);
    dim3 threads(params.TX,params.TY);

    color* matrix;
    checkCudaErrors(cudaMallocManaged((void **)&matrix, params.height*params.width*sizeof(color)));
    curandState *curand_States = nullptr;
    checkCudaErrors(cudaMalloc((void **)&curand_States, params.height*params.width*sizeof(curandState)));

    setup_kernel<<<blocks, threads>>>(params.width, params.height, curand_States);
    check_and_wait();

    fill_colors<<<blocks, threads>>>(matrix, params, curand_States);

    check_and_wait();
    
    std::shared_ptr<Imagem> image (new Imagem(matrix, params.height, params.width));
    checkCudaErrors(cudaFree(curand_States));
    return image;
}
