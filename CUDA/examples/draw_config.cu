#include <vector>
#include <memory>
#include <iostream>
#include "imagem.cuh"
#include "vec3.cuh"
#include "scene.cuh"
#include "camera.cuh"
#include "shape.cuh"
#include "renderer.cuh"
#include "utilities.cuh"
#include "ray.cuh"
#include "material.cuh"
#include "config_reader.cuh"

__device__ color ray_color(const ray& r, Shape** objects, int n_obj, float t_min, float t_max, int depth, curandState curand_St)
{
    if (depth <= 0)
        return color(0,0,0);

    ray cur_ray = r;
    color cur_attenuation = color(1.0,1.0,1.0);

    bool any_hit = false;
    float min_t = 0;
    float tmp;
    Shape* closest_hit = nullptr;
    Shape* shape;

    for(int j = depth; j > 0; j--){
        any_hit = false;
        min_t = 0;
        closest_hit = nullptr;

        for(int i = 0; i < n_obj; i++){
            shape = objects[i];
            tmp = shape->hit(cur_ray, t_min, t_max);
            
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
            if (closest_hit->obj_material->scatter(cur_ray, n, p, cur_attenuation, scattered, &curand_St)){
                cur_attenuation *= attenuation;
                cur_ray = scattered;
                //return attenuation * ray_color(scattered, objects, n_obj, t_min, t_max, depth-1, curand_St);
            }
                
            return color(0,0,0);
        }
        else{
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5f*(unit_direction.y() + 1.0f);
            color c = (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
            return cur_attenuation*c;
        }
    }

    return color(0.0,0.0,0.0);
}

__global__ void fill_colors(
    color *matrix, int height, int width, Camera* camera, Shape** objects, int n_objs, int samples_per_pixel, float t_min, float t_max, int depth, curandState *curand_St
    ){
    curandState curand_States = curand_St[threadIdx.x + blockIdx.x * blockDim.x];

    for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < height; j++) {
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < width; ++i) {
            printf("%d, %d\n", j, i);
            curand_States = curand_St[j*height+i];
            auto u = float(i) / (width-1);
            auto v = float(j) / (height-1);
            color pixel_color = color();
            
            for(int k = 0; k < samples_per_pixel; k++){
                pixel_color += ray_color(camera->get_ray(u+curand_uniform(&curand_States)/(width-1), 
                                                         v+curand_uniform(&curand_States)/(height-1),
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

__host__ std::shared_ptr<Imagem> Renderer::render(
    Scene* scene, int samples_per_pixel, float t_min, float t_max, int depth, bool log
    ){
    auto height = scene->image_height;
    auto width = scene->image_width;
    auto camera = scene->camera;
    auto objects = scene->objects;
    int n_obj = scene->n_obj;

    dim3 blocks(width/TX+1,height/TY+1);
    dim3 threads(TX,TY);

    color* matrix;

    checkCudaErrors(cudaMallocManaged((void **)&matrix, height*width*sizeof(color)));

    curandState *curand_States = nullptr;
    checkCudaErrors(cudaMalloc((void **)&curand_States, height*width*sizeof(curandState)));

    setup_kernel<<<blocks, threads>>>(curand_States);

    cudaError_t err = cudaGetLastError();        // Get error code
    if ( err != cudaSuccess )
    {
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
      exit(-1);
    }

    checkCudaErrors(cudaDeviceSynchronize());

    fill_colors<<<blocks, threads>>>(matrix, height, width, camera, objects, n_obj, samples_per_pixel, t_min, t_max, depth, curand_States);

    err = cudaGetLastError();
    if ( err != cudaSuccess ){
        std::cout << "CUDA Error:" << cudaGetErrorString(err) << std::endl;       
        exit(-1);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    
    std::shared_ptr<Imagem> image (new Imagem(matrix, height, width));
    checkCudaErrors(cudaFree(curand_States));

    return image;
}

int main() {
    // Criando cena
    SceneParams params = read_scene_from("../examples/config.scene");
    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(  params.scene, 
                                                                params.samples_per_pixel,
                                                                0.001, 
                                                                infinity, 
                                                                params.max_depth, 
                                                                params.log);

    generated_image->salvar_imagem(params.filename.c_str());
    
    return 0;
}