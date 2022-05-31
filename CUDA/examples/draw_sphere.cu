#include <vector>
#include <memory>
#include "imagem.cuh"
#include "vec3.cuh"
#include "scene.cuh"
#include "camera.cuh"
#include "shape.cuh"
#include "renderer.cuh"
#include "utilities.cuh"
#include "ray.cuh"
#include "material.cuh"
#include <curand_kernel.h>
#include "renderer.cuh"
#include <iostream>
#include "stdio.h"

__device__ color ray_color(const ray& r, Shape** objects, int n_obj, float t_min, float t_max, int depth, curandState curand_St)
{
    if (depth <= 0)
        return color(0,0,0);

    ray cur_ray = r;
    color cur_attenuation = color(1.0f,1.0f,1.0f);

    bool any_hit = false;
    float min_t = 0;
    float tmp = 0;
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

            if (closest_hit->obj_material->scatter(cur_ray, n, p, attenuation, scattered, &curand_St)){
                cur_attenuation *= attenuation;
                cur_ray = scattered;
                //return attenuation * ray_color(scattered, objects, n_obj, t_min, t_max, depth-1, curand_St);
            }
            else
                return color(0,0,0);
        }
        else{
            vec3 unit_direction = unit_vector(r.direction());
            auto t = 0.5f*(unit_direction.y() + 1.0f);
            color c = (1.0f-t)*color(1.0f, 1.0f, 1.0f) + t*color(0.5f, 0.7f, 1.0f);
            return cur_attenuation*c;
        }
    }

    return color(0.0f,0.0f,0.0f);
}

__global__ void fill_colors(
    color *matrix, Scene** scene, int samples_per_pixel, float t_min, float t_max, int depth, curandState *curand_St
    ){
    int height = (*scene)->image_height;
    int width = (*scene)->image_width;
    Camera* camera = (*scene)->camera;
    Shape** objects = (*scene)->objects;
    int n_objs = (*scene)->n_obj;

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if((i >= width) || (j >= height)) return;
    curandState curand_States = curand_St[j*width+i];

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

__host__ std::shared_ptr<Imagem> Renderer::render(
    Scene** scene, int height, int width, int samples_per_pixel, float t_min, float t_max, int depth, bool log
    ){

    dim3 blocks(width/TX+1,height/TY+1);
    dim3 threads(TX,TY);

    color* matrix;
    checkCudaErrors(cudaMallocManaged((void **)&matrix, height*width*sizeof(color)));
    curandState *curand_States = nullptr;
    checkCudaErrors(cudaMalloc((void **)&curand_States, height*width*sizeof(curandState)));

    setup_kernel<<<blocks, threads>>>(width, height, curand_States);

    check_and_wait();

    const float aspect_ratio = (float) (16.0/9.0);
    const int largura = 100;
    const int altura = static_cast<int> (largura / aspect_ratio);
    //fill_colors<<<blocks, threads>>>(matrix, height, width, camera, objects, n_obj, samples_per_pixel, t_min, t_max, depth, curand_States);

    fill_colors<<<blocks, threads>>>(matrix, scene, samples_per_pixel, t_min, t_max, depth, curand_States);

    check_and_wait();
    
    std::shared_ptr<Imagem> image (new Imagem(matrix, height, width));
    checkCudaErrors(cudaFree(curand_States));

    return image;
}

__global__ void initialize_scene(Shape **objects, 
                                 Camera **camera,
                                 Scene **cena){

    const float aspect_ratio = (float) (16.0/9.0);
    const int largura = 100;
    const int altura = static_cast<int> (largura / aspect_ratio);
    
    auto viewport_height = 2.0f;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0f;
    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    // Criando objetos
    objects[0] = new Sphere(point3(0,0,-1), 0.5,  new Opaque(color(1,0,0)));
    objects[1] = new Sphere(point3(0,-100.5,-1), 100, new Opaque(color(0,1,0)));

    *camera = new Camera(viewport_height,
                         viewport_width,
                         focal_length,
                         origin,
                         vertical,
                         horizontal,
                         lower_left_corner);

    *cena = new Scene(*camera, objects, 2, largura, altura);
    }

int main() {
    // Camera
    Camera** camera;
    checkCudaErrors(cudaMallocManaged((void **)&camera, sizeof(Camera *)));
    Shape** objects;
    checkCudaErrors(cudaMallocManaged((void **)&objects, sizeof(Shape *)*2));
    Scene** cena;
    checkCudaErrors(cudaMallocManaged((void **)&cena, sizeof(Scene *)));

    initialize_scene<<<1, 1>>>(objects, 
                               camera,
                               cena
    );

    check_and_wait();

    // Criando cena
    Renderer renderer;
    const float aspect_ratio = (float) (16.0/9.0);
    const int largura = 100;
    const int altura = static_cast<int> (largura / aspect_ratio);

    std::shared_ptr<Imagem> generated_image = renderer.render(cena, altura, largura, 100, 0.001f, infinity, 50, true);
    std::cout << "Finalizei!" << std::endl;
    generated_image->salvar_imagem("../images/exemplo_draw_sphere.ppm");
 
    checkCudaErrors(cudaFree(objects));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(cena));

    return 0;
}