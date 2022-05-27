#include "renderer.cuh"
#include <iostream>

__host__ std::shared_ptr<Imagem> Renderer::render(
    Scene* scene, int samples_per_pixel, float t_min, float t_max, int depth, bool log
    ){
    auto height = scene->image_height;
    auto width = scene->image_width;
    auto camera = scene->camera;
    auto objects = scene->objects;

    dim3 blocks(width/TX+1,height/TY+1);
    dim3 threads(TX,TY);

    color* matrix;

    checkCudaErrors(cudaMallocManaged((void **)&matrix, height*width*sizeof(color)));

    curandState *curand_States = nullptr;

    setup_kernel<<<blocks, threads>>>(curand_States);
    fill_colors<<<blocks, threads>>>(matrix, height, width, camera, objects, scene->n_obj, samples_per_pixel, t_min, t_max, depth, curand_States);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    std::shared_ptr<Imagem> image (new Imagem(matrix, height, width));
    checkCudaErrors(cudaFree(curand_States));

    return image;
}