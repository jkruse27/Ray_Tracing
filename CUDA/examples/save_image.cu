#pragma once
#include <vector>
#include "utilities.cuh"
#include "imagem.cuh"
#include "vec3.cuh"
#include "color.cuh"

int main() {

    // Image

    const int image_width = 256;
    const int image_height = 256;

    color* matrix;

    checkCudaErrors(cudaMallocManaged((void **)&matrix, image_height*image_width*sizeof(color)));

    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;

            matrix[i*image_height+j] = color(r, g, b);
        }
    }

    Imagem image = Imagem(matrix, image_height, image_width);
    image.salvar_imagem("../images/exemplo_save_image.ppm");

    checkCudaErrors(cudaFree(matrix));
    
    return 0;
}