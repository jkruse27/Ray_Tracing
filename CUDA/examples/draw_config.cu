#include <vector>
#include <memory>
#include <iostream>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
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
#include "stdio.h"

int main(int argc, char *argv[]) {
    // Criando cena
    char* filename = "../examples/config.scene";
    if(argc > 1)
        filename = argv[1];

    SceneParams params = read_scene_from(filename);
    Renderer renderer;

    auto start = chrono::steady_clock::now();
    std::shared_ptr<Imagem> generated_image = renderer.render(params);
    auto end = chrono::steady_clock::now();

    std::cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    generated_image->salvar_imagem(params.filename.c_str());

    free_scene(params);

    return 0;
}