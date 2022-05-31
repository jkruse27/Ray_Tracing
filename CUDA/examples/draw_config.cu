#include <vector>
#include <memory>
#include <iostream>
#include <curand_kernel.h>
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
#include "stdio.h"

int main() {
    // Criando cena
    SceneParams params = read_scene_from("../examples/config.scene");
    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(params);

    generated_image->salvar_imagem(params.filename.c_str());

    free_scene(params);

    return 0;
}