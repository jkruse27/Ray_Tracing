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