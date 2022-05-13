#include <vector>
#include <memory>
#include <iostream>
#include "imagem.hpp"
#include "vec3.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "shape.hpp"
#include "renderer.hpp"
#include "utilities.hpp"
#include "ray.hpp"
#include "material.hpp"
#include "config_reader.hpp"

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