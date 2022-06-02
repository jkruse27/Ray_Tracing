#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
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

int main(int argc, char *argv[]) {
    // Criando cena
    char* filename = "../examples/config.scene";
    if(argc > 1)
        filename = argv[1];

    SceneParams params = read_scene_from(filename);
    Renderer renderer;

    auto start = chrono::steady_clock::now();
    std::shared_ptr<Imagem> generated_image = renderer.render(  params.scene, 
                                                                params.samples_per_pixel,
                                                                0.001, 
                                                                infinity, 
                                                                params.max_depth, 
                                                                params.log);

    auto end = chrono::steady_clock::now();

    std::cout << chrono::duration_cast<chrono::milliseconds>(end - start).count() << endl;

    generated_image->salvar_imagem(params.filename.c_str());
    
    return 0;
}