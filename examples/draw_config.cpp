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
    std::shared_ptr<Scene> cena = read_scene_from("../examples/config.scene");

    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(cena, 500, 0.001, infinity, 50, true);

    generated_image->salvar_imagem("exemplo_draw_config.ppm");

    return 0;
}