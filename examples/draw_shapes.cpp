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

int main() {
    const float aspect_ratio = 16/9;
    const int largura = 400;
    const int altura = static_cast<int> (largura / aspect_ratio);

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = point3(0, 0, 0);
    auto horizontal = vec3(viewport_width, 0, 0);
    auto vertical = vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);

    // Camera
    std::shared_ptr<Camera> camera(new Camera(viewport_height,
                                              viewport_width,
                                              focal_length,
                                              origin,
                                              vertical,
                                              horizontal,
                                              lower_left_corner));

    // Criando objetos
    std::shared_ptr<Material> material_1(new Opaque(color(1,0,0)));
    std::shared_ptr<Material> material_2(new Opaque(color(0,1,0)));
    std::shared_ptr<Material> material_3(new Metal(color(0.8,0.8,0.8), 0));
    std::vector<std::shared_ptr<Shape>> objetos;
    std::shared_ptr<Shape> sphere_1(new Sphere(point3(0,0,-1), 0.3, material_1));
    std::shared_ptr<Shape> sphere_2(new Sphere(point3(0,-100.5,-1), 100, material_2));
    std::shared_ptr<Shape> plane_1(new Plane(point3(-1,0,-2), vec3(-1,0,1), vec3(0,1,0), 2, 2, material_3));
    
    objetos.push_back(sphere_1);
    objetos.push_back(sphere_2);
    objetos.push_back(plane_1);

    // Criando cena
    std::shared_ptr<Scene> cena(new Scene(camera, objetos, largura, altura));

    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(cena, 100, 0.001, infinity, 50, true);
    generated_image->salvar_imagem("exemplo_draw_shapes.ppm");   

    return 0;
}