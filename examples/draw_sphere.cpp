#include <vector>
#include <memory>
#include "imagem.hpp"
#include "vec3.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "shape.hpp"
#include "renderer.hpp"
#include "utilities.hpp"
#include "ray.hpp"

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
                                              horizontal,
                                              vertical,
                                              lower_left_corner));

    // Criando objetos
    std::vector<std::shared_ptr<Shape>> objetos;
    std::shared_ptr<Shape> sphere_1(new Sphere(point3(0,0,-1), color(1,0,0), 0.5));
    std::shared_ptr<Shape> sphere_2(new Sphere(point3(0,-101,-1), color(0,1,0.2), 100));

    objetos.push_back(sphere_1);
    objetos.push_back(sphere_2);

    // Criando cena
    std::shared_ptr<Scene> cena(new Scene(camera, objetos, altura, largura));

    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(cena, 0, infinity, 2);
    generated_image->salvar_imagem("exemplo.ppm");
}