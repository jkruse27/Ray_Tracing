#include <vector>
#include <memory>
#include "imagem.cuh"
#include "vec3.cuh"
#include "scene.cuh"
#include "camera.cuh"
#include "shape.cuh"
#include "renderer.cuh"
#include "utilities.cuh"
#include "ray.cuh"
#include "material.cuh"

int main() {
    const float aspect_ratio = (float) (16.0/9.0);
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
    Camera* camera;
    checkCudaErrors(cudaMallocManaged((void **)&camera, sizeof(Camera)));

    camera = new (camera) Camera(viewport_height,
                                              viewport_width,
                                              focal_length,
                                              origin,
                                              vertical,
                                              horizontal,
                                              lower_left_corner);

    // Criando objetos
    Material* material_1;
    checkCudaErrors(cudaMallocManaged((void **)&material_1, sizeof(Opaque)));
    material_1 = new (material_1) Opaque(color(1,0,0));

    Material* material_2;
    checkCudaErrors(cudaMallocManaged((void **)&material_2, sizeof(Opaque)));
    material_2 = new (material_2) Opaque(color(0,1,0));

    Shape** objetos;
    checkCudaErrors(cudaMallocManaged((void **)&objetos, sizeof(Shape*)*2));

    Shape* sphere_1;
    checkCudaErrors(cudaMallocManaged((void **)&sphere_1, sizeof(Sphere)));
    sphere_1 = new (sphere_1) Sphere(point3(0,0,-1), 0.5, material_1);

    Shape* sphere_2;
    checkCudaErrors(cudaMallocManaged((void **)&sphere_2, sizeof(Sphere)));
    sphere_2 = new (sphere_2) Sphere(point3(0,-100.5,-1), 100, material_2);

    objetos[0] = sphere_1;
    objetos[1] = sphere_2;

    // Criando cena
    Scene* cena;
    checkCudaErrors(cudaMallocManaged((void **)&cena, sizeof(Scene)));
    cena = new (cena) Scene(camera, objetos, 2, largura, altura);

    Renderer renderer;

    std::shared_ptr<Imagem> generated_image = renderer.render(cena, 100, (float) 0.001, infinity, 50, true);
    generated_image->salvar_imagem("../images/exemplo_draw_sphere.ppm");

    checkCudaErrors(cudaFree(material_1));
    checkCudaErrors(cudaFree(material_2));   
    checkCudaErrors(cudaFree(objetos[0]));
    checkCudaErrors(cudaFree(objetos[1]));
    checkCudaErrors(cudaFree(objetos));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(cena));


    return 0;
}