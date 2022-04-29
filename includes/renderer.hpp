#pragma once

#include <vector>
#include <memory>
#include "ray.hpp"
#include "imagem.hpp"
#include "camera.hpp"
#include "color.hpp"
#include "shape.hpp"
#include "scene.hpp"
#include "utilities.hpp"

class Renderer {
    public:
        std::shared_ptr<Imagem> render(std::shared_ptr<Scene> scene, int samples_per_pixel, float t_min, float t_max, int depth);
        color ray_color(const ray& r, std::vector<std::shared_ptr<Shape>> objects, float t_min, float t_max, int depth);
};