#include "renderer.hpp"
#include <iostream>

std::shared_ptr<Imagem> Renderer::render(std::shared_ptr<Scene> scene, float t_min, float t_max, int depth){
    auto height = scene->image_height;
    auto width = scene->image_width;
    auto origin = scene->camera->origin;
    auto lower_left_corner = scene->camera->lower_left_corner;
    auto horizontal = scene->camera->horizontal;
    auto vertical = scene->camera->vertical;

    vector<vector<color>> matrix(height, vector<color>(width));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; ++i) {
            auto u = double(i) / (width-1);
            auto v = double(j) / (height-1);
            ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
            color pixel_color = ray_color(r, scene->objects, t_min, t_max, depth);
            matrix[i][j] = pixel_color;
        }
    }

    std::shared_ptr<Imagem> image (new Imagem(matrix));
    return image;
}

color Renderer::ray_color(const ray& r, std::vector<std::shared_ptr<Shape>> objects, float t_min, float t_max, int depth){
    //if (depth <= 0)
    //    return color(0,0,0);

    bool any_hit = false;
    float min_t = 0;
    float tmp;
    std::shared_ptr<Shape> closest_hit(nullptr);

    for(std::shared_ptr<Shape> shape : objects){
        tmp = shape->hit(r, t_min, t_max);
        if((tmp < min_t || !any_hit) && tmp >= 0){
            min_t = tmp;
            closest_hit = shape;
            any_hit = true;
        }
    }

    if(any_hit){
        point3 p = r.at(min_t);
        vec3 n = closest_hit->normal(p);
        point3 target = p + n;
        return 0.8*closest_hit->get_color() + 0.2*ray_color(ray(p, target - p), objects, t_min, t_max, depth-1);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}