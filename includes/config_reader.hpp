#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstddef> 
#include <memory>
#include <sstream>
#include <algorithm>
#include "material.hpp"
#include "shape.hpp"
#include "scene.hpp"
#include "vec3.hpp"
#include "color.hpp"

typedef struct Configs {
        float aspect_ratio;
        int largura;
        int altura;
        float viewport_height;
        float viewport_width;
        float focal_length;
        point3 origin;
        vec3 horizontal;
        vec3 vertical;
        point3 lower_left_corner;
        std::vector<std::shared_ptr<Shape>> objects;
} Configs;

std::shared_ptr<Scene> read_scene_from(const char* config_file);
int get_int(std::string text);
float get_float(std::string text);
double get_double(std::string text);
vec3 get_vec3(std::string text);
color get_color(std::string text);
std::shared_ptr<Sphere> get_sphere(std::string text);
std::shared_ptr<Plane> get_plane(std::string text);
std::string get_string(std::string text);