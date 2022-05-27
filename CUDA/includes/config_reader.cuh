#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstddef> 
#include <memory>
#include <sstream>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "material.cuh"
#include "shape.cuh"
#include "scene.cuh"
#include "vec3.cuh"
#include "color.cuh"


typedef struct SceneParams {
        Scene *scene;
        int samples_per_pixel;
        int max_depth;
        int log;
        std::string filename;
} SceneParams;

typedef struct Configs {
        float aspect_ratio;
        int largura;
        int altura;
        float viewport_height;
        float viewport_width;
        float focal_length;
        float vfov;
        float aperture;
        float focus_dist;
        point3 origin;
        point3 lookat;
        vec3 horizontal;
        vec3 vertical;
        vec3 vup;
        point3 lower_left_corner;
        std::vector<std::shared_ptr<Shape>> objects;
} Configs;

SceneParams read_scene_from(const char* config_file);
int get_int(std::string text);
float get_float(std::string text);
double get_double(std::string text);
vec3 get_vec3(std::string text);
color get_color(std::string text);
Sphere* get_sphere(std::string text);
Plane* get_plane(std::string text);
std::string get_string(std::string text);