#pragma once

#include <vector>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ray.cuh"
#include "imagem.cuh"
#include "camera.cuh"
#include "color.cuh"
#include "shape.cuh"
#include "scene.cuh"
#include "utilities.cuh"
#include "vec3.cuh"
#include "cuda_utilities.cuh"
#include "cuda_parameters.cuh"

class Renderer {
    public:
        std::shared_ptr<Imagem> render(
            Scene** scene, int height, int width, int samples_per_pixel, float t_min, float t_max, int depth, bool log
            );
};