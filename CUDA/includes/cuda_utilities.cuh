#include <vector>
#include <memory>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "ray.cuh"
#include "imagem.cuh"
#include "camera.cuh"
#include "color.cuh"
#include "shape.cuh"
#include "scene.cuh"
#include "utilities.cuh"
#include "vec3.cuh"
//#include "cuda_parameters.cuh"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

__device__ color ray_color(const ray& r, Shape** objects, int n_obj, float t_min, float t_max, int depth, curandState curand_St);

__global__ void fill_colors(
    color *matrix, int height, int width, Camera* camera, Shape** objects, int n_objs, int samples_per_pixel, float t_min, float t_max, int depth, curandState *curand_St
    );

__global__ void setup_kernel(curandState *state);