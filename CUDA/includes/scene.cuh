#pragma once

#include <vector>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "camera.cuh"
#include "shape.cuh"
#include "utilities.cuh"

class Scene{
    public:
        Camera* camera;
        Shape** objects;
        int n_obj;
        int image_width, image_height;
    
    public:
        Scene(){};
        Scene(Camera* cam, Shape **shapes, int n_objects, int width, int height)
            : camera(cam), objects(shapes), n_obj(n_objects), image_width(width), image_height(height) 
        {}

        ~Scene(){
            checkCudaErrors(cudaFree(this->camera));
            for(int i = 0; i < n_obj; i++){
                checkCudaErrors(cudaFree(objects[i]->obj_material));
                checkCudaErrors(cudaFree(objects[i]));
            }

            checkCudaErrors(cudaFree(objects));
        }
};