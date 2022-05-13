#pragma once

#include <vector>
#include <memory>
#include "camera.hpp"
#include "shape.hpp"

class Scene{
    public:
        std::shared_ptr<Camera> camera;
        std::vector<std::shared_ptr<Shape>> objects;
        double image_width, image_height;
    
    public:
        Scene(){};
        Scene(std::shared_ptr<Camera> cam, std::vector<std::shared_ptr<Shape>> shapes, double width, double height)
            : camera(cam), objects(shapes), image_width(width), image_height(height) 
        {}
};