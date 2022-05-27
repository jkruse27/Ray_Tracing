# Versao minima do cmake
cmake_minimum_required(VERSION 3.0)

# Nome do meu projeto
project(RayTracer LANGUAGES CUDA CXX)

#
add_executable(save_image_test examples/save_image.cu)
add_executable(sphere_test examples/draw_sphere.cu)
add_executable(materials_test examples/draw_materials.cu)
add_executable(shapes_test examples/draw_shapes.cu)
add_executable(from_file_test examples/draw_config.cu)

# Especificar onde estao os header files
include_directories(includes/)

# Bibliotecas locais (estaticas)
add_library(Imagem src/imagem.cu)
add_library(CUDA_Utils src/cuda_utilities.cu)
add_library(Renderer src/renderer.cu)
add_library(Sphere src/sphere.cu)
add_library(Plane src/plane.cu)
add_library(Materials src/material.cu)
add_library(Camera src/camera.cu)
add_library(Config src/config_reader.cu)

target_link_libraries(Config Sphere)
target_link_libraries(Config Plane)
target_link_libraries(Config Materials)
target_link_libraries(Config Camera)

target_link_libraries(CUDA_Utils Sphere)
target_link_libraries(CUDA_Utils Plane)
target_link_libraries(CUDA_Utils Materials)
target_link_libraries(CUDA_Utils Camera)

target_link_libraries(Renderer CUDA_Utils)

# Exemplo de criacao de imagem
target_link_libraries(save_image_test Imagem)
target_link_libraries(save_image_test Camera)

# Exemplo de geracao de imagem com circulo
target_link_libraries(sphere_test Imagem)
target_link_libraries(sphere_test Sphere)
target_link_libraries(sphere_test Materials)
target_link_libraries(sphere_test Renderer)
target_link_libraries(sphere_test Camera)
target_link_libraries(sphere_test CUDA_Utils)

# Exemplo de geracao de imagem com difrentes materiais
target_link_libraries(materials_test Imagem)
target_link_libraries(materials_test Sphere)
target_link_libraries(materials_test Materials)
target_link_libraries(materials_test Renderer)
target_link_libraries(materials_test Camera)

# Exemplo de geracao de imagem com difrentes formas
target_link_libraries(shapes_test Imagem)
target_link_libraries(shapes_test Sphere)
target_link_libraries(shapes_test Plane)
target_link_libraries(shapes_test Materials)
target_link_libraries(shapes_test Renderer)
target_link_libraries(shapes_test Camera)

# Exemplo de geracao de imagem a partir de config
target_link_libraries(from_file_test Imagem)
target_link_libraries(from_file_test Sphere)
target_link_libraries(from_file_test Plane)
target_link_libraries(from_file_test Materials)
target_link_libraries(from_file_test Renderer)
target_link_libraries(from_file_test Config)
target_link_libraries(from_file_test Camera)

# Garantir que usaremos C++11 (ou superior, se preciso)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#set(CMAKE_BUILD_TYPE Debug)

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES all)
endif()

#set_target_properties(CUDA_Utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET sphere_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET Imagem PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET CUDA_Utils PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Renderer PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Sphere PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Plane PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Materials PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Camera PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET Config PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET save_image_test PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET sphere_test PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET materials_test PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET shapes_test PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET from_file_test PROPERTY CUDA_ARCHITECTURES all)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Automatizacao do Doxygen
find_package(Doxygen REQUIRED dot)

set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN EXTRACT_STATIC YES)
set(DOXYGEN_TEMPLATE_RELATIONS YES)
set(DOXYGEN_COLLABORATION_GRAPH YES)
set(DOXYGEN_UML_LOOK YES)
set(DOXYGEN_CLASS_GRAPH YES)

doxygen_add_docs(doc)
