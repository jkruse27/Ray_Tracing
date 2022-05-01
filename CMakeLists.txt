# Versao minima do cmake
cmake_minimum_required(VERSION 3.0)

# Nome do meu projeto
project(RayTracer)

#
add_executable(save_image_test examples/save_image.cpp)
add_executable(sphere_test examples/draw_sphere.cpp)
add_executable(materials_test examples/draw_materials.cpp)
add_executable(shapes_test examples/draw_shapes.cpp)

# Especificar onde estao os header files
include_directories(includes/)

# Bibliotecas locais (estaticas)
add_library(Imagem src/imagem.cpp)
add_library(Renderer src/renderer.cpp)
add_library(Sphere src/sphere.cpp)
add_library(Plane src/plane.cpp)
add_library(Materials src/material.cpp)

# Exemplo de criacao de imagem
target_link_libraries(save_image_test Imagem)

# Exemplo de geracao de imagem com circulo
target_link_libraries(sphere_test Imagem)
target_link_libraries(sphere_test Sphere)
target_link_libraries(sphere_test Materials)
target_link_libraries(sphere_test Renderer)

# Exemplo de geracao de imagem com difrentes materiais
target_link_libraries(materials_test Imagem)
target_link_libraries(materials_test Sphere)
target_link_libraries(materials_test Materials)
target_link_libraries(materials_test Renderer)

# Exemplo de geracao de imagem com difrentes formas
target_link_libraries(shapes_test Imagem)
target_link_libraries(shapes_test Sphere)
target_link_libraries(shapes_test Plane)
target_link_libraries(shapes_test Materials)
target_link_libraries(shapes_test Renderer)

# Garantir que usaremos C++11 (ou superior, se preciso)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

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
