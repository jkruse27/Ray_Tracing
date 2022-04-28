# Versao minima do cmake
cmake_minimum_required(VERSION 3.0)

# Nome do meu projeto
project(RayTracer)

#
add_executable(save_image_test examples/save_image.cpp)
add_executable(sphere_test examples/draw_sphere.cpp)

# Especificar onde estao os header files
include_directories(includes/)

# Bibliotecas locais (estaticas)
add_library(Imagem src/imagem.cpp)
add_library(Renderer src/renderer.cpp)
add_library(Sphere src/sphere.cpp)

# Exemplo de criacao de imagem
target_link_libraries(save_image_test Imagem)

# Exemplo de geracao de imagem com circulo
target_link_libraries(sphere_test Imagem)
target_link_libraries(sphere_test Sphere)
target_link_libraries(sphere_test Renderer)

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
