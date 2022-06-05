#
cmake_minimum_required ( VERSION 3.11 )
# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # for multi-target pre-compilations
else()
  message(STATUS "No CUDA support")
endif()

# Nome do meu projeto
project(RayTracer LANGUAGES CUDA CXX)

set ( COMMON_ALL_CUDA
  includes/camera.cuh
  includes/imagem.cuh
  includes/scene.cuh
  includes/renderer.cuh
  includes/ray.cuh
  includes/vec3.cuh
  includes/cuda_parameters.cuh
  includes/color.cuh
  includes/config_reader.cuh
  includes/material.cuh
  includes/shape.cuh
  includes/utilities.cuh
  )

set ( SOURCE_ONE_WEEKEND_CUDA_4
  ${COMMON_ALL_CUDA}
  examples/draw_config.cu
)

add_executable(Config  ${SOURCE_ONE_WEEKEND_CUDA_4})
set_property(TARGET Config PROPERTY CUDA_ARCHITECTURES all)
target_include_directories(Config  PRIVATE .)
target_include_directories(Config  PRIVATE ./includes)


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
