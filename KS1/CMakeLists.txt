cmake_minimum_required(VERSION 2.6)
project(KS1_project)

include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE_EXE main.cpp)
set(SOURCE_LIB ks1.cpp)

add_library(ks1 STATIC ${SOURCE_LIB})
add_executable(KS1 ${SOURCE_EXE})

set(CMAKE_CXX_FLAGS "-Ofast -std=c++11 -march=native -fPIC -w -fopenmp")

target_link_libraries(KS1 ks1)
