cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")

set(SOURCE_EXE main.cpp)            
set(SOURCE_LIB ks2.cpp)

add_library(ks2 STATIC ${SOURCE_LIB})
add_executable(KS2 ${SOURCE_EXE})

# Clean up and modernize compiler flags
SET(CMAKE_CXX_FLAGS "-Ofast -std=c++11 -march=native -fPIC -w -fopenmp")

# Link libraries
target_link_libraries(KS2 ks2)
