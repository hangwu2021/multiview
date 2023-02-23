cmake_minimum_required(VERSION 3.0)

project(ch5)

# OpenCV 
find_package(OpenCV REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")

add_executable(stereo_disparty stereo.cpp)
target_link_libraries(stereo_disparty ${OpenCV_LIBS})
