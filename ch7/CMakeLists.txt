cmake_minimum_required(VERSION 2.8)

project(ch7)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_executable(ch7 src/test_ransac.cpp)
target_link_libraries(ch7 ${OpenCV_LIBS})
