cmake_minimum_required(VERSION 3.15)
project(gcs_task LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
find_package(Python3 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

# MOSEK 경로 환경에 맞게 수정
include_directories($ENV{HOME}/mosek/mosek/11.0/tools/platform/linux64x86/h)
link_directories($ENV{HOME}/mosek/mosek/11.0/tools/platform/linux64x86/bin)

pybind11_add_module(gcs_task gcs_task.cpp)
target_link_libraries(gcs_task PRIVATE mosek64)
