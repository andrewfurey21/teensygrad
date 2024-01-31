cmake_minimum_required(VERSION 3.28)
project(Teensygrad VERSION 0.1)

set(CMAKE_CC_FLAGS "-Wall -Werror")

set(LIBRARY
  teensygrad/tensor.c
  teensygrad/teensygrad.h
)
add_library(teensygrad ${LIBRARY})

add_executable(tensor_example examples/tensor_example.c)

target_link_libraries(tensor_example teensygrad)
