cmake_minimum_required(VERSION 3.22)
project(Teensygrad VERSION 0.1)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -Werror")

set(LIBRARY
  teensygrad/shape.c
  teensygrad/tensor.c
  teensygrad/teensygrad.h
)
add_library(teensy ${LIBRARY})

add_executable(tensor_example examples/tensor_example.c)
target_link_libraries(tensor_example teensy)
