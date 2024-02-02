#include "stdint.h"
#include "stdbool.h"
#include "stdlib.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

struct shape {
    uint32_t* dims;
    uint32_t size;
    char* str;
};

struct shape create_shape(uint32_t* dims, uint32_t size);
struct shape create_shape_1d(uint32_t dim);
struct shape create_shape_2d(uint32_t dim1, uint32_t dim2);
void free_shape(struct shape* s);
//TODO:print shape, shape_to_str

struct tensor {
    struct shape* shape_b;
    float* buffer;
    uint64_t size;
};

struct tensor create_tensor(struct shape* s);
//TODO:
//print function, properly shaped like numpy
void print_t(struct tensor* t);
//struct tensor random_tensor(struct shape* s, size_t min, size_t max);
struct tensor from_buffer(struct shape* s, float* buffer);
bool same_shape(struct tensor* a, struct tensor* b);

//elementwise ops
struct tensor add_tensors(struct tensor* a, struct tensor* b);
struct tensor mul_tensor(struct tensor* a, struct tensor* b);
struct tensor relu_tensor(struct tensor* t);

#endif
