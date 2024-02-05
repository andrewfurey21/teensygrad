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
//TODO: change to simple array, with max size, null terminated like a string. (kind of like ggml)
struct shape* create_shape(uint32_t* dims, uint32_t size);
struct shape* create_shape_1d(uint32_t dim);
struct shape* create_shape_2d(uint32_t dim1, uint32_t dim2);
void print_shape(struct shape* s);
void free_shape(struct shape* s);
//TODO:print shape, shape_to_str

enum Op {
    NOOP=0,
    RELU,
    ADD,
    MUL
};


//rename structs (teensygrad namespace, sort of)
struct tensor {
    struct shape* shape_b;
    float* buffer;
    uint64_t size;

    //maybe a context struct?
    struct tensor** parents;
    void (*pfn)(struct tensor*);
    enum Op op;

    bool calculate_grads;
    struct tensor* grads;
};

size_t op_radix(enum Op op);

struct tensor* create_tensor(struct shape* s, bool requires_grad, struct tensor** parents, enum Op op);
struct tensor* ones_tensor(struct shape* s, bool requires_grad, struct tensor** parents, enum Op op);
void destroy_tensor(struct tensor* t);
//TODO:
//format function, properly shaped like numpy
//scalar mul, scaled_uniform, random, dot product (maybe multiple ops make up dot product, like tinygrad?)
//cleanup, mort structures, multiple types (void* buffer), some more functions for creating/manipulating tensors

void print_t(struct tensor* t);
struct tensor* from_buffer(struct shape* s, float* buffer, bool requires_grads);
bool same_shape(struct tensor* a, struct tensor* b);

//elementwise ops
struct tensor* add_tensors(struct tensor* a, struct tensor* b, bool requires_grad);
void add_backwards(struct tensor* self);

struct tensor* mul_tensors(struct tensor* a, struct tensor* b);

struct tensor* relu_tensor(struct tensor* t);

void zero(struct tensor* t);

//backprop
void backwards(struct tensor* current);

//TODO: loss function

#endif
