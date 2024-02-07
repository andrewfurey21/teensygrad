#include "stdint.h"
#include "stdbool.h"
#include "stdlib.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

enum teensy_op {
    NOOP=0,
    RELU,
    NEG,
    SUM_REDUCE,
    ADD,
    MUL
};

size_t op_radix(enum teensy_op op);

struct teensy_shape {
    uint32_t* dims;
    uint32_t size;
};

struct teensy_shape* teensy_shape_create(uint32_t* dims, uint32_t size);
struct teensy_shape* teensy_shape_create_1d(uint32_t dim);
struct teensy_shape* teensy_shape_create_2d(uint32_t dim1, uint32_t dim2);
void teensy_shape_print(struct teensy_shape* s);
void teensy_shape_destroy(struct teensy_shape* s);


struct teensy_tensor {
    struct teensy_shape* shape;
    float* buffer;
    uint64_t size;

    struct teensy_tensor** parents;
    void (*_backwards)(struct teensy_tensor*);
    enum teensy_op op;

    bool requires_grad;
    struct teensy_tensor* grads;
};

struct teensy_tensor* teensy_tensor_from_buffer(struct teensy_shape* s, float* buffer, bool requires_grads);
struct teensy_tensor* teensy_tensor_zeros(struct teensy_shape* s, bool requires_grad);
struct teensy_tensor* teensy_tensor_ones(struct teensy_shape* s, bool requires_grad);
struct teensy_tensor* teensy_tensor_full_like(struct teensy_shape* s, float fill_value, bool requires_grad);
void teensy_tensor_to_zeros(struct teensy_tensor* t);
bool teensy_tensor_same_shape(struct teensy_tensor* a, struct teensy_tensor* b);
void teensy_tensor_print(struct teensy_tensor* t);
void teensy_tensor_destroy(struct teensy_tensor* t);

//elementwise ops
struct teensy_tensor* teensy_tensor_add(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad);
struct teensy_tensor* teensy_tensor_neg(struct teensy_tensor* a, bool requires_grad);
struct teensy_tensor* teensy_tensor_mul(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad);
struct teensy_tensor* teensy_tensor_relu(struct teensy_tensor* t, bool requires_grad);
//reduce ops
struct teensy_tensor* teensy_tensor_sum(struct teensy_tensor* a, bool requires_grad);
//backprop
void teensy_backwards(struct teensy_tensor* current);

struct teensy_optimizer {
    struct teensy_tensor** params;
    float learning_rate;
    void (*step)();
};

void teensy_sgd(struct teensy_optimizer* optim);
struct teensy_optimizer* teensy_optimizer_create(struct teensy_tensor** params, float learning_rate, void (*step)(struct teensy_optimizer*));

#endif
