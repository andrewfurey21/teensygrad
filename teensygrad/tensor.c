#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"

#include "teensygrad.h"

uint64_t buflen(struct shape* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->dims[i];
    }
    return size;
}

struct tensor* create_tensor(struct shape* s, bool requires_grad, struct tensor** parents, enum Op op) {
    uint64_t size = buflen(s);
    float* buffer = (float*)calloc(size, size*(uint64_t)sizeof(float));


    struct tensor* grads;
    if (requires_grad) {
        grads = create_tensor(s, false, NULL, NOOP);
    }

    //struct tensor t = {s, buffer, size, requires_grad, parents, op, grads };
    struct tensor* t = (struct tensor*)malloc(sizeof(struct tensor*));
    t->shape_b = s;
    t->buffer = buffer;
    t->size = size;
    t->calculate_grads = requires_grad;
    t->parents = parents;
    t->op = op;
    t->grads = grads;
    return t;
}

struct tensor from_buffer(struct shape* s, float* buffer, bool requires_grads) {
    struct tensor ret;
    ret.shape_b = s;
    uint64_t size = buflen(s);

    ret.buffer = buffer;
    ret.size = size;

    ret.op = NOOP;
    ret.parents = NULL;
    ret.calculate_grads = requires_grads;
    return ret;
}


//TODO: clean up, print shape as well, better formatting
void print_t(struct tensor* t) {
    printf("Tensor buffer:[");
    for (int i = 0; i < t->size; i++) {
        printf("%f, ", t->buffer[i]);
    }
    printf("]\n");
}

bool same_shape(struct tensor* a, struct tensor* b) {
    if (a->shape_b->size != b->shape_b->size) {
        return false;
    }
    for (uint32_t i = 0; i < a->shape_b->size; i++) {
        if (a->shape_b->dims[i] != b->shape_b->dims[i]) {
            return false;
        }
    }
    return true;
}

struct tensor* add_tensors(struct tensor* a, struct tensor* b) {
    assert(same_shape(a, b) && "Tensors are not the same shape.");
    struct shape shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);


    struct tensor** parents = (struct tensor**)malloc(op_radix(ADD)*sizeof(struct tensor**));
    parents[0] = a;
    parents[1] = b;

    struct tensor* t = create_tensor(&shape_copy, true, parents, ADD);

    //could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] + b->buffer[i];
    }

    return t;
}

struct tensor* mul_tensors(struct tensor* a, struct tensor* b) {
    assert(same_shape(a, b) && "Tensors are not the same shape.");
    struct shape shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);

    //eh
    struct tensor** parents = (struct tensor**)malloc(op_radix(MUL)*sizeof(struct tensor**));
    parents[0] = a;
    parents[1] = b;

    struct tensor* t = create_tensor(&shape_copy, true, parents, MUL);

    //could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * b->buffer[i];
    }

    return t;
}

struct tensor* relu_tensor(struct tensor* a) {
    struct shape shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);

    struct tensor** parents = (struct tensor**)malloc(op_radix(RELU)*sizeof(struct tensor**));
    parents[0] = a;

    struct tensor* t = create_tensor(&shape_copy, true, parents, RELU);

    //could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}

