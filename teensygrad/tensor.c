#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

#include "teensygrad.h"

uint64_t buflen(struct teensy_shape* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->dims[i];
    }
    return size;
}

struct teensy_tensor* teensy_tensor_zeros(struct teensy_shape* s, bool requires_grad) {
    uint64_t size = buflen(s);
    float* buffer = (float*)calloc(size, size*(uint64_t)sizeof(float));

    struct teensy_shape* teensy_shape_copy = teensy_shape_create(s->dims, s->size);

    struct teensy_tensor* grads = NULL;
    if (requires_grad) {
        grads = teensy_tensor_zeros(s, false);
    }

    struct teensy_tensor* t = (struct teensy_tensor*)malloc(sizeof(struct teensy_tensor));

    t->shape = teensy_shape_copy;
    t->buffer = buffer;
    t->size = size;
    t->requires_grad = requires_grad;
    t->parents = NULL;
    t->op = NOOP;
    t->grads = grads;
    t->_backwards = NULL;
    return t;
}

struct teensy_tensor* teensy_tensor_ones(struct teensy_shape* s, bool requires_grad) {
    struct teensy_tensor* ones = teensy_tensor_zeros(s, requires_grad);
    for (size_t i = 0; i < ones->size; i++) {
        ones->buffer[i] = 1.0f;
    }
    return ones;
}

struct teensy_tensor* teensy_tensor_from_buffer(struct teensy_shape* s, float* buffer, bool requires_grad) {
    struct teensy_tensor* ret = (struct teensy_tensor*)malloc(sizeof(struct teensy_tensor));
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(s->dims, s->size);
    ret->shape = teensy_shape_copy;
    uint64_t size = buflen(s);

    ret->buffer = buffer;
    ret->size = size;

    struct teensy_tensor* grads = NULL;
    if (requires_grad) {
        grads = teensy_tensor_zeros(s, false);
    }
    ret->op = NOOP;
    ret->parents = NULL;
    ret->requires_grad = requires_grad;
    ret->_backwards = NULL;
    ret->grads = grads;
    return ret;
}

struct teensy_tensor* teensy_tensor_full_like(struct teensy_shape* s, float fill_value, bool requires_grad) {
    struct teensy_tensor* t = teensy_tensor_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = fill_value;
    }
    return t;
}

void teensy_tensor_to_zeros(struct teensy_tensor* t) {
    memset(t->buffer, 0, t->size);
}

bool teensy_tensor_same_shape(struct teensy_tensor* a, struct teensy_tensor* b) {
    if (a->shape->size != b->shape->size) {
        return false;
    }
    for (uint32_t i = 0; i < a->shape->size; i++) {
        if (a->shape->dims[i] != b->shape->dims[i]) {
            return false;
        }
    }
    return true;
}

void teensy_tensor_print(struct teensy_tensor* t) {
    if (t == NULL) {
        printf("tensor: (null)\n");
        return;
    }
    printf("tensor:[ ");
    for (int i = 0; i < t->size; i++) {
        printf("%f, ", t->buffer[i]);
    }
    printf("]\n");
}

void teensy_tensor_destroy(struct teensy_tensor* t) {
    teensy_shape_destroy(t->shape);
    free(t->buffer);
    free(t->parents);
    if (t->requires_grad) {
        teensy_tensor_destroy(t->grads);
    }
    free(t);
}

void _add_backwards(struct teensy_tensor* self) {
    if (self->parents[0]->requires_grad) {
        struct teensy_tensor* grads_0 = teensy_tensor_add(self->grads, self->parents[0]->grads, false);
        teensy_tensor_destroy(self->parents[0]->grads);
        self->parents[0]->grads = grads_0;
    }

    if (self->parents[1]->requires_grad) {
        struct teensy_tensor* grads_1 = teensy_tensor_add(self->grads, self->parents[1]->grads, false);
        teensy_tensor_destroy(self->parents[1]->grads);
        self->parents[1]->grads = grads_1;
    }
}

struct teensy_tensor* teensy_tensor_add(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad) {
    assert(teensy_tensor_same_shape(a, b) && "Tensors are not the same shape.");
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(ADD)*sizeof(struct teensy_tensor*));
    parents[0] = a;
    parents[1] = b;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad);
    t->parents = parents;
    t->op = ADD;
    t->_backwards = &_add_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] + b->buffer[i];
    }

    return t;
}

void _neg_backwards(struct teensy_tensor* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct teensy_tensor* grads = teensy_tensor_full_like(self->shape, -1.0f, false);
    struct teensy_tensor* mul_grads = teensy_tensor_mul(grads, self->grads, false);
    struct teensy_tensor* acc_grads = teensy_tensor_add(mul_grads, self->parents[0]->grads, false);
    teensy_tensor_destroy(self->parents[0]->grads);
    teensy_tensor_destroy(grads);
    teensy_tensor_destroy(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct teensy_tensor* teensy_tensor_neg(struct teensy_tensor* a, bool requires_grad) {
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(NEG)*sizeof(struct teensy_tensor*));
    parents[0] = a;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad);
    t->parents = parents;
    t->op = NEG;
    t->_backwards = &_neg_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = -a->buffer[i];
    }

    return t;
}

void _mul_backwards(struct teensy_tensor* self) {
    if (self->parents[0]->requires_grad) {
        struct teensy_tensor* grads_0 = teensy_tensor_mul(self->grads, self->parents[1], false);
        struct teensy_tensor* acc_grads_0 = teensy_tensor_add(grads_0, self->parents[0]->grads, false);
        teensy_tensor_destroy(self->parents[0]->grads);
        teensy_tensor_destroy(grads_0);
        self->parents[0]->grads = acc_grads_0;
    }

    if (self->parents[1]->requires_grad) {
        struct teensy_tensor* grads_1 = teensy_tensor_mul(self->grads, self->parents[0], false);
        struct teensy_tensor* acc_grads_1 = teensy_tensor_add(grads_1, self->parents[1]->grads, false);
        teensy_tensor_destroy(self->parents[1]->grads);
        teensy_tensor_destroy(grads_1);
        self->parents[1]->grads = acc_grads_1;
    }
}


struct teensy_tensor* teensy_tensor_mul(struct teensy_tensor* a, struct teensy_tensor* b, bool requires_grad) {
    assert(teensy_tensor_same_shape(a, b) && "Tensors are not the same shape.");
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(MUL)*sizeof(struct teensy_tensor*));
    parents[0] = a;
    parents[1] = b;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad);
    t->parents = parents;
    t->op = MUL;
    t->_backwards = &_mul_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * b->buffer[i];
    }

    return t;
}

void _relu_backwards(struct teensy_tensor* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }

    struct teensy_tensor* grads = teensy_tensor_zeros(self->shape, false);
    for (size_t i = 0; i < self->parents[0]->size; i++) {
        if (grads->buffer[i] < self->parents[0]->buffer[i]) {
            grads->buffer[i] = 1;
        }
    }
    //TODO:refactor this into a function
    struct teensy_tensor* mul_grads = teensy_tensor_mul(self->grads, grads, false);
    struct teensy_tensor* acc_grads = teensy_tensor_add(self->parents[0]->grads, mul_grads, false);
    teensy_tensor_destroy(grads);
    teensy_tensor_destroy(self->parents[0]->grads);
    teensy_tensor_destroy(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct teensy_tensor* teensy_tensor_relu(struct teensy_tensor* a, bool requires_grad) {
    struct teensy_shape* teensy_shape_copy = teensy_shape_create(a->shape->dims, a->shape->size);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(RELU)*sizeof(struct teensy_tensor*));
    parents[0] = a;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad);
    t->parents = parents;
    t->op = RELU;
    t->_backwards = &_relu_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}

void _sum_backwards(struct teensy_tensor* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct teensy_tensor* grads = teensy_tensor_ones(self->parents[0]->shape, false);
    //TODO:Expand
    struct teensy_tensor* expanded_grads = teensy_tensor_full_like(grads->shape, self->grads->buffer[0], false);
    struct teensy_tensor* mul_grads = teensy_tensor_mul(expanded_grads, grads, false);
    struct teensy_tensor* acc_grads = teensy_tensor_add(self->parents[0]->grads, mul_grads, false);

    teensy_tensor_destroy(grads);
    teensy_tensor_destroy(mul_grads);
    teensy_tensor_destroy(self->parents[0]->grads);

    self->parents[0]->grads = acc_grads;
}

struct teensy_tensor* teensy_tensor_sum(struct teensy_tensor* a, bool requires_grad) {
    struct teensy_shape* teensy_shape_copy = teensy_shape_create_1d(1);

    struct teensy_tensor** parents = (struct teensy_tensor**)malloc(op_radix(SUM_REDUCE)*sizeof(struct teensy_tensor*));
    parents[0] = a;

    struct teensy_tensor* t = teensy_tensor_zeros(teensy_shape_copy, requires_grad);
    t->parents = parents;
    t->op = SUM_REDUCE;
    t->_backwards = &_sum_backwards;

    double sum = 0.0f;
    for (uint64_t i = 0; i < a->size; i++) {
        sum += a->buffer[i];
    }

    t->buffer[0] = sum;

    return t;
}
