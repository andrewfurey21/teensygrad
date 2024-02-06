#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

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

    struct shape* shape_copy = create_shape(s->dims, s->size);

    struct tensor* grads;
    if (requires_grad) {
        grads = create_tensor(s, false, NULL, NOOP);
    }

    struct tensor* t = (struct tensor*)malloc(sizeof(struct tensor));
    //TODO: copy everything!
    t->shape_b = shape_copy;
    t->buffer = buffer;
    t->size = size;
    t->calculate_grads = requires_grad;
    t->parents = parents;
    t->op = op;
    t->grads = grads;
    t->pfn = NULL;
    return t;
}

void destroy_tensor(struct tensor* t) {
    free(t->shape_b);
    free(t->buffer);
    free(t->parents);
    if (t->calculate_grads) {
        destroy_tensor(t->grads);
    }
    free(t);
}

struct tensor* ones_tensor(struct shape* s, bool requires_grad, struct tensor** parents, enum Op op) {
    struct tensor* ones = create_tensor(s, requires_grad, parents, op);
    for (size_t i = 0; i < ones->size; i++) {
        ones->buffer[i] = 1.0f;
    }
    return ones;
}

struct tensor* from_buffer(struct shape* s, float* buffer, bool requires_grads) {
    struct tensor* ret = (struct tensor*)malloc(sizeof(struct tensor));
    struct shape* shape_copy = create_shape(s->dims, s->size);
    ret->shape_b = shape_copy;
    uint64_t size = buflen(s);

    ret->buffer = buffer;
    ret->size = size;

    struct tensor* grads = NULL;
    if (requires_grads) {
        grads = create_tensor(s, false, NULL, NOOP);
    }
    ret->op = NOOP;
    ret->parents = NULL;
    ret->calculate_grads = requires_grads;
    ret->pfn = NULL;
    ret->grads = grads;
    return ret;
}


//TODO: clean up, print shape as well, better formatting
void print_t(struct tensor* t) {
    printf("Tensor:[");
    for (int i = 0; i < t->size; i++) {
        printf("%f, ", t->buffer[i]);
    }
    printf("]\n");
}

//TODO:zeroing out function
//TODO:take requires grad into account!
void add_backwards(struct tensor* self) {
    struct tensor* grads_0 = add_tensors(self->grads, self->parents[0]->grads, false);
    struct tensor* grads_1 = add_tensors(self->grads, self->parents[1]->grads, false);

    destroy_tensor(self->parents[0]->grads);
    destroy_tensor(self->parents[1]->grads);

    self->parents[0]->grads = grads_0;
    self->parents[1]->grads = grads_1;
}

struct tensor* add_tensors(struct tensor* a, struct tensor* b, bool requires_grad) {
    assert(same_shape(a, b) && "Tensors are not the same shape.");
    struct shape* shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);

    struct tensor** parents = (struct tensor**)malloc(op_radix(ADD)*sizeof(struct tensor*));
    parents[0] = a;
    parents[1] = b;

    struct tensor* t = create_tensor(shape_copy, requires_grad, parents, ADD);
    t->pfn = &add_backwards;
    //TODO:could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] + b->buffer[i];
    }

    return t;
}

void mul_backwards(struct tensor* self) {
    struct tensor* grads_1 = mul_tensors(self->grads, self->parents[0], false);
    struct tensor* grads_0 = mul_tensors(self->grads, self->parents[1], false);

    struct tensor* acc_grads_0 = add_tensors(grads_0, self->parents[0]->grads, false);
    struct tensor* acc_grads_1 = add_tensors(grads_1, self->parents[1]->grads, false);

    destroy_tensor(self->parents[0]->grads);
    destroy_tensor(self->parents[1]->grads);

    destroy_tensor(grads_1);
    destroy_tensor(grads_0);

    self->parents[0]->grads = acc_grads_0;
    self->parents[1]->grads = acc_grads_1;
}


struct tensor* mul_tensors(struct tensor* a, struct tensor* b, bool requires_grad) {
    assert(same_shape(a, b) && "Tensors are not the same shape.");
    struct shape* shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);

    //eh
    struct tensor** parents = (struct tensor**)malloc(op_radix(MUL)*sizeof(struct tensor*));
    parents[0] = a;
    parents[1] = b;

    struct tensor* t = create_tensor(shape_copy, requires_grad, parents, MUL);
    t->pfn = &mul_backwards;

    //could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * b->buffer[i];
    }

    return t;
}

void relu_backwards(struct tensor* self) {
    struct tensor* grads = create_tensor(self->shape_b, false, NULL, NOOP);
    for (size_t i = 0; i < self->parents[0]->size; i++) {
        if (grads->buffer[i] < self->parents[0]->buffer[i]) {
            grads->buffer[i] = 1;
        }
    }
    destroy_tensor(self->parents[0]->grads);
    self->parents[0]->grads = grads;
}

//TODO:should really be a max op
struct tensor* relu_tensor(struct tensor* a, bool requires_grad) {
    struct shape* shape_copy = create_shape(a->shape_b->dims, a->shape_b->size);

    struct tensor** parents = (struct tensor**)malloc(op_radix(RELU)*sizeof(struct tensor*));
    parents[0] = a;

    struct tensor* t = create_tensor(shape_copy, requires_grad, parents, RELU);
    t->pfn = &relu_backwards;

    //could easily be vectorized.
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}

void sum_reduce_backwards(struct tensor* self) {
    //struct tensor* grads = add_tensors(self->grads, self->parents[0]->grads, false);
    struct tensor* grads = ones_tensor(self->parents[0]->shape_b, false, NULL, NOOP);

    destroy_tensor(self->parents[0]->grads);

    self->parents[0]->grads = grads;
}

struct tensor* sum_reduce_tensors(struct tensor* a, bool requires_grad) {
    struct shape* shape_copy = create_shape_1d(1);

    struct tensor** parents = (struct tensor**)malloc(op_radix(SUM_REDUCE)*sizeof(struct tensor*));
    parents[0] = a;

    struct tensor* t = create_tensor(shape_copy, requires_grad, parents, SUM_REDUCE);
    t->pfn = &sum_reduce_backwards;

    //TODO:could easily be vectorized.
    double sum = 0.0f;
    for (uint64_t i = 0; i < a->size; i++) {
        sum += a->buffer[i];
    }

    t->buffer[0] = sum;

    return t;
}

void zero(struct tensor* t) {
    memset(t->buffer, 0, t->size);
}
