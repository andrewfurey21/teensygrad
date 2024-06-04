#include "stdlib.h"
#include "time.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

#include "../include/teensygrad.h"

uint64_t buflen(struct tshape* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->dims[i];
    }
    return size;
}

struct tt* tt_zeros(struct tshape* s, bool requires_grad) {
    uint64_t size = buflen(s);
    float* buffer = (float*)calloc(size, size*(uint64_t)sizeof(float));

    struct tshape* copy = tshape_copy(s);

    struct tt* grads = NULL;
    if (requires_grad) {
        grads = tt_zeros(s, false);
    }

    struct tt* t = (struct tt*)malloc(sizeof(struct tt));

    t->shape = copy;
    t->buffer = buffer;
    t->size = size;
    t->requires_grad = requires_grad;
    t->parents = NULL;
    t->op = NOOP;
    t->grads = grads;
    t->_backwards = NULL;
    return t;
}

struct tt* tt_ones(struct tshape* s, bool requires_grad) {
    struct tt* ones = tt_zeros(s, requires_grad);
    for (size_t i = 0; i < ones->size; i++) {
        ones->buffer[i] = 1.0f;
    }
    return ones;
}

struct tt* tt_from_buffer(struct tshape* s, float* buffer, bool requires_grad) {
    struct tt* ret = (struct tt*)malloc(sizeof(struct tt));
    struct tshape* copy = tshape_copy(s);
    ret->shape = copy;
    uint64_t size = buflen(s);

    ret->buffer = buffer;
    ret->size = size;

    struct tt* grads = NULL;
    if (requires_grad) {
        grads = tt_zeros(s, false);
    }
    ret->op = NOOP;
    ret->parents = NULL;
    ret->requires_grad = requires_grad;
    ret->_backwards = NULL;
    ret->grads = grads;
    return ret;
}

struct tt* tt_full_like(struct tshape* s, float fill_value, bool requires_grad) {
    struct tt* t = tt_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = fill_value;
    }
    return t;
}

struct tt* tt_scaled_uniform(struct tshape* s, float min, float max, bool requires_grad) {
    struct tt* t = tt_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = (max-min)/(float)t->size*i+min;//-fabs(min);
    }
    return t;
}

void tt_to_zeros(struct tt* t) {
    memset(t->buffer, 0, t->size*4);
}

void tt_print(struct tt* t) {
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

void tt_copy_buffer(struct tt* a, struct tt* b) {
    assert(tshape_compare(a, b) && "Tensors are not the same shape.");
    for (uint64_t i = 0; i < a->size; i++) {
        a->buffer[i] = b->buffer[i];
    }
}

void tt_destroy(struct tt* t) {
    tshape_destroy(t->shape);
    free(t->buffer);
    free(t->parents);
    if (t->requires_grad) {
        tt_destroy(t->grads);
    }
    free(t);
}

void _add_backwards(struct tt* self) {
    if (self->parents[0]->requires_grad) {
        struct tt* grads_0 = tt_add(self->grads, self->parents[0]->grads, false);
        tt_destroy(self->parents[0]->grads);
        self->parents[0]->grads = grads_0;
    }

    if (self->parents[1]->requires_grad) {
        struct tt* grads_1 = tt_add(self->grads, self->parents[1]->grads, false);
        tt_destroy(self->parents[1]->grads);
        self->parents[1]->grads = grads_1;
    }
}

struct tt* tt_add(struct tt* a, struct tt* b, bool requires_grad) {
    assert(tshape_compare(a, b) && "Tensors are not the same shape.");
    struct tshape* copy = tshape_copy(a->shape);

    struct tt** parents = NULL;
    //irrelevant if not requires_grad
    if (requires_grad) {
        parents = (struct tt**)malloc(top_radix(ADD)*sizeof(struct tt*));
        parents[0] = a;
        parents[1] = b;
    }

    struct tt* t = tt_zeros(copy, requires_grad);
    t->parents = parents;
    t->op = ADD;
    t->_backwards = &_add_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] + b->buffer[i];
    }

    return t;
}

void _neg_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct tt* grads = tt_full_like(self->shape, -1.0f, false);
    struct tt* mul_grads = tt_mul(grads, self->grads, false);
    struct tt* acc_grads = tt_add(mul_grads, self->parents[0]->grads, false);
    tt_destroy(self->parents[0]->grads);
    tt_destroy(grads);
    tt_destroy(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct tt* tt_neg(struct tt* a, bool requires_grad) {
    struct tshape* copy = tshape_copy(a->shape);

    struct tt** parents = NULL;
    if (requires_grad) {
        parents = (struct tt**)malloc(top_radix(NEG)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(copy, requires_grad);
    t->parents = parents;
    t->op = NEG;
    t->_backwards = &_neg_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = -a->buffer[i];
    }

    return t;
}

void _mul_backwards(struct tt* self) {
    if (self->parents[0]->requires_grad) {
        struct tt* grads_0 = tt_mul(self->grads, self->parents[1], false);
        struct tt* acc_grads_0 = tt_add(grads_0, self->parents[0]->grads, false);
        tt_destroy(self->parents[0]->grads);
        tt_destroy(grads_0);
        self->parents[0]->grads = acc_grads_0;
    }

    if (self->parents[1]->requires_grad) {
        struct tt* grads_1 = tt_mul(self->grads, self->parents[0], false);
        struct tt* acc_grads_1 = tt_add(grads_1, self->parents[1]->grads, false);
        tt_destroy(self->parents[1]->grads);
        tt_destroy(grads_1);
        self->parents[1]->grads = acc_grads_1;
    }
}


struct tt* tt_mul(struct tt* a, struct tt* b, bool requires_grad) {
    assert(tshape_compare(a, b) && "Tensors are not the same shape.");
    struct tshape* copy = tshape_copy(a->shape);

    struct tt** parents = NULL;
    if (requires_grad) {
        parents = (struct tt**)malloc(top_radix(MUL)*sizeof(struct tt*));
        parents[0] = a;
        parents[1] = b;
    }

    struct tt* t = tt_zeros(copy, requires_grad);
    t->parents = parents;
    t->op = MUL;
    t->_backwards = &_mul_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * b->buffer[i];
    }

    return t;
}

void _relu_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }

    struct tt* grads = tt_zeros(self->shape, false);
    for (size_t i = 0; i < self->parents[0]->size; i++) {
        if (grads->buffer[i] < self->parents[0]->buffer[i]) {
            grads->buffer[i] = 1;
        }
    }
    //TODO:refactor this into a function
    struct tt* mul_grads = tt_mul(self->grads, grads, false);
    struct tt* acc_grads = tt_add(self->parents[0]->grads, mul_grads, false);
    tt_destroy(grads);
    tt_destroy(self->parents[0]->grads);
    tt_destroy(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct tt* tt_relu(struct tt* a, bool requires_grad) {
    struct tshape* copy = tshape_copy(a->shape);
    struct tt**  parents = NULL;
    if (requires_grad) {
        parents = (struct tt**)malloc(top_radix(RELU)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(copy, requires_grad);
    t->parents = parents;
    t->op = RELU;
    t->_backwards = &_relu_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}

void _sum_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct tt* grads = tt_ones(self->parents[0]->shape, false);
    //TODO:Expand
    struct tt* expanded_grads = tt_full_like(grads->shape, self->grads->buffer[0], false);
    struct tt* mul_grads = tt_mul(expanded_grads, grads, false);
    struct tt* acc_grads = tt_add(self->parents[0]->grads, mul_grads, false);

    tt_destroy(grads);
    tt_destroy(mul_grads);
    tt_destroy(self->parents[0]->grads);

    self->parents[0]->grads = acc_grads;
}

struct tt* tt_sum(struct tt* a, bool requires_grad) {
    struct tshape* shape = tshape_create(1, 1);
    struct tt** parents = NULL;
    if (requires_grad) {
        parents = (struct tt**)malloc(top_radix(SUM_REDUCE)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(shape, requires_grad);
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
