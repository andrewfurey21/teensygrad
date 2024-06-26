#include "stdlib.h"
#include "time.h"
#include "stdint.h"
#include "stdbool.h"
#include "stdio.h"
#include "assert.h"
#include "string.h"

#include "../include/teensygrad.h"

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

struct tt* tt_fill(struct tshape* s, float fill_value, bool requires_grad) {
    struct tt* t = tt_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = fill_value;
    }
    return t;
}

struct tt* tt_linspace(struct tshape* s, float min, float max, bool requires_grad) {
    struct tt* t = tt_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = (max-min)/(float)t->size*i+min;//-fabs(min);
    }
    return t;
}

struct tt* tt_uniform(struct tshape* s, float min, float max, bool requires_grad) {
    struct tt* t = tt_zeros(s, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = (float)rand()/(float)RAND_MAX * (max-min) + min;
    }
    return t;
}

struct tt* tt_uniformint(struct tshape* s, float min, float max, bool requires_grad) {
    struct tt* t = tt_uniform(s, min, max, requires_grad);
    for (uint64_t i = 0; i < t->size; i++) {
        t->buffer[i] = (float)(int)t->buffer[i];
    }
    return t;
}

void tt_to_zeros(struct tt* t) {
    memset(t->buffer, 0, t->size*4);
}

void tt_to_n(struct tt* t, float n) {
    for (uint32_t i = 0; i < t->size; i++) {
        t->buffer[i] = n;
    }
}

void tt_print(struct tt* t) {
    printf("teensy tensor: \n  ");
    if (!t) {
        printf("values: (null)\n");
        return;
    }
    tshape_print(t->shape);
    if (t->requires_grad) {
        printf("  op: ");
        print_op_string(t->op);
    }
    printf("  values: [ ");
    for (int i = 0; i < t->size; i++) {
        printf("%f, ", t->buffer[i]);
    }
    printf("]\n");
}

void tt_copy_buffer(struct tt* dest, struct tt* src) {
    assert(tshape_equal(dest->shape, src->shape) && "Tensors are not the same shape.");
    for (uint64_t i = 0; i < dest->size; i++) {
        dest->buffer[i] = src->buffer[i];
    }
}

struct tt* tt_copy(struct tt* original, bool requires_grad) {
    struct tshape* shape_copy = tshape_copy(original->shape);
    struct tt* tensor_copy = tt_zeros(shape_copy, requires_grad);
    tt_copy_buffer(tensor_copy, original);
    return tensor_copy;
}

void tt_free(struct tt* t) {
    tshape_free(t->shape);
    free(t->buffer);
    free(t->parents);
    if (t->requires_grad) {
        tt_free(t->grads);//make sure grads cant have grads
    }
    free(t);
}

// Unary ops
void _neg_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct tt* grads = tt_fill(self->shape, -1.0f, false);
    struct tt* mul_grads = tt_mul(grads, self->grads);
    struct tt* acc_grads = tt_add(mul_grads, self->parents[0]->grads);
    tt_free(self->parents[0]->grads);
    tt_free(grads);
    tt_free(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct tt* tt_neg(struct tt* a) {
    struct tshape* copy = tshape_copy(a->shape);

    struct tt** parents = NULL;
    if (a->requires_grad) {
        parents = (struct tt**)malloc(top_radix(NEG)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(copy, a->requires_grad);
    t->parents = parents;
    t->op = NEG;
    t->_backwards = &_neg_backwards;
    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = -a->buffer[i];
    }

    return t;
}

void _relu_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }

    struct tt* grads = tt_zeros(self->shape, false);
    for (size_t i = 0; i < self->parents[0]->size; i++) {
        if (self->parents[0]->buffer[i] > 0) {
            grads->buffer[i] = 1;
        }
    }
    struct tt* mul_grads = tt_mul(self->grads, grads);
    struct tt* acc_grads = tt_add(self->parents[0]->grads, mul_grads);
    tt_free(grads);
    tt_free(self->parents[0]->grads);
    tt_free(mul_grads);
    self->parents[0]->grads = acc_grads;
}

struct tt* tt_relu(struct tt* a) {
    struct tshape* copy = tshape_copy(a->shape);
    struct tt**  parents = NULL;
    if (a->requires_grad) {
        parents = (struct tt**)malloc(top_radix(RELU)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(copy, a->requires_grad);
    t->parents = parents;
    t->op = RELU;
    t->_backwards = &_relu_backwards;

    for (uint64_t i = 0; i < a->size; i++) {
        t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
    }

    return t;
}
// Binary ops
void _add_backwards(struct tt* self) {
    if (self->parents[0]->requires_grad) {
        //self->grads are the grads, must accumulate.
        struct tt* grads_0 = tt_add(self->grads, self->parents[0]->grads);
        tt_free(self->parents[0]->grads);
        self->parents[0]->grads = grads_0;
    }

    if (self->parents[1]->requires_grad) {
       struct tt* grads_1 = tt_add(self->grads, self->parents[1]->grads);
        tt_free(self->parents[1]->grads);
        self->parents[1]->grads = grads_1;
    }
}

struct tt* tt_add(struct tt* a, struct tt* b) {
    assert(tshape_equal(a->shape, b->shape) && "Tensors are not the same shape.");
    struct tshape* copy = tshape_copy(a->shape);
    bool requires_grad = a->requires_grad || b->requires_grad;

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

void _mul_backwards(struct tt* self) {
    if (self->parents[0]->requires_grad) {
        struct tt* grads_0 = tt_mul(self->grads, self->parents[1]);
        struct tt* acc_grads_0 = tt_add(grads_0, self->parents[0]->grads);
        tt_free(self->parents[0]->grads);
        tt_free(grads_0);
        self->parents[0]->grads = acc_grads_0;
    }

    if (self->parents[1]->requires_grad) {
        struct tt* grads_1 = tt_mul(self->grads, self->parents[0]);
        struct tt* acc_grads_1 = tt_add(grads_1, self->parents[1]->grads);
        tt_free(self->parents[1]->grads);
        tt_free(grads_1);
        self->parents[1]->grads = acc_grads_1;
    }
}

struct tt* tt_mul(struct tt* a, struct tt* b) {
    assert(tshape_equal(a->shape, b->shape) && "Tensors are not the same shape.");
    struct tshape* copy = tshape_copy(a->shape);

    bool requires_grad = a->requires_grad || b->requires_grad;
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

// Reduce ops
void _sum_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct tt* expanded_grads = tt_fill(self->parents[0]->shape, self->grads->buffer[0], false);
    struct tt* acc_grads = tt_add(self->parents[0]->grads, expanded_grads);

    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);

    self->parents[0]->grads = acc_grads;
}

struct tt* tt_sum(struct tt* a, struct tshape* axes) {
    struct tshape* new_shape = tshape_build(1, 1);
    struct tt** parents = NULL;
    if (a->requires_grad) {
        parents = (struct tt**)malloc(top_radix(SUM_REDUCE)*sizeof(struct tt*));
        parents[0] = a;
    }

    struct tt* t = tt_zeros(new_shape, a->requires_grad);
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

// Movement ops
// Permute
// void _permute_backwards(struct tt* self) {
//
// }
//
// struct tt* tt_permute(struct tt* t, struct tshape* axes) {
//     assert(t->shape->size == axes->size);
//     struct tt* tensor_copy = tt_copy(t, t->requires_grad);
//     struct tshape* permuted_shape = tshape_permute(t->shape, axes);
//     uint64_t buf_size = buflen(tensor_copy->shape);
//
//     for (uint64_t i = 0; i < buf_size; i++) {
//         size_t new_index = 0;
//         for (int j = 0; j < axes->size; j++) {
//             uint32_t shape_coord = permuted_shape->dims[j];
//             uint32_t axis = axes->dims[i];
//             uint32_t old_coord =
//             new_index += axis*shape_coord*old_coord;
//         }
//         tensor_copy->buffer[new_index] = t->buffer[i];
//
//     }
//     free(tensor_copy->shape);
//     tensor_copy->shape = permuted_shape;
//     return tensor_copy;
// }

// Expand
void _expand_backwards(struct tt* self) {

}

struct tt* tt_expand(struct tt* a, struct tshape* shape) {
    int diff = shape->size - a->shape->size;
    assert(diff >= 0 && "shape must have higher dimensions");
    struct tt** parents = NULL;

    for (int i = 0; i < a->shape->size; i++) {
        assert(shape->dims[i+diff] == a->shape->dims[i]);
    }
    struct tt* expanded_tensor = tt_zeros(shape, a->requires_grad);

    for (uint32_t i = 0; i < expanded_tensor->size; i++) {
        expanded_tensor->buffer[i] = a->buffer[i % a->size];
    }

    if (a->requires_grad) {
        parents = (struct tt**)malloc(top_radix(EXPAND)*sizeof(struct tt*));
        parents[0] = a;
    }

    expanded_tensor->parents = parents;
    expanded_tensor->_backwards = &_expand_backwards;
    expanded_tensor->op = EXPAND;

    return expanded_tensor;
}

// Reshape
void _reshape_backwards(struct tt* self) {
    if (!self->parents[0]->requires_grad) {
        return;
    }
    struct tt* grads = tt_reshape(self->grads, self->parents[0]->shape);
    struct tt* acc_grads = tt_add(grads, self->parents[0]->grads);

    free(grads);
    self->parents[0]->grads = acc_grads;
}

struct tt* tt_reshape(struct tt* a, struct tshape* new_shape) {
    struct tshape* new_shape_copy = tshape_copy(new_shape);
    assert(buflen(new_shape) == buflen(a->shape));
    struct tt** parents = NULL;
    struct tt* reshaped_grads = NULL;
    if (a->requires_grad) {
        parents = (struct tt**)malloc(top_radix(RESHAPE)* sizeof(struct tt*));
        parents[0] = a;
        reshaped_grads = tt_reshape(a->grads, new_shape_copy);
    }
    struct tt* t = tt_copy(a, a->requires_grad);
    free(t->grads);
    t->shape = new_shape_copy;
    t->parents = parents;
    t->op = RESHAPE;
    t->_backwards = &_reshape_backwards;
    t->grads = reshaped_grads;
    return t;
}


