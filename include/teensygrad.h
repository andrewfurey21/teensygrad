#include "stdint.h"
#include "stdbool.h"
#include "stdlib.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

enum top {
    NOOP=0,
    RELU,
    NEG,
    SUM_REDUCE,
    ADD,
    MUL
};

size_t top_radix(enum top);

struct tshape {
    uint32_t* dims;
    uint32_t size;
};

struct tt {
    struct tshape* shape;
    float* buffer;
    uint64_t size;

    struct tt** parents;
    void (*_backwards)(struct tt*);
    enum top op;

    bool requires_grad;
    struct tt* grads;
};


bool tshape_compare(struct tt* a, struct tt* b);
struct tshape* tshape_create(uint32_t* dims, uint32_t size);
struct tshape* tshape_create_1d(uint32_t dim);
struct tshape* tshape_create_2d(uint32_t dim1, uint32_t dim2);
void tshape_print(struct tshape* s);
void tshape_destroy(struct tshape* s);


struct tt* tt_from_buffer(struct tshape* s, float* buffer, bool requires_grads);
struct tt* tt_zeros(struct tshape* s, bool requires_grad);
struct tt* tt_ones(struct tshape* s, bool requires_grad);
struct tt* tt_full_like(struct tshape* s, float fill_value, bool requires_grad);
struct tt* tt_scaled_uniform(struct tshape* s, float min, float max, bool requires_grad);
void tt_to_zeros(struct tt* t);
void tt_copy_buffer(struct tt* a, struct tt* b);
void tt_print(struct tt* t);
void tt_destroy(struct tt* t);

//elementwise ops
struct tt* tt_add(struct tt* a, struct tt* b, bool requires_grad);
struct tt* tt_neg(struct tt* a, bool requires_grad);
struct tt* tt_mul(struct tt* a, struct tt* b, bool requires_grad);
struct tt* tt_relu(struct tt* t, bool requires_grad);
//reduce ops
struct tt* tt_sum(struct tt* a, bool requires_grad);

//computational graph
struct tgraph {
    struct tt** nodes;
    size_t size;
    bool training;
};
struct tgraph* tgraph_build(struct tt* x);
//backprop
void tbackwards(struct tgraph* net);

//could have a void* with other params for different optimizers.
struct toptimizer {
    struct tt** params;
    uint64_t size;
    float learning_rate;
    void (*step)();
};

void tsgd(struct toptimizer* optim);
struct toptimizer* toptimizer_create(struct tt** params, uint64_t size, float lr, void (*step)(struct toptimizer*));

#endif
