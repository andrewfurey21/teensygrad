#include "stdint.h"
#include "stdarg.h"
#include "stdbool.h"
#include "stdlib.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

enum top {
    NOOP=0,
    RELU,
    NEG,
    SUM_REDUCE,
    RESHAPE,
    ADD,
    MUL
};

size_t top_radix(enum top);

struct tshape {
    int32_t* dims;
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


struct tshape* tshape_build(uint32_t size, ...);
void tshape_free(struct tshape* s);
struct tshape* tshape_copy(struct tshape* other);
bool tshape_compare(struct tt* a, struct tt* b);
void tshape_print(struct tshape* s);


struct tt* tt_from_buffer(struct tshape* s, float* buffer, bool requires_grads);
struct tt* tt_zeros(struct tshape* s, bool requires_grad);
struct tt* tt_ones(struct tshape* s, bool requires_grad);
struct tt* tt_fill(struct tshape* s, float fill_value, bool requires_grad);
struct tt* tt_linspace(struct tshape* s, float min, float max, bool requires_grad);
struct tt* tt_uniform(struct tshape* s, float min, float max, bool requires_grad);
struct tt* tt_uniformint(struct tshape* s, float min, float max, bool requires_grad);
struct tt* tt_copy(struct tt* original, bool requires_grad);
void tt_to_zeros(struct tt* t);
void tt_copy_buffer(struct tt* dest, struct tt* src);
void tt_print(struct tt* t);
void tt_free(struct tt* t);

//elementwise ops
struct tt* tt_add(struct tt* a, struct tt* b, bool requires_grad);
struct tt* tt_neg(struct tt* a, bool requires_grad);
struct tt* tt_mul(struct tt* a, struct tt* b, bool requires_grad);
struct tt* tt_relu(struct tt* t, bool requires_grad);
//reduce ops
struct tt* tt_sum(struct tt* a, bool requires_grad);
//movement ops
struct tt* tt_reshape(struct tt* a, struct tshape* new_shape, bool requires_grad);

//computational graph
struct tgraph {
    struct tt** nodes;
    size_t size;
    bool training;
};
struct tgraph* tgraph_build(struct tt* x);
void tgraph_free(struct tgraph* net);
void tgraph_zeroed(struct tgraph* net);
//backprop
void tgraph_backprop(struct tgraph* net);

struct toptimizer {
    struct tgraph* net;
    float learning_rate;// TODO:make part of step function call (struct)
    void (*step)();
};

struct toptimizer* toptimizer_build(struct tt** params, uint64_t size, float lr, void (*step)(struct toptimizer*));
void toptimizer_free(struct toptimizer* topt);

//optimization steps
void tsgd(struct toptimizer* optim);

#endif
