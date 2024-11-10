#include "stdint.h"
#include "stdarg.h"
#include "stdbool.h"
#include "stdlib.h"
#include <cstdint>
#include <stdint.h>

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H


enum top {//need enough for: conv, batchnorm, maxpool, linear, relu
    NOOP=0,
    RELU,// TODO: remove, use min/max instead.
    NEG,
    SUM_REDUCE,
    RESHAPE,
    EXPAND,
    ADD,
    MUL,
};

size_t top_radix(enum top);
void print_op_string(enum top op);

typedef struct {
    int32_t* items;
    uint32_t size;
} ttuple;

ttuple* ttuple_build(uint32_t size, ...);
ttuple* ttuple_ones(uint32_t size);
uint64_t ttuple_mul(ttuple* s);
ttuple* ttuple_copy(ttuple* other);
bool ttuple_equal(ttuple* a, ttuple* b);
void ttuple_free(ttuple* s);
void ttuple_print(ttuple* s);

//ttuple* ttuple_permute(ttuple* shape, ttuple* axes);
//bool tshape_duplicates(struct tshape* axes);

typedef struct {
    float* buffer;
    uint64_t refcount;
    uint64_t size;
} tstorage;

typedef struct {
    ttuple* shape;
    ttuple* strides;
    uint64_t offset;
} tview;

typedef struct tt tt;
struct tt {
    tstorage* data;
    tview* view;

    tt** parents;
    void (*_backwards)(tt*);
    enum top op;

    bool requires_grad;
    struct tt* grads;
};


// TODO: empty, logical index to physical index, setitem/item, arange, tostring (cache in repr, use inside print), view/reshape

tt* tt_zeros(ttuple* s, bool requires_grad);
tt* tt_ones(ttuple* s, bool requires_grad);
tt* tt_from_buffer(ttuple* s, float* buffer, bool requires_grads);
tt* tt_fill(ttuple* s, float fill_value, bool requires_grad);
tt* tt_linspace(ttuple* s, float min, float max, bool requires_grad);
tt* tt_uniform(ttuple* s, float min, float max, bool requires_grad);
tt* tt_uniformint(ttuple* s, float min, float max, bool requires_grad);
void tt_copy_buffer(tt* dest, tt* src);
tt* tt_copy(tt* original, bool requires_grad);
void tt_to_zeros(tt* t);
void tt_to_n(struct tt* t, float n);
tt* tt_slice();//, good way to write function for slicing?
void tt_print(tt* t);
void tt_free(tt* t);
//tt* tt_reshape (tt* t);//gunna get tricky with nested views. looping etc. reshape like numpy/tinygrad.

//elementwise ops
tt* tt_add(tt* a, tt* b);
tt* tt_neg(tt* a);
tt* tt_mul(tt* a, tt* b);
tt* tt_max(tt* a, tt* b);
//reduce ops
tt* tt_sum(tt* a, ttuple* axes);
//movement ops
tt* tt_permute(tt* t, ttuple* axes);
tt* tt_expand(tt* a, ttuple* shape);
tt* tt_reshape(tt* a, ttuple* new_shape);

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

//nn
struct toptimizer {
    struct tgraph* net;
    struct toptimizer_params* opt_params;
    void (*step)(struct toptimizer* optim);
};

struct toptimizer_params {
    float learning_rate;
};

struct toptimizer* toptimizer_build(struct tt** params, uint64_t size, struct toptimizer_params* opt_params , void (*step)(struct toptimizer*));
void toptimizer_free(struct toptimizer* topt);

//optimization steps
void tsgd(struct toptimizer* optim);
// TODO: adam
#endif
