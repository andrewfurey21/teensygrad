#include "assert.h"
#include "../include/teensygrad.h"

void teensy_sgd(struct toptimizer* optim) {
    for (uint64_t i = 0; i < optim->size; i++) {
        struct tt* t = optim->params[i];

        struct tt* lrs = tt_fill(t->shape, -optim->learning_rate, false);
        struct tt* updated_grads = tt_mul(lrs, t->grads, false);

        struct tt* updated_params = tt_add(updated_grads, t, false);

        tt_copy_buffer(t, updated_params);

        tt_free(lrs);
        tt_free(updated_grads);
        tt_free(updated_params);
    }
}

struct toptimizer* toptimizer_create(struct tt** params, uint64_t size, float lr, void (*step)(struct toptimizer*)) {
    assert(lr >= 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
    assert(size > 0 && "Must have 1 or more param.");
    struct toptimizer* optim = (struct toptimizer*)malloc(sizeof (struct toptimizer));
    optim->params = params;
    optim->learning_rate = lr;
    optim->step = step;
    optim->size = size;
    return optim;
}
