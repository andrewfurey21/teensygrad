#include "assert.h"
#include "../include/teensygrad.h"

void tsgd(struct toptimizer* optim) {
    for (uint64_t i = 0; i < optim->net->size; i++) {
        struct tt* t = optim->net->nodes[i];

        struct tt* lrs = tt_fill(t->shape, -optim->opt_params->learning_rate, false);
        struct tt* updated_grads = tt_mul(lrs, t->grads);

        struct tt* updated_params = tt_add(updated_grads, t);

        tt_copy_buffer(t, updated_params);

        tt_free(lrs);
        tt_free(updated_grads);
        tt_free(updated_params);
    }
}

struct toptimizer* toptimizer_create(struct tgraph* net, uint64_t size, struct toptimizer_params* opt_params, void (*step)(struct toptimizer*)) {
    //assert(lr > 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
    // TODO: free/copy opt_params, check lrs etc.
    assert(size > 0 && "Must have 1 or more param.");
    struct toptimizer* optim = (struct toptimizer*)malloc(sizeof (struct toptimizer));
    optim->net= net;
    optim->opt_params = opt_params;
    optim->step = step;
    return optim;
}

void toptimizer_free(struct toptimizer* topt) {
    free(topt);//dont free net
}

//void toptimizer_gather_params(): params require grads but are NOOPS.
//retain grad? for non leaf nodes?
