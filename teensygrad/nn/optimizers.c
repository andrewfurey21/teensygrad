#include "assert.h"
#include "stdlib.h"
#include "../teensygrad.h"

void teensy_sgd(struct teensy_optimizer* optim) {
    for (uint64_t i = 0; i < optim->size; i++) {
        struct teensy_tensor* t = optim->params[i];

        struct teensy_tensor* lrs = teensy_tensor_full_like(t->shape, -optim->learning_rate, false);
        struct teensy_tensor* updated_grads = teensy_tensor_mul(lrs, t->grads, false);

        struct teensy_tensor* updated_params = teensy_tensor_add(updated_grads, t, false);

        teensy_tensor_copy_buffer(t, updated_params);

        teensy_tensor_destroy(lrs);
        teensy_tensor_destroy(updated_grads);
        teensy_tensor_destroy(updated_params);
    }
}

struct teensy_optimizer* teensy_optimizer_create(struct teensy_tensor** params, uint64_t size, float lr, void (*step)(struct teensy_optimizer*)) {
    assert(lr >= 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
    assert(size > 0 && "Must have 1 or more param.");
    struct teensy_optimizer* optim = (struct teensy_optimizer*)malloc(sizeof (struct teensy_optimizer));
    optim->params = params;
    optim->learning_rate = lr;
    optim->step = step;
    optim->size = size;
    return optim;
}
