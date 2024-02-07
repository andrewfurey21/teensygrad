#include "assert.h"
#include "stdlib.h"
#include "../teensygrad.h"

void teensy_sgd(struct teensy_optimizer* optim) {
}

struct teensy_optimizer* teensy_optimizer_create(struct teensy_tensor** params, float lr, void (*step)(struct teensy_optimizer*)) {
    assert(lr >= 0 && lr <= 1 && "Learning rate should be between 0 and 1.");
    struct teensy_optimizer* optim = (struct teensy_optimizer*)malloc(sizeof (struct teensy_optimizer));
    optim->params = params;
    optim->learning_rate = lr;
    optim->step = step;
    return optim;
}
