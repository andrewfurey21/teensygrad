#include "../include/teensygrad.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

int main(void) {
    srand(time(NULL));
    ttuple* input_shape = ttuple_build(1, 4);

    tt* inputs = tt_uniformint(input_shape, 0, 10, false);
    tt_print(inputs);

    tt* weights = tt_uniform(input_shape, 0, 10, true);
    tt_print(weights);

    tt* wi = tt_mul(inputs, weights);
    tt_print(wi);

    ttuple* new_shape = ttuple_build(2, 2, 2);
    tt* reshape_wi = tt_reshape(wi, new_shape);
    tt_print(reshape_wi);

    // tt* activation = tt_relu(reshape_wi);
    // tt_print(activation);

    // tt* sum_activation = tt_sum(reshape_wi);
    // tt_print(sum_activation);

    struct tgraph* cg = tgraph_build(reshape_wi);
    tgraph_zeroed(cg);
    tgraph_backprop(cg);

    tt_print(inputs->grads);
    tt_print(weights->grads);
    tt_print(wi->grads);
    //tt_print(activation->grads);
    // tt_print(sum_activation->grads);
}
