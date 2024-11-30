#include "../include/teensygrad.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

int main(void) {
    srand(time(NULL));

    //check summing with (2, 1, 1) or something with ones
    ttuple* input_shape = ttuple_build(2, 4, 3);//make sure to free
    tt* a = tt_linspace(input_shape, 0, 4*3, true);
    // tt* b = tt_uniform(input_shape, 0, 2*2*3, true);

    tt* b = tt_sum(a, 1);
    
    tt_print(a);
    tt_print(b);

    float buffer[4] = {0, 1, 2, 3};
    b->grads = tt_from_buffer(b->grads->view->shape, buffer, false);

    b->_backwards(b);

    tt_print(a->grads);
    tt_print(b->grads);
    //
    // tt* inputs = tt_uniformint(input_shape, 0, 10, false);
    // tt_print(inputs);
    //
    // tt* weights = tt_uniform(input_shape, 0, 10, true);
    // tt_print(weights);
    //
    // tt* wi = tt_mul(inputs, weights);
    // tt_print(wi);
    //
    // ttuple* new_shape = ttuple_build(2, 2, 2);
    // tt* reshape_wi = tt_reshape(wi, new_shape);
    // tt_print(reshape_wi);

    // tt* activation = tt_relu(reshape_wi);
    // tt_print(activation);

    // tt* sum_activation = tt_sum(reshape_wi);
    // tt_print(sum_activation);

    // tgraph* cg = tgraph_build(reshape_wi);
    // tgraph_zeroed(cg);
    // tgraph_backprop(cg);
    //
    // tt_print(inputs->grads);
    // tt_print(weights->grads);
    // tt_print(wi->grads);
    //tt_print(activation->grads);
    // tt_print(sum_activation->grads);
}
