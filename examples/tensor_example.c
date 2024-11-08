#include "../include/teensygrad.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

int main(void) {
    //srand(time(NULL));

    ttuple* t = ttuple_build(3, 1, 2, 3);

    uint64_t size = ttuple_mul(t);
    printf("tuple mul: %d", (int)size);

    ttuple* t_copy = ttuple_copy(t);
    ttuple_print(t);
    ttuple_copy(t_copy);

    ttuple* other = ttuple_build(2, 1, 2);
    bool other_equal = ttuple_equal(t, other);
    bool equal_copy = ttuple_equal(t, t_copy);
    printf("t == other: %d", other_equal);
    printf("t == copy: %d", equal_copy);

    ttuple_print(t);
    ttuple_print(t_copy);
    ttuple_print(other);

    ttuple_free(t);
    ttuple_free(t_copy);
    ttuple_free(other);

    // struct tshape* input_shape = tshape_build(1, 4);
    //
    // struct tt* inputs = tt_uniformint(input_shape, 0, 10, false);
    // tt_print(inputs);
    //
    // struct tt* weights = tt_uniform(input_shape, 0, 10, true);
    // tt_print(weights);
    //
    // struct tt* wi = tt_mul(inputs, weights);
    // tt_print(wi);
    //
    // struct tshape* new_shape = tshape_build(2, 2, 2);
    // struct tt* reshape_wi = tt_reshape(wi, new_shape);
    // tt_print(reshape_wi);
    //
    // struct tt* activation = tt_relu(reshape_wi);
    // tt_print(activation);
    //
    // struct tt* sum_activation = tt_sum(activation);
    // tt_print(sum_activation);
    //
    // struct tgraph* cg = tgraph_build(sum_activation);
    // tgraph_zeroed(cg);
    // tgraph_backprop(cg);
    //
    // tt_print(inputs->grads);
    // tt_print(weights->grads);
    // tt_print(wi->grads);
    // tt_print(activation->grads);
    // tt_print(sum_activation->grads);
    
}
