#include "../include/tensor.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

// TODO: encapsulating ops with functions is broken, i dont think gradients flow correcly.
// TODO: check summing with (2, 1, 1) or something with ones
// TODO: get name of linspace/arange correct

// TODO: get this working correctly, compare with proper tinygrad/pytorch impl

int main(void) {
    srand(time(NULL));

    // Example: b @ a
    ttuple* a_shape= ttuple_build(2, 4, 3);
    tt* a = tt_linspace(a_shape, 0, 3*4, false);

    ttuple* b_shape= ttuple_build(2, 5, 4);
    tt* b = tt_linspace(b_shape, 0, 5*4, false);

    tt_print(a);
    tt_print(b);


    ttuple* reshape_a_shape = ttuple_build(3, 1, 4, 3);
    tt* reshape_a = tt_reshape(a, reshape_a_shape);

    ttuple* reshape_b_shape = ttuple_build(3, 5, 4, 1);
    tt* reshape_b = tt_reshape(b, reshape_b_shape);

    tt* expand_a = tt_expand(reshape_a, 0, 5);
    tt* expand_b = tt_expand(reshape_b, 2, 3);

    tt* mul = tt_mul(expand_a, expand_b);

    tt* dot = tt_sum(mul, 1);

    ttuple* reshaped_dot_shape = ttuple_build(2, 5, 3);
    tt* reshaped_dot = tt_reshape(dot, reshaped_dot_shape);

    tt_print(reshaped_dot);
}
