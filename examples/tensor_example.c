#include "../include/tensor.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

int main(void) {
    srand(time(NULL));

    //check summing with (2, 1, 1) or something with ones
    ttuple* input_shape = ttuple_build(2, 4, 6);//make sure to free
    tt* a = tt_linspace(input_shape, -4*6, 4*6, true);
    tt* b = tt_uniform(input_shape, -10, 10, true);

    tt* mul = tt_mul(a, b);
    tt* sum = tt_sum(mul, 1);

    tt* relu = tt_relu(sum);

    ttuple* new_shape = ttuple_build(3, 2, 1, 2);
    tt* reshaped = tt_reshape(relu, new_shape);

    tt* sum2= tt_sum(reshaped, 2);

    tt* total = tt_sum(sum2, -1);

    tt_print(a);
    tt_print(b);
    tt_print(mul);
    tt_print(sum);
    tt_print(relu);
    tt_print(reshaped);
    tt_print(sum2);
    tt_print(total);

    tgraph* cg = tgraph_build(total);
    tgraph_zeroed(cg);
    tgraph_backprop(cg);

    tt_print(a->grads);
    tt_print(b->grads);
    tt_print(mul->grads);
    tt_print(sum->grads);
    tt_print(relu->grads);
    tt_print(reshaped->grads);
    tt_print(sum2->grads);
    tt_print(total->grads);
}
