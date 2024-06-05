#include "../include/teensygrad.h"
#include "stdio.h"
#include "stdlib.h"
#include <time.h>

int main(void) {
    //srand(time(NULL));

    struct tshape* input_shape = tshape_build(1, 4);
    struct tshape* bias_shape = tshape_build(1, 1);

    struct tt* inputs = tt_uniformint(input_shape, 0, 10, false);
    tt_print(inputs);

    struct tt* weights = tt_uniform(input_shape, 0, 10, true);
    tt_print(weights);

    struct tt* bias = tt_uniformint(bias_shape, 0, 10, true);
    tt_print(bias);

    struct tt* wi = tt_mul(inputs, weights, true);
    tt_print(wi);

    struct tt* dot_product = tt_sum(wi, true);
    tt_print(dot_product);

    struct tt* hidden = tt_add(dot_product, bias, true);
    tt_print(hidden);

    struct tt* activation = tt_relu(hidden, true);
    tt_print(activation);
    
    struct tgraph* cg = tgraph_build(activation);
    tgraph_zeroed(cg);
    tgraph_backprop(cg);

    printf("input grads address: %d\n", inputs->grads);

    tt_print(inputs->grads);
    tt_print(weights->grads);
    tt_print(bias->grads);//
    tt_print(wi->grads);//
    tt_print(dot_product->grads);//
    tt_print(hidden->grads);
    tt_print(activation->grads);
    
}
