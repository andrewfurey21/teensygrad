#include "stdio.h"
#include "stdlib.h"
#include "../include/teensygrad.h"

int main(void) {
    int buf_size = 2;
    struct tshape* s100 = tshape_create(1, 100);
    tshape_print(s100);
    struct tt* uni = tt_scaled_uniform(s100, -8, 17, false);
    tt_print(uni);

    struct tshape* s = tshape_create(1, buf_size);
    struct tshape* s_bias = tshape_create(1, 1);

    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
    }
    buffer3[0] = 4.0f;

    struct tt* input = tt_from_buffer(s, buffer2, false);
    struct tt* weight = tt_from_buffer(s, buffer, true);
    struct tt* bias = tt_from_buffer(s_bias, buffer3, true);

    struct tt* wi = tt_mul(weight, input, true);
    struct tt* dot_wi = tt_sum(wi, true);
    struct tt* out = tt_add(dot_wi, bias, true);

    struct tt* act = tt_relu(out, true);
    struct tt* neg_act = tt_neg(act, true);

    printf("input:");
    tt_print(input);
    printf("bias:");
    tt_print(bias);
    printf("weight:");
    tt_print(weight);
    printf("input * weight:");
    tt_print(wi);
    printf("sum(inputs * weights)");
    tt_print(dot_wi);
    printf("sum(inputs * weights) + bias");
    tt_print(out);
    printf("relu(sum(inputs * weights) + bias):");
    tt_print(act);
    printf("-relu(sum(inputs * weights) + bias):");
    tt_print(neg_act);

    printf("-------------------------------\n");
    printf("Grads before:\n");
    printf("input:");
    tt_print(input->grads);
    printf("bias:");
    tt_print(bias->grads);
    printf("weight:");
    tt_print(weight->grads);
    printf("input * weight:");
    tt_print(wi->grads);
    printf("sum(inputs * weights)");
    tt_print(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    tt_print(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    tt_print(act->grads);
    printf("-relu(sum(inputs * weights) + bias):");
    tt_print(neg_act->grads);

    printf("-------------------------------\n");
    printf("Building computational graph and performing backprop:\n");

    struct tgraph* cg = tgraph_build(neg_act);
    tgraph_zeroed(cg);
    tbackwards(cg);

    printf("-------------------------------\n");
    printf("Grads after:\n");
    printf("input:");
    tt_print(input->grads);
    printf("bias:");
    tt_print(bias->grads);
    printf("weight:");
    tt_print(weight->grads);
    printf("input * weight:");
    tt_print(wi->grads);
    printf("sum(inputs * weights)");
    tt_print(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    tt_print(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    tt_print(act->grads);
    printf("-relu(sum(inputs * weights) + bias):");
    tt_print(neg_act->grads);
}
