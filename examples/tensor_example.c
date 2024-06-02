#include "stdio.h"
#include "stdlib.h"
#include "../include/teensygrad.h"

int main(void) {
    int buf_size = 2;
    struct teensy_shape* s100 = teensy_shape_create_1d(100);
    struct teensy_tensor* uni = teensy_tensor_scaled_uniform(s100, -8, 17, false);
    teensy_tensor_print(uni);

    struct teensy_shape* s = teensy_shape_create_1d(buf_size);
    struct teensy_shape* s_bias = teensy_shape_create_1d(1);

    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
    }
    buffer3[0] = 4.0f;

    struct teensy_tensor* input = teensy_tensor_from_buffer(s, buffer2, false);
    struct teensy_tensor* weight = teensy_tensor_from_buffer(s, buffer, true);
    struct teensy_tensor* bias = teensy_tensor_from_buffer(s_bias, buffer3, true);

    struct teensy_tensor* wi = teensy_tensor_mul(weight, input, false);
    struct teensy_tensor* dot_wi = teensy_tensor_sum(wi, true);
    struct teensy_tensor* out = teensy_tensor_add(dot_wi, bias, true);

    struct teensy_tensor* act = teensy_tensor_relu(out, true);
    struct teensy_tensor* neg_act = teensy_tensor_neg(act, true);

    printf("input:");
    teensy_tensor_print(input);
    printf("bias:");
    teensy_tensor_print(bias);
    printf("weight:");
    teensy_tensor_print(weight);
    printf("input * weight:");
    teensy_tensor_print(wi);
    printf("sum(inputs * weights)");
    teensy_tensor_print(dot_wi);
    printf("sum(inputs * weights) + bias");
    teensy_tensor_print(out);
    printf("relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(act);
    printf("-relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(neg_act);

    printf("-------------------------------\n");
    printf("Grads before:\n");
    printf("input:");
    teensy_tensor_print(input->grads);
    printf("bias:");
    teensy_tensor_print(bias->grads);
    printf("weight:");
    teensy_tensor_print(weight->grads);
    printf("input * weight:");
    teensy_tensor_print(wi->grads);
    printf("sum(inputs * weights)");
    teensy_tensor_print(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    teensy_tensor_print(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(act->grads);
    printf("-relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(neg_act->grads);

    printf("-------------------------------\n");
    printf("Performing backprop:\n");
    teensy_backwards(neg_act);

    printf("-------------------------------\n");
    printf("Grads after:\n");
    printf("input:");
    teensy_tensor_print(input->grads);
    printf("bias:");
    teensy_tensor_print(bias->grads);
    printf("weight:");
    teensy_tensor_print(weight->grads);
    printf("input * weight:");
    teensy_tensor_print(wi->grads);
    printf("sum(inputs * weights)");
    teensy_tensor_print(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    teensy_tensor_print(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(act->grads);
    printf("-relu(sum(inputs * weights) + bias):");
    teensy_tensor_print(neg_act->grads);
}
