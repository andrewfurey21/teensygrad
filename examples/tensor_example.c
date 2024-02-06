#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    int buf_size = 2;
    struct shape* s = create_shape_1d(buf_size);
    struct shape* s_bias = create_shape_1d(1);

    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
    }
    buffer3[0] = 4.0f;

    struct tensor* input = from_buffer(s, buffer2, true);
    struct tensor* weight = from_buffer(s, buffer, true);
    struct tensor* bias = from_buffer(s_bias, buffer3, true);

    struct tensor* wi = mul_tensors(weight, input, true);
    struct tensor* dot_wi = sum_reduce_tensors(wi, true);
    struct tensor* out = add_tensors(dot_wi, bias, true);

    struct tensor* act = relu_tensor(out, true);

    printf("input:");
    print_t(input);
    printf("bias:");
    print_t(bias);
    printf("weight:");
    print_t(weight);
    printf("input * weight:");
    print_t(wi);
    printf("sum(inputs * weights)");
    print_t(dot_wi);
    printf("sum(inputs * weights) + bias");
    print_t(out);
    printf("relu(sum(inputs * weights) + bias):");
    print_t(act);

    printf("-------------------------------\n");
    printf("Grads before:\n");
    printf("input:");
    print_t(input->grads);
    printf("bias:");
    print_t(bias->grads);
    printf("weight:");
    print_t(weight->grads);
    printf("input * weight:");
    print_t(wi->grads);
    printf("sum(inputs * weights)");
    print_t(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    print_t(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    print_t(act->grads);

    printf("-------------------------------\n");
    printf("Performing backprop:\n");
    backwards(act);

    printf("-------------------------------\n");
    printf("Grads after:\n");
    printf("input:");
    print_t(input->grads);
    printf("bias:");
    print_t(bias->grads);
    printf("weight:");
    print_t(weight->grads);
    printf("input * weight:");
    print_t(wi->grads);
    printf("sum(inputs * weights)");
    print_t(dot_wi->grads);
    printf("sum(inputs * weights) + bias");
    print_t(out->grads);
    printf("relu(sum(inputs * weights) + bias):");
    print_t(act->grads);
}
