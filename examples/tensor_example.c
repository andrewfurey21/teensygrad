#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    int buf_size = 2;
    struct shape* s = create_shape_1d(buf_size);

    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(buf_size*sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
        buffer3[i] = (float)i+4;
    }

    struct tensor* weight = from_buffer(s, buffer, true);
    struct tensor* bias = from_buffer(s, buffer3, true);
    struct tensor* input = from_buffer(s, buffer2, true);

    struct tensor* wi = mul_tensors(weight, input, true);
    struct tensor* out = add_tensors(wi, bias, true);

    struct tensor* act = relu_tensor(out, true);

    printf("input:\n");
    print_t(input);
    printf("bias:\n");
    print_t(bias);
    printf("weight:\n");
    print_t(weight);
    printf("input * weight:\n");
    print_t(wi);
    printf("wi + bias:\n");
    print_t(out);
    printf("relu(wi + bias):\n");
    print_t(act);

    printf("-------------------------------\n");
    printf("Grads before:\n");
    print_t(input->grads);
    print_t(bias->grads);
    print_t(weight->grads);
    print_t(wi->grads);
    print_t(out->grads);
    print_t(act->grads);

    backwards(act);

    printf("-------------------------------\n");
    printf("Grads after:\n");
    printf("input:\n");
    print_t(input->grads);
    printf("bias:\n");
    print_t(bias->grads);
    printf("weight:\n");
    print_t(weight->grads);
    printf("input * weight:\n");
    print_t(wi->grads);
    printf("wi + bias:\n");
    print_t(out->grads);
    printf("relu(wi + bias):\n");
    print_t(act->grads);
}
