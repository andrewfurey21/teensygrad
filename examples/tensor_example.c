#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    struct shape* s = create_shape_1d(10);

    int buf_size = 10;
    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(buf_size*sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
        buffer3[i] = (float)i-4;
    }

    //struct tensor* weight = from_buffer(&s, buffer, true);
    struct tensor* bias = from_buffer(s, buffer3, true);
    struct tensor* input = from_buffer(s, buffer2, true);

    struct tensor* ib = add_tensors(input, bias, true);

    //struct tensor* wi = mul_tensors(weight, input);
    //struct tensor* out = add_tensors(wi, bias);

    //struct tensor* act = relu_tensor(out);

    printf("input:\n");
    print_t(input);
    printf("bias:\n");
    print_t(bias);
    printf("input + bias:\n");
    print_t(ib);

    printf("-------------------------------\n");
    printf("Grads before:\n");
    print_t(input->grads);
    print_t(bias->grads);
    print_t(ib->grads);

    backwards(ib);

    printf("-------------------------------\n");
    printf("Grads after:\n");
    print_t(input->grads);
    print_t(bias->grads);
    print_t(ib->grads);
    //backwards(act);
}
