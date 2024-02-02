#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    struct shape s = create_shape_2d(1, 10);

    int buf_size = 10;
    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));
    float* buffer3 = (float*)malloc(buf_size*sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i+2;
        buffer3[i] = (float)i-4;
    }

    struct tensor weight = from_buffer(&s, buffer);
    struct tensor bias = from_buffer(&s, buffer3);
    struct tensor input = from_buffer(&s, buffer2);

    struct tensor wi = mul_tensors(&weight, &input);
    struct tensor out = add_tensors(&wi, &bias);

    struct tensor act = relu_tensor(&out);

    print_t(&input);
    print_t(&weight);

    print_t(&wi);
    print_t(&bias);

    print_t(&out);
    print_t(&act);
}
