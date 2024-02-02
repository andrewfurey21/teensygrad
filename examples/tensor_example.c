#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    struct shape s = create_shape_2d(2, 5);

    int buf_size = 10;
    float* buffer = (float*)malloc(buf_size*sizeof(float));
    float* buffer2 = (float*)malloc(buf_size*sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
        buffer2[i] = (float)i-10;
    }

    struct tensor t1 = from_buffer(&s, buffer);
    struct tensor t2 = from_buffer(&s, buffer2);
    struct tensor added = add_tensors(&t1, &t2);

    print_t(&t1);
    print_t(&t2);
    print_t(&added);
}
