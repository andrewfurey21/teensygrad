#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    struct shape s = create_shape_2d(2, 5);

    int buf_size = 10;
    float* buffer = (float*)malloc(buf_size*sizeof(float));

    for (int i = 0; i < buf_size; i++) {
        buffer[i] = (float)i;
    }

    struct tensor t = from_buffer(&s, buffer);

    for (int i = 0; i < t.size; i++) {
        printf("%f, ", t.buffer[i]);
    }
}
