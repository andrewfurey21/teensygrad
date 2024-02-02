#include "stdio.h"
#include "stdlib.h"
#include "../teensygrad/teensygrad.h"

int main(void) {
    struct shape s = create_shape_2d(1, 10);
    printf("(");
    for (int i = 0; i < s.size; i++) {
        printf("%d,", s.dims[i]);
    }
    printf(")\n");
}
