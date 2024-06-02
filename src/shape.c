#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "../include/teensygrad.h"

#define MAX_DIMS 4

struct teensy_shape* teensy_shape_create(uint32_t* dims, uint32_t size) {
    uint32_t* dims_copy = (uint32_t*)malloc(size * sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        dims_copy[i] = dims[i];
    }
    struct teensy_shape* ret = (struct teensy_shape*)malloc(sizeof(struct teensy_shape));
    ret->dims = dims_copy;
    ret->size = size;
    return ret;
}

struct teensy_shape* teensy_shape_create_1d(uint32_t dim) {
    uint32_t* dims = (uint32_t*)malloc(sizeof(uint32_t));
    dims[0] = dim;
    return teensy_shape_create(dims, 1);
}

struct teensy_shape* teensy_shape_create_2d(uint32_t dim1, uint32_t dim2) {
    uint32_t* dims = (uint32_t*)malloc(2*sizeof(uint32_t));
    dims[0] = dim1;
    dims[1] = dim2;
    return teensy_shape_create(dims, 2);
}

void teensy_shape_destroy(struct teensy_shape* s) {
    free(s->dims);
    free(s);
}


void teensy_shape_print(struct teensy_shape* s) {
    assert(s->size <= MAX_DIMS && "Too many dimensions in teensy_shape.");
    printf("teensy_shape:(");
    for (size_t i = 0; s->dims[i] != 0 && i < MAX_DIMS; i++) {
        printf("%d,", s->dims[i]);
    }
    printf(")\n");
}
