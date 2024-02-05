#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "teensygrad.h"

#define MAX_DIMS 4

struct shape* create_shape(uint32_t* dims, uint32_t size) {
    uint32_t* dims_copy = (uint32_t*)malloc(size * sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        dims_copy[i] = dims[i];
    }
    struct shape* ret = (struct shape*)malloc(sizeof(struct shape));
    ret->dims = dims_copy;
    ret->size = size;
    return ret;
}

struct shape* create_shape_1d(uint32_t dim) {
    uint32_t* dims = (uint32_t*)malloc(sizeof(uint32_t));
    dims[0] = dim;
    return create_shape(dims, 1);
}

struct shape* create_shape_2d(uint32_t dim1, uint32_t dim2) {
    uint32_t* dims = (uint32_t*)malloc(2*sizeof(uint32_t));
    dims[0] = dim1;
    dims[1] = dim2;
    return create_shape(dims, 2);
}

void free_shape(struct shape* s) {
    free(s->dims);
}

bool same_shape(struct tensor* a, struct tensor* b) {
    if (a->shape_b->size != b->shape_b->size) {
        return false;
    }
    for (uint32_t i = 0; i < a->shape_b->size; i++) {
        if (a->shape_b->dims[i] != b->shape_b->dims[i]) {
            return false;
        }
    }
    return true;
}

void print_shape(struct shape* s) {
    assert(s->size <= MAX_DIMS && "Too many dimensions in shape.");
    printf("shape:(");
    for (size_t i = 0; s->dims[i] != 0 && i < MAX_DIMS; i++) {
        printf("%d,", s->dims[i]);
    }
    printf(")\n");
}
