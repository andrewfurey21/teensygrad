#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "../include/teensygrad.h"

#define MAX_DIMS 4

struct tshape* tshape_create(uint32_t* dims, uint32_t size) {
    uint32_t* dims_copy = (uint32_t*)malloc(size * sizeof(uint32_t));
    for (uint32_t i = 0; i < size; i++) {
        dims_copy[i] = dims[i];
    }
    struct tshape* ret = (struct tshape*)malloc(sizeof(struct tshape));
    ret->dims = dims_copy;
    ret->size = size;
    return ret;
}

bool tshape_compare(struct tt* a, struct tt* b) {
    if (a->shape->size != b->shape->size) {
        return false;
    }
    for (uint32_t i = 0; i < a->shape->size; i++) {
        if (a->shape->dims[i] != b->shape->dims[i]) {
            return false;
        }
    }
    return true;
}

struct tshape* tshape_create_1d(uint32_t dim) {
    uint32_t* dims = (uint32_t*)malloc(sizeof(uint32_t));
    dims[0] = dim;
    return tshape_create(dims, 1);
}

struct tshape* tshape_create_2d(uint32_t dim1, uint32_t dim2) {
    uint32_t* dims = (uint32_t*)malloc(2*sizeof(uint32_t));
    dims[0] = dim1;
    dims[1] = dim2;
    return tshape_create(dims, 2);
}

void tshape_destroy(struct tshape* s) {
    free(s->dims);
    free(s);
}

void tshape_print(struct tshape* s) {
    assert(s->size <= MAX_DIMS && "Too many dimensions in tshape.");
    printf("tshape:(");
    for (size_t i = 0; s->dims[i] != 0 && i < MAX_DIMS; i++) {
        printf("%d,", s->dims[i]);
    }
    printf(")\n");
}
