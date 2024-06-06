#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdarg.h"
#include "../include/teensygrad.h"

#define MAX_DIMS 4

struct tshape* tshape_build(uint32_t size, ...) {
    assert(MAX_DIMS >= size);
    assert(size > 0 && "Size must be positive");
    va_list ap;

    struct tshape* ret = (struct tshape*)malloc(sizeof(struct tshape));
    ret->size = size;
    ret->dims = (int32_t*)malloc(size * sizeof(int32_t));
    va_start(ap, size);

    for (uint32_t i = 0; i < size; i++) {
        assert(size > 0 && "Dimensions must be positive");
        ret->dims[i] = va_arg(ap, uint32_t);
    }
    va_end(ap);

    return ret;
}

struct tshape* tshape_copy(struct tshape* other) {
    struct tshape* copy = (struct tshape*)malloc(sizeof(struct tshape));
    copy->size = other->size;
    copy->dims = (int32_t*)malloc(sizeof(int32_t) * copy->size);
    for (uint32_t i = 0; i < copy->size; i++) {
        copy->dims[i] = other->dims[i];
    }
    return copy;
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

void tshape_free(struct tshape* s) {
    free(s->dims);
    free(s);
}

// TODO: return char*
void tshape_print(struct tshape* s) {
    assert(s->size <= MAX_DIMS && "Too many dimensions in tshape.");
    printf("shape (");
    for (size_t i = 0; i < s->size; i++) {
        assert(s->dims[i] > 0 && "Shape must be positive numbers.");
        printf("%d,", s->dims[i]);
    }
    printf(")\n");
}
