#include "stdint.h"
#include "assert.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdarg.h"
#include "../include/teensygrad.h"
#include <stdint.h>

#define MAX_items 4

ttuple* ttuple_build(uint32_t size, ...) {
    assert(MAX_items >= size);
    assert(size > 0 && "Size must be positive");
    va_list ap;

    ttuple* ret = (ttuple*)malloc(sizeof(ttuple));
    ret->size = size;
    ret->items = (int32_t*)malloc(size * sizeof(int32_t));
    va_start(ap, size);

    for (uint32_t i = 0; i < size; i++) {
        ret->items[i] = va_arg(ap, uint32_t);
        //should there be a max items[i]?
        assert(ret->items[i] > 0 && "Dimensions must be positive");
    }
    va_end(ap);
    return ret;
}

uint64_t ttuple_mul(ttuple* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->items[i];
    }
    return size;
}

ttuple* ttuple_copy(ttuple* other) {
    ttuple* copy = (ttuple*)malloc(sizeof(ttuple));
    copy->size = other->size;
    copy->items = (int32_t*)malloc(sizeof(int32_t) * copy->size);
    for (uint32_t i = 0; i < MAX_items || i < copy->size; i++) {
        copy->items[i] = other->items[i];
    }
    return copy;
}


bool ttuple_equal(ttuple* a, ttuple* b) {
    if (a->size != b->size) return false;
    for (int i = 0; i < a->size || i < MAX_items; i++) {
        if (a->items[i] != b->items[i]) return false;
    }
    return true;
}


void ttuple_free(ttuple* s) {
    free(s->items);
    free(s);
}

void ttuple_print(ttuple* s) {
    assert(s->size <= MAX_items && "Too many dimensions in tshape.");
    printf("(");
    for (size_t i = 0; i < s->size; i++) {
        printf("%d", s->items[i]);
        if (i < s->size - 1) printf(",");
    }
    printf(")\n");
}

// ttuple* ttuple_permute(ttuple* shape, ttuple* axes){
//     ttuple* permuted = ttuple_copy(shape);
//     for (int i = 0; i < shape->size; i++) {
//         int axis = axes->items[i];
//         assert(axis >= 0 && axis <= MAX_items);
//         permuted->items[i] = shape->items[axis];
//     }
//     assert(ttuple_mul(permuted) == ttuple_mul(shape) && "Possibly repeated axis");
//     return permuted;
// }

// bool tshape_duplicates(struct tshape* axes) {
//     assert(axes->size <= MAX_items);
//     for (int i = 0; i < axes->size-1; i++) {
//         for (int j = i+1; j < axes->size; j++) {
//             if (axes->items[i] == axes->items[j]) return true;
//         }
//     }
//     return false;
// } why?
