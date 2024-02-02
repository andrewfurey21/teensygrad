#include "stdint.h"
#include "stdbool.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

struct shape {
    uint32_t* dims;
    uint32_t size;
    char* str;
};

struct shape create_shape(uint32_t* dims, uint32_t size);
struct shape create_shape_1d(uint32_t dim);
struct shape create_shape_2d(uint32_t dim1, uint32_t dim2);
void free_shape(struct shape* s);
//TODO:print shape, shape_to_str

#endif
