#include "stdlib.h"
#include "stdint.h"
#include "teensygrad.h"

uint64_t buflen(struct shape* s) {
    uint64_t size = 1;
    for (uint32_t i = 0; i < s->size; i++) {
        size *= s->dims[i];
    }
    return size;
}

struct tensor create_tensor(struct shape* s) {
    uint64_t size = buflen(s);
    float* buffer = (float*)calloc(size, size*(uint64_t)sizeof(float));
    struct tensor t = {s, buffer, size};
    return t;
}

struct tensor from_buffer(struct shape* s, float* buffer) {
    struct tensor ret;
    ret.shape_b = s;
    uint64_t size = buflen(s);

    ret.buffer = buffer;
    ret.size = size;
    return ret;
}
