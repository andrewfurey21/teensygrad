#include "stdint.h"
#include "stdbool.h"

#ifndef _TEENSYGRAD_H
#define _TEENSYGRAD_H

struct Shape {
    uint32_t* dims;
    uint32_t size;
};

enum Ops {
    ADD,
    MUL
};

enum Device {
    CPU
};

struct Tensor {
    struct Tensor* grads;
    struct Shape* shape;
    bool calculate_grads;
    struct Tensor* parents;
    enum Ops op;
    float* buffer;
};

#endif
