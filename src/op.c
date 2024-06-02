#include "assert.h"
#include "../include/teensygrad.h"

size_t op_radix(enum teensy_op op) {
    switch (op) {
        case NOOP:
            return 0;
        case RELU:
        case NEG:
        case SUM_REDUCE:
            return 1;
        case ADD:
        case MUL:
            return 2;
        default:
            assert(false && "This op is not implemented.");
    }
}
