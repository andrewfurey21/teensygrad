#include "assert.h"
#include "../include/teensygrad.h"

size_t top_radix(enum top op) {
    switch (op) {
        case NOOP:
            return 0;
        case RELU:
        case NEG:
        case SUM_REDUCE:
        case RESHAPE:
            return 1;
        case ADD:
        case MUL:
            return 2;
        default:
            assert(false && "This op is not implemented.");
    }
}
