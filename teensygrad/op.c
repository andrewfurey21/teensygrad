#include "assert.h"
#include "teensygrad.h"

size_t op_radix(enum Op op) {
    switch (op) {
        case NOOP:
            return 0;
        case RELU:
            return 1;
        case ADD:
        case MUL:
            return 2;
        default:
            assert(false && "This op is not implemented.");
    }
}
