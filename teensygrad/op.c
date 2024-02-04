#include "teensygrad.h"

//this could be done better, do switch case
//

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
            return 0;
    }
}
