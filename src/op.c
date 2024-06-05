#include "assert.h"
#include "stdio.h"
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

void print_op_string(enum top op) {
    switch (op) {
        case NOOP:
            printf("NO OP\n");
            return;
        case RELU:
            printf("RELU\n");
            return;
        case NEG:
            printf("NEGATE\n");
            return;
        case SUM_REDUCE:
            printf("SUM REDUCE\n");
            return;
        case RESHAPE:
            printf("RESHAPE\n");
            return;
        case ADD:
            printf("ADD\n");
            return;
        case MUL:
            printf("MUL\n");
            return;
        default:
            assert(false && "This op is not implemented.");
    }
}
