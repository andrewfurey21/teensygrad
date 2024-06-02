#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "../include/teensygrad.h"

#define MAX_NODES 100

struct tgraph {
    struct tt** nodes;
    size_t size;
};

struct tgraph* tgraph_create() {
    struct tgraph* list = (struct tgraph*)malloc(sizeof(struct tgraph));
    list->nodes = (struct tt**)malloc(sizeof(struct tt*)*MAX_NODES);
    list->size = 0;
    return list;
}

bool already_visited(struct tgraph* list, struct tt* t) {
    for (size_t i = 0; i < list->size; i++) {
        if (list->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

//sorts graph in reversed topological order
void topo_sort(struct tgraph* list, struct tt* current) {
    for (size_t i = 0; i < top_radix(current->op); i++) {
        struct tt* parent = current->parents[i];
        if (!already_visited(list, parent) && parent->requires_grad) {
            topo_sort(list, parent);
        }
    }
    list->nodes[list->size] = current;
    list->size += 1;
    assert(list->size < MAX_NODES && "Too many nodes in the tgraph.");
}

//iteratively call _backwards on nodes, which calculates gradients on parent nodes.
void tbackwards(struct tt* current) {
    if (!current->requires_grad) return;
    struct tgraph* list = tgraph_create();
    topo_sort(list, current);

    struct tt* grads = tt_ones(current->shape, false);
    tt_destroy(current->grads);
    current->grads = grads;
    for (int32_t i = list->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = list->nodes[i];
    }
}

