#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "teensygrad.h"

#define MAX_NODES 100

struct teensy_graph {
    struct teensy_tensor** nodes;
    size_t size;
};

struct teensy_graph* teensy_graph_create() {
    struct teensy_graph* list = (struct teensy_graph*)malloc(sizeof(struct teensy_graph));
    list->nodes = (struct teensy_tensor**)malloc(sizeof(struct teensy_tensor*)*MAX_NODES);
    list->size = 0;
    return list;
}

bool already_visited(struct teensy_graph* list, struct teensy_tensor* t) {
    for (size_t i = 0; i < list->size; i++) {
        if (list->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

//sorts graph in reversed topological order
void topo_sort(struct teensy_graph* list, struct teensy_tensor* current) {
    for (size_t i = 0; i < op_radix(current->op); i++) {
        struct teensy_tensor* parent = current->parents[i];
        if (!already_visited(list, parent) && parent->requires_grad) {
            topo_sort(list, parent);
        }
    }
    list->nodes[list->size] = current;
    list->size += 1;
    assert(list->size < MAX_NODES && "Too many nodes in the teensy_graph.");
}

//iteratively call _backwards on nodes, which calculates gradients on parent nodes.
void teensy_backwards(struct teensy_tensor* current) {
    if (!current->requires_grad) return;
    struct teensy_graph* list = teensy_graph_create();
    topo_sort(list, current);

    struct teensy_tensor* grads = teensy_tensor_ones(current->shape, false, NULL, NOOP);
    teensy_tensor_destroy(current->grads);
    current->grads = grads;
    for (int32_t i = list->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = list->nodes[i];
    }
}

