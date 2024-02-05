#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "teensygrad.h"

#define MAX_NODES 100

struct graph {
    struct tensor** nodes;
    size_t size;
};

struct graph* create_graph() {
    struct graph* list = (struct graph*)malloc(sizeof(struct graph));
    list->nodes = (struct tensor**)malloc(sizeof(struct tensor*)*MAX_NODES);
    list->size = 0;
    return list;
}

//O(n) :(, need hash_set like ggml
bool already_visited(struct graph* list, struct tensor* t) {
    for (size_t i = 0; i < list->size; i++) {
        if (list->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

//sorts graph in reversed topological order
void topo_sort(struct graph* list, struct tensor* current) {
    for (size_t i = 0; i < op_radix(current->op); i++) {
        struct tensor* parent = current->parents[i];
        if (!already_visited(list, parent)) {
            topo_sort(list, parent);
        }
    }
    list->nodes[list->size] = current;
    list->size += 1;
    assert(list->size < MAX_NODES && "Too many nodes in the graph.");
}

//calculate gradient of current,
void backwards(struct tensor* current) {
    struct graph* list = create_graph();
    topo_sort(list, current);

    struct tensor* grads = ones_tensor(current->shape_b, false, NULL, NOOP);
    destroy_tensor(current->grads);
    current->grads = grads;
    if (list->size <= 1) {
        return;
    }
    for (int32_t i = list->size-2; i >= 0; i--) {
        if (current->op) {
            current->pfn(current);
        }
        current = list->nodes[i];
    }
}

