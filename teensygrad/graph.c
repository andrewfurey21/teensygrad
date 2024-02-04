#include "stdbool.h"
#include "assert.h"
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

//O(n) :(
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
//TODO: accumulate gradients and add zeroing out function instead
void backwards(struct tensor* current) {
    //calculate gradients for each item in list
    //add those gradients to the current gradients.
    struct graph* list = create_graph();
    topo_sort(list, current);
    for (size_t i = 0; i < list->size; i++) {
        print_t(list->nodes[i]);
    }
//    struct tensor* grad = ones_tensor(current->shape_b, false, NULL, NOOP);
}




void add_backwards(struct tensor* current, struct tensor* current_grads) {
}
