#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "../include/teensygrad.h"

#define MAX_NODES 100


struct tgraph* tgraph_create(bool training) {
    struct tgraph* net = (struct tgraph*)malloc(sizeof(struct tgraph));
    net->nodes = (struct tt**)malloc(sizeof(struct tt*)*MAX_NODES);
    net->size = 0;
    net->training = training;
    return net;
}

bool already_visited(struct tgraph* net, struct tt* t) {
    for (size_t i = 0; i < net->size; i++) {
        if (net->nodes[i] == t) {
            return true;
        }
    }
    return false;
}

//sorts graph in reversed topological order
void topo_sort(struct tgraph* net, struct tt* current) {
    for (size_t i = 0; i < top_radix(current->op); i++) {
        struct tt* parent = current->parents[i];
        if (!already_visited(net, parent) && parent->requires_grad) {
            topo_sort(net, parent);
        }
    }
    net->nodes[net->size] = current;
    net->size += 1;
    assert(net->size < MAX_NODES && "Too many nodes in the tgraph.");
}

struct tgraph* tgraph_build(struct tt* x) {
    assert(x->requires_grad && "Will not build graph on something that doesn't require gradients");

    struct tgraph* network = tgraph_create(true);
    topo_sort(network, x);

    return network;
}

//iteratively call backwards on nodes, which calculates gradients on parent nodes.
void tbackwards(struct tgraph* net) {
    assert(net->training && "Training is set to false!");

    // TODO: current->shape must be (1)
    struct tt* current = net->nodes[net->size-1];
    if (!current->requires_grad) return;

    struct tt* grads = tt_ones(current->shape, false);
    tt_destroy(current->grads);
    current->grads = grads;

    for (int32_t i = net->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = net->nodes[i];
    }
}

