#include "stdbool.h"
#include "assert.h"
#include "stdio.h"
#include "../include/teensygrad.h"
#define MAX_NODES 100


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
        if (!already_visited(net, parent) && parent->requires_grad) {//all tensors in graph require grads
            topo_sort(net, parent);
        }
    }
    net->nodes[net->size] = current;
    net->size += 1;
    assert(net->size < MAX_NODES && "Too many nodes in the tgraph.");
}

struct tgraph* tgraph_build(struct tt* x) {
    assert(x->requires_grad && "Will not build graph on something that doesn't require gradients");

    struct tgraph* net = (struct tgraph*)malloc(sizeof(struct tgraph));
    net->nodes = (struct tt**)malloc(sizeof(struct tt*)*MAX_NODES);
    net->size = 0;
    net->training = true;

    topo_sort(net, x);

    return net;
}

void tgraph_free(struct tgraph* net) {
    for (size_t i = 0; i < net->size; i++) {
        tt_free(net->nodes[i]);
    }
    free(net);
}

// zero out gradients in the graph
void tgraph_zeroed(struct tgraph* net) {
    if (!net->training) return;
    for (uint32_t i = 0; i < net->size; i++) {
        struct tt* t = net->nodes[i];
        tt_to_zeros(t->grads);//all tensors in graph require grads
    }
}

//iteratively call backwards on nodes, which calculates gradients on parent nodes.
void tgraph_backprop(struct tgraph* net) {
    assert(net->training && "Training is set to false!");

    // TODO: current->shape must be (1)
    struct tshape* unit_shape = tshape_build(1, 1);
    struct tt* current = net->nodes[net->size-1];
    assert(tshape_equal(current->shape, unit_shape) && "Last tensor must be scalar");
    assert(current->requires_grad && "Can't do backprop on tensor without grads");
    free(unit_shape);

    struct tt* grads = tt_ones(current->shape, false);
    tt_free(current->grads);
    current->grads = grads;

    for (int32_t i = net->size-2; i >= 0; i--) {
        if (current->op) {
            current->_backwards(current);
        }
        current = net->nodes[i];
    }
}

// TODO: graph to string (or visualize graph, output to image)

