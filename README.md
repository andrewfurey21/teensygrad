# libtensor

<!--  TODO: Show image of generated graph  -->

I built a small tensor library with automatic differentiation for deep learning.

This library has no dependencies, it's written from scratch in C.

<!-- TODO: compile mnist example with emscripten, host at mnist.andrew.industries -->

## current features

- [x] tensor
- [x] backwards mode autodifferentiation
- [ ] ops
    - [x] add (with broadcasting)
    - [x] mul (with broadcasting)
    - [x] sum along an axis
    - [x] relu
    - [ ] reshape
    - [ ] matmul (indirectly, using reshape, mul and sum)
    - [ ] flatten
    - [ ] max pool
    - [ ] convolutions
    - [ ] batch norm (indirectly, using add and mul)
    - [ ] sparse categorical cross entropy
- [ ] adam optimizer
- [ ] import/export weights

## notes

Here's a mini example of working with tensors to compute gradients.

```c
    ttuple* input_shape = ttuple_build(2, 4, 6);
    tt* a = tt_uniform(input_shape, -10, 10, true);
    tt* b = tt_uniform(input_shape, -10, 10, true);
    tt* a_b_mul = tt_mul(a, b);
    tt* sum = tt_sum(a_b_mul, -1);
    tgraph* comp_graph = tgraph_build(sum);
    tgraph_zeroed(comp_graph);
    tgraph_backprop(cg);
```

Right now there is no direct matmul op. You have to do it manually with a reshape, mul and sum.

```c
// TODO: give examples
```

The same is true for batch norm.

```c
// TODO: give examples
```

Convolutions are done manually however.

```c
// TODO: give examples
```

## examples

- [ ] mnist handwritten digit recognition

<!--  TODO: Add a image as example  -->

## future ideas

- different backends (opencl/vulkan, cuda, metal, avx/sse)
