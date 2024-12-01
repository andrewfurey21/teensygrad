# libtensor

<!--  TODO: Show image of generated graph  -->

I built a small tensor library with automatic differentiation for deep learning.

This library has no dependencies, it's written from scratch in C.

My approach was building a small set of base operations that can be used to build more complicated operations like matmul, sort of like tinygrad.

<!-- TODO: compile mnist example with emscripten, host at mnist.andrew.industries -->

## current features

- [x] tensor
- [x] backwards mode autodifferentiation
- [ ] ops
    - [x] add
    - [x] mul
    - [x] sum along an axis
    - [x] relu
    - [x] reshape
    - [ ] expand
    - [ ] matmul/dot (indirectly, using reshape, expand, mul and sum)
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
    ttuple* input_shape = ttuple_build(2, 4, 6); // shape (4, 6)
    tt* a = tt_uniform(input_shape, -10, 10, true);
    tt* b = tt_uniform(input_shape, -10, 10, true);
    tt* a_b_mul = tt_mul(a, b);
    tt* sum = tt_sum(a_b_mul, -1);
    tgraph* comp_graph = tgraph_build(sum);
    tgraph_zeroed(comp_graph);
    tgraph_backprop(comp_graph);
```

### matmul

There is no direct matmul op. You have to do it manually with a reshape, mul and sum.

There is a function with all this behaviour though.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of matmul -->

### batchnorm

There is also no direct batch norm op. You have to do it manually with add and mul.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of matmul -->

### convolution

Convolutions are done manually however.

```c
// TODO: give examples
```

<!-- TODO: show image of graph of matmul -->

## examples

- [ ] mnist handwritten digit recognition

<!--  TODO: Add a image/gif as example  -->

## future ideas

- python bindings
- different backends (opencl/vulkan, cuda, metal, avx/sse, triton, rocm, tenstorrent)
- other convolution implementations (singular value decomposition, FFT, winograd)
