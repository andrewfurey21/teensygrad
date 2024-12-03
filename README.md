# libtensor

<!--  TODO: Show image of generated graph + mnist example  -->

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
    - [x] expand
    - [x] matmul/dot (indirectly, using reshape, expand, mul and sum)
    - [x] flatten (just a reshape)
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

### how are matmul, batch norms and convolutions done?

There is no direct matmul op. You have to do it manually with a reshape, expand, mul and sum.

<!-- TODO: show image from excalidraw doing matmul -->

There is a function with all this behaviour though.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of matmul -->

There is also no direct batch norm op. You have to do it manually with add and mul.

```c
// TODO: give examples
```
<!-- TODO: show image of graph of batch norm -->

Convolutions are done manually for the moment.

## examples

- [ ] mnist handwritten digit recognition

## future ideas

- different backends (opencl/vulkan, cuda, metal, avx/sse, triton, rocm, tenstorrent)
- other convolution implementations (singular value decomposition, FFT, winograd)
- different types (bfloat, mx-compliant)
- python bindings
- could totally do a refactor, might be nice to have a context like ggml
- could do permute+pad op then i think convs/maxpools could done indirectly as well
