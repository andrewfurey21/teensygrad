#include "../include/tensor.h"
#include "assert.h"
#include "stdio.h"
#include "stdlib.h"
#include <stdint.h>
#include <time.h>

// TODO:
// encapsulating ops with functions is broken, i dont think gradients flow
// correcly. check summing with (2, 1, 1) or something with ones get name of
// linspace/arange correct get this working correctly, compare with proper
// tinygrad/pytorch impl variable shapes etc. add to hl_ops or something. need
// to free stuff in function if not being used later. use getenv for batchsize,
// learning_rate, etc other params

// 2d matmul
tt *linear_layer(tt *input, tt *weights) {
  int input_width = input->view->shape->items[1];
  int input_height = input->view->shape->items[0];

  int weights_width = weights->view->shape->items[1];
  int weights_height = weights->view->shape->items[0];

  assert(input_width == weights_height);

  ttuple *new_input_shape = ttuple_build(3, input_height, input_width, 1);
  tt *reshaped_input = tt_reshape(input, new_input_shape);

  ttuple *new_weights_shape = ttuple_build(3, 1, weights_height, weights_width);
  tt *reshaped_weights = tt_reshape(weights, new_weights_shape);

  tt *expanded_input = tt_expand(reshaped_input, 2, weights_width);
  tt *expanded_weights = tt_expand(reshaped_weights, 0, input_height);

  tt *mul = tt_mul(expanded_input, expanded_weights);

  tt *output = tt_sum(mul, 1);

  ttuple *new_output_shape = ttuple_zeros(2);
  new_output_shape->items[0] = output->view->shape->items[0];
  new_output_shape->items[1] = output->view->shape->items[2];

  tt *reshaped_output = tt_reshape(output, new_output_shape);
  return reshaped_output;
}

tt *flatten(tt *input, int start_dim) {
  assert(start_dim >= 0 && start_dim < input->view->shape->size);
  ttuple *new_shape = ttuple_zeros(start_dim + 1);
  uint64_t end = 1;
  for (int i = 0; i < input->view->shape->size; i++) {
    if (i >= start_dim) {
      end *= input->view->shape->items[i];
    } else {
      new_shape->items[i] = input->view->shape->items[i];
    }
  }
  new_shape->items[start_dim] = end;
  tt *flattened = tt_reshape(input, new_shape);
  return flattened;
}

int main(void) {
  srand(time(NULL));

  int batch_size = 16;

  ttuple *input_shape = ttuple_build(2, batch_size, 576);
  tt *input = tt_linspace(input_shape, 0, 10, batch_size * 576, true);
  
  ttuple *weights_shape = ttuple_build(2, 576, 10);
  tt *weights = tt_linspace(weights_shape, 0, 10, 576*10, true);
  
  tt *mm = linear_layer(input, weights);
}
