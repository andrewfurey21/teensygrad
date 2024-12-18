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
// learning_rate, etc other params. add training param to each function

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

tt *mean(tt* input, int axis) {
  int size;
  if (axis == -1) {
    size = ttuple_prod(input->view->shape);
  } else {
    size = input->view->shape->items[axis];
  }
  tt* summed = tt_sum(input, axis);
  tt* div = tt_fill(summed->view->shape, 1.0f / size, true);
  return tt_mul(summed, div);
}

tt *log_softmax(tt* input, int axis) {
  tt* exp = tt_exp(input);
  tt* sum_exp = tt_sum(exp, axis);
  tt* log_sum_exp = tt_log(sum_exp);
  tt* expand_log_sum_exp = tt_expand(log_sum_exp, axis, input->view->shape->items[axis]);// would be fixed with broadcasting
  return tt_sub(input, expand_log_sum_exp);
}

int main(void) {
  srand(time(NULL));

  ttuple* shape = ttuple_build(4, 1, 1, 4, 4);
  tt* input = tt_linspace(shape, 1, 16, 1*1*4*4, true);

  tt* other = tt_fill(shape, 3.5, true);

  int axis = 3;
  
  tt* lsm = log_softmax(input, 3);

  tt* sum = tt_sum(lsm, -1);

  tgraph* comp_graph = tgraph_build(sum);
  tgraph_zeroed(comp_graph);
  tgraph_backprop(comp_graph);


  tt_print(sum, true, true);
  tt_print(lsm, true, true);
  tt_print(input, true, true);

}
