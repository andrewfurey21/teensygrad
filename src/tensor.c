#include "assert.h"
#include "malloc.h"
#include "math.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"

#include "../include/tensor.h"
#include <math.h>
#include <stdint.h>

tstorage *tstorage_new(uint64_t buffer_length) {
  float *buffer = (float *)calloc(buffer_length, sizeof(float));
  tstorage *storage = (tstorage *)malloc(sizeof(tstorage));
  storage->buffer = buffer;
  storage->refcount = 1;
  storage->size = buffer_length;
  return storage;
}

// maybe not a good idea memory wise but whatever
tstorage *tstorage_from_buffer(uint64_t size, float *buffer) {
  // TODO: check if this works
  //
  // uint64_t size = malloc_usable_size(buffer)/sizeof(float);
  float *buffer_copy = (float *)calloc(size, sizeof(float));
  for (int i = 0; i < size; i++) {
    buffer_copy[i] = buffer[i];
  }
  tstorage *data = (tstorage *)malloc(sizeof(tstorage));
  data->size = size;
  data->buffer = buffer_copy;
  data->refcount = 1;
  return data;
}

void tstorage_free(tstorage *s) {
  free(s->buffer);
  free(s);
}

float tstorage_getitem(tstorage *s, uint64_t index) {
  assert(index >= 0 && index < s->size);
  return s->buffer[index];
}

void tstorage_setitem(tstorage *s, uint64_t index, float val) {
  assert(index >= 0 && index < s->size);
  s->buffer[index] = val;
}

void tstorage_inc_refcount(tstorage *s) { s->refcount++; }

void tstorage_dec_refcount(tstorage *s) {
  s->refcount--;
  if (s->refcount <= 0) {
    tstorage_free(s);
  }
}

tstorage *tstorage_copy(tstorage *s) {
  return tstorage_from_buffer(s->size, s->buffer);
}

void tstorage_to_zeros(tstorage *s) {
  free(s->buffer);
  s->buffer = (float *)calloc(s->size, sizeof(float));
}

// TODO: test please
uint64_t tstorage_logical_to_physical(tt *t, ttuple *logical_index) {
  ttuple *t_strides = t->view->strides;
  assert(logical_index->size == t->data->size);
  assert(logical_index->size == t_strides->size);

  uint64_t index = 0;
  for (int i = 0; i < logical_index->size; i++) {
    index += logical_index->items[i] * t_strides->items[i];
  }
  return index + t->view->offset;
}

void tview_free(tview *view) {
  ttuple_free(view->shape);
  ttuple_free(view->strides);
  free(view);
}

tt *tt_zeros(ttuple *s, bool requires_grad) {
  uint64_t size = ttuple_prod(s);
  ttuple *copy = ttuple_copy(s);

  tstorage *data = tstorage_new(size);

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(s, false);
  }

  tt *t = (tt *)malloc(sizeof(tt));

  // TODO: Make functions for views
  tview *view = (tview *)malloc(sizeof(tview));
  t->view = view;
  t->view->shape = copy;
  t->view->strides = ttuple_ones(copy->size);
  t->view->offset = 0;

  t->data = data;
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;
  return t;
}

tt *tt_ones(ttuple *s, bool requires_grad) {
  tt *ones = tt_zeros(s, requires_grad);
  for (size_t i = 0; i < ones->data->size; i++) {
    tstorage_setitem(ones->data, i, 1.0f);
  }
  return ones;
}

tt *tt_from_buffer(ttuple *s, float *buffer, bool requires_grad) {
  uint64_t size = ttuple_prod(s);
  tstorage *data = tstorage_from_buffer(size, buffer);

  tt *ret = (tt *)malloc(sizeof(tt));
  ttuple *copy = ttuple_copy(s);
  ttuple *strides = ttuple_ones(copy->size);

  tview *view = (tview *)malloc(sizeof(tview));
  ret->view = view;

  ret->view->shape = copy;
  ret->view->strides = strides;
  ret->view->offset = 0;

  ret->data = data;

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(s, false);
  }
  ret->op = NOOP;
  ret->parents = NULL;
  ret->requires_grad = requires_grad;
  ret->_backwards = NULL;
  ret->grads = grads;
  return ret;
}

// TODO: test please
float tt_getindex(tt *self, ttuple *s) {
  ttuple *self_shape = self->view->shape;
  assert(s->size == self->view->shape->size);
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    assert(s->items[i] < self_shape->items[i]);
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  return self->data->buffer[index];
}

void tt_setindex(tt *self, ttuple *s, float num) {
  ttuple *self_shape = self->view->shape;
  assert(s->size == self->view->shape->size);
  uint64_t index = 0;
  for (int i = 0; i < s->size; i++) {
    assert(s->items[i] < self_shape->items[i]);
    uint64_t mul = 1;
    for (int j = i + 1; j < s->size; j++) {
      mul *= self_shape->items[j];
    }
    index += mul * s->items[i];
  }
  self->data->buffer[index] = num;
}

tt *tt_fill(ttuple *s, float fill_value, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    tstorage_setitem(t->data, i, fill_value);
  }
  return t;
}

tt *tt_linspace(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (max - min) / (float)t->data->size * i + min;
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniform(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = (float)rand() / (float)RAND_MAX * (max - min) + min;
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_uniformint(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_uniform(s, min, max, requires_grad);
  for (uint64_t i = 0; i < t->data->size; i++) {
    float value = round(tstorage_getitem(t->data, i));
    tstorage_setitem(t->data, i, value);
  }
  return t;
}

tt *tt_copy(tt *original, bool requires_grad) {
  ttuple *shape = ttuple_copy(original->view->shape);
  ttuple *strides = ttuple_copy(original->view->strides);

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(shape, false);
  }

  tt *t = (tt *)malloc(sizeof(tt));

  tview *view = (tview *)malloc(sizeof(tview));
  t->view = view;
  t->view->shape = shape;
  t->view->strides = strides;
  t->view->offset = 0;

  t->data = tstorage_copy(original->data);
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;

  return t;
}

void tt_to_zeros(tt *t) { tstorage_to_zeros(t->data); }

void tt_to_n(struct tt *t, float n) {
  for (int i = 0; i < t->data->size; i++) {
    tstorage_setitem(t->data, i, n);
  }
}

void tt_print(tt *t) {
  printf("tensor: \n  ");
  if (!t) {
    printf("values: (null)\n");
    return;
  }
  ttuple_print(t->view->shape);
  if (t->requires_grad) {
    printf("  op: ");
    print_op_string(t->op);
  } else {
    // TODO: if not requires grad, maybe check where it came from, like grad of mul or something
    printf("  NO GRADS\n");
  }
  printf("  values: [ ");
  for (int i = 0; i < t->data->size; i++) {
    printf("%f, ", t->data->buffer[i]);
  }
  printf("]\n");
}

// should probably free any grads from children.
void tt_free(tt *t) {
  tview_free(t->view);
  tstorage_dec_refcount(t->data);

  free(t->parents);
  if (t->requires_grad) {
    tt_free(t->grads); // make sure grads cant have grads
  }
  free(t);
}

void tt_free_parents(tt *t) {
  for (int i = 0; i < top_radix(t->op); i++) {
    tt_free(t->parents[i]);
  }
  free(t->parents);
}

void tt_destroy_grads(tt *t) {
  t->requires_grad = false;
  tt_free(t->grads);
  t->_backwards = NULL;
  t->op = NOOP;
  tt_free_parents(t);
}

void _add_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    // grads get created if requires_grad, so not checking.
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads);
    tt_destroy_grads(grads_0);
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_add(self->grads, self->parents[1]->grads);
    tt_destroy_grads(grads_1);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}

tt *tt_add(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->view->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(ADD) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = ADD;
  t->_backwards = &_add_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] + b->data->buffer[i];
  }
  return t;
}

void _mul_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_mul(self->grads, self->parents[1]);
    tt *acc_grads_0 = tt_add(grads_0, self->parents[0]->grads);
    tt_destroy_grads(acc_grads_0);
    tt_free(self->parents[0]->grads);
    tt_free(grads_0);
    self->parents[0]->grads = acc_grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_mul(self->grads, self->parents[0]);
    tt *acc_grads_1 = tt_add(grads_1, self->parents[1]->grads);
    tt_destroy_grads(acc_grads_1);
    tt_free(self->parents[1]->grads);
    tt_free(grads_1);
    self->parents[1]->grads = acc_grads_1;
  }
}

tt *tt_mul(tt *a, tt *b) {
  assert(ttuple_equal(a->view->shape, b->view->shape) &&
         "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->view->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(MUL) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = MUL;
  t->_backwards = &_mul_backwards;
  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * b->data->buffer[i];
  }
  return t;
}

void _sum_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  ttuple *unit_shape = ttuple_build(1, 1);

  ttuple *self_shape = self->view->shape;
  ttuple *par_shape = self->parents[0]->view->shape;
  if (ttuple_equal(unit_shape, self_shape)) {
    tt *expanded_grads =
        tt_fill(par_shape, self->grads->data->buffer[0], false);
    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads);
    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  } else {
    int expand_axis = 0;
    assert(self_shape->size == par_shape->size);

    // TODO: i don't think this works if one of the dimensions was always 1
    for (int i = 0; i < self_shape->size; i++) {
      if (self_shape->items[i] == 1 && par_shape->items[i] != 1) {
        expand_axis = i;
        break;
      }
    }

    tt *expanded_grads = tt_zeros(par_shape, false);
    ttuple *current = ttuple_zeros(par_shape->size);

    uint64_t along_axis = par_shape->items[expand_axis];
    for (uint64_t i = 0; i < self->grads->data->size; i++) {
      //expanding
      for (uint64_t j = 0; j < along_axis; j++) {
        ttuple *current_grads = ttuple_copy(current);
        current_grads->items[expand_axis] = 0;
        float num = tt_getindex(self->grads, current_grads);
        tt_setindex(expanded_grads, current, num);
        current->items[expand_axis]++;
        ttuple_free(current_grads);
      }

      current->items[expand_axis] = 0;
      //updating current (with expanded axis set to 0)
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == expand_axis) {
          continue;
        }
        current->items[k]++;
        if (current->items[k] >= par_shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }

    tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads);
    tt_free(self->parents[0]->grads);
    tt_free(expanded_grads);
    self->parents[0]->grads = acc_grads;
  }
  ttuple_free(unit_shape);
}

// axis=-1 => sum up all elements
// currently always keepdims, except for axis=-1
// could seriously use some tests here
tt *tt_sum(tt *a, int axis) {
  assert(axis >= -1 && axis < (int)a->view->shape->size);
  ttuple *new_shape;
  if (axis == -1) {
    new_shape = ttuple_build(1, 1);
  } else {
    new_shape = ttuple_copy(a->view->shape);
    new_shape->items[axis] = 1;
  }

  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(SUM_REDUCE) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(new_shape, a->requires_grad);
  t->parents = parents;
  t->op = SUM_REDUCE;
  t->_backwards = &_sum_backwards;

  if (axis == -1) {
    double sum = 0.0f;
    for (uint64_t i = 0; i < a->data->size; i++) {
      sum += a->data->buffer[i];
    }
    t->data->buffer[0] = sum;
  } else {
    ttuple *stride = ttuple_zeros(a->view->shape->size);
    stride->items[axis] = 1;

    uint64_t along_axis = a->view->shape->items[axis];
    uint64_t num_accumulate = ttuple_prod(a->view->shape) / along_axis;
    ttuple *current = ttuple_zeros(a->view->shape->size);
    for (uint64_t i = 0; i < num_accumulate; i++) {
      float sum = 0.0f;
      for (uint64_t j = 0; j < along_axis; j++) {
        sum += tt_getindex(a, current);
        current->items[axis]++;
      }
      current->items[axis] = 0;
      tt_setindex(t, current, sum);
      // this looks kinda fucked but i think it works
      for (int k = current->size - 1; k >= 0; k--) {
        if (k == axis)
          continue;
        current->items[k]++;
        if (current->items[k] >= a->view->shape->items[k]) {
          current->items[k] = 0;
          continue;
        }
        break;
      }
    }
  }
  return t;
}

void _relu_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }

  tt *grads = tt_zeros(self->view->shape, false);
  for (size_t i = 0; i < self->parents[0]->data->size; i++) {
    if (self->parents[0]->data->buffer[i] > 0) {
      grads->data->buffer[i] = 1;
    }
  }
  tt *mul_grads = tt_mul(self->grads, grads);
  tt *acc_grads = tt_add(self->parents[0]->grads, mul_grads);
  tt_free(grads);
  tt_free(self->parents[0]->grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_relu(tt *a) {
  ttuple *copy = ttuple_copy(a->view->shape);
  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RELU) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, a->requires_grad);
  t->parents = parents;
  t->op = RELU;
  t->_backwards = &_relu_backwards;

  for (uint64_t i = 0; i < a->data->size; i++) {
    t->data->buffer[i] = a->data->buffer[i] * (a->data->buffer[i] > 0);
  }

  return t;
}

void _matmul_backwards(tt *self) {}

tt *tt_matmul(tt *a, tt *b) {
  
}

// Unary ops
// void _neg_backwards(tt *self) {
//   if (!self->parents[0]->requires_grad) {
//     return;
//   }
//   tt *grads = tt_fill(self->shape, -1.0f, false);
//   tt *mul_grads = tt_mul(grads, self->grads);
//   tt *acc_grads = tt_add(mul_grads, self->parents[0]->grads);
//   tt_free(self->parents[0]->grads);
//   tt_free(grads);
//   tt_free(mul_grads);
//   self->parents[0]->grads = acc_grads;
// }
//
// tt *tt_neg(tt *a) {
//   ttuple *shape = ttuple_copy(a->shape);
//   tt *t = tt_zeros(shape, a->requires_grad);
//
//   tt **parents = NULL;
//   if (a->requires_grad) {
//     parents = (tt **)malloc(top_radix(NEG) * sizeof(tt *));
//     parents[0] = a;
//   }
//
//   t->parents = parents;
//   t->op = NEG;
//   t->_backwards = &_neg_backwards;
//
//   for (uint64_t i = 0; i < a->data->size; i++) {
//     float value = tstorage_getitem(a->data, i);
//     tstorage_setitem(t->data, i, -value);
//   }
//
//   return t;
// }
// // Binary ops
//
//
// // Reduce ops
//
// // Movement ops
// // Permute
// // void _permute_backwards( tt* self) {
// //
// // }
// //
// //  tt* tt_permute( tt* t,  ttuple* axes) {
// //     assert(t->shape->size == axes->size);
// //      tt* tensor_copy = tt_copy(t, t->requires_grad);
// //      ttuple* permuted_shape = ttuple_permute(t->shape, axes);
// //     uint64_t buf_size = buflen(tensor_copy->shape);
// //
// //     for (uint64_t i = 0; i < buf_size; i++) {
// //         size_t new_index = 0;
// //         for (int j = 0; j < axes->size; j++) {
// //             uint32_t shape_coord = permuted_shape->dims[j];
// //             uint32_t axis = axes->dims[i];
// //             uint32_t old_coord =
// //             new_index += axis*shape_coord*old_coord;
// //         }
// //         tensor_copy->buffer[new_index] = t->buffer[i];
// //
// //     }
// //     free(tensor_copy->shape);
// //     tensor_copy->shape = permuted_shape;
// //     return tensor_copy;
// // }
//
// // Expand
// void _expand_backwards(tt *self) {}
//
// tt *tt_expand(tt *a, ttuple *shape) {
//   int diff = shape->size - a->shape->size;
//   assert(diff >= 0 && "shape must have higher dimensions");
//   tt **parents = NULL;
//
//   for (int i = 0; i < a->shape->size; i++) {
//     assert(shape->dims[i + diff] == a->shape->dims[i]);
//   }
//   tt *expanded_tensor = tt_zeros(shape, a->requires_grad);
//
//   for (uint32_t i = 0; i < expanded_tensor->size; i++) {
//     expanded_tensor->buffer[i] = a->buffer[i % a->size];
//   }
//
//   if (a->requires_grad) {
//     parents = (tt **)malloc(top_radix(EXPAND) * sizeof(tt *));
//     parents[0] = a;
//   }
//
//   expanded_tensor->parents = parents;
//   expanded_tensor->_backwards = &_expand_backwards;
//   expanded_tensor->op = EXPAND;
//
//   return expanded_tensor;
// }
//
// // Reshape
// void _reshape_backwards(tt *self) {
//   if (!self->parents[0]->requires_grad) {
//     return;
//   }
//   tt *grads = tt_reshape(self->grads, self->parents[0]->shape);
//   tt *acc_grads = tt_add(grads, self->parents[0]->grads);
//
//   free(grads);
//   self->parents[0]->grads = acc_grads;
// }
//
// tt *tt_reshape(tt *a, ttuple *new_shape) {
//   ttuple *new_shape_copy = ttuple_copy(new_shape);
//   assert(buflen(new_shape) == buflen(a->shape));
//   tt **parents = NULL;
//   tt *reshaped_grads = NULL;
//   if (a->requires_grad) {
//     parents = (tt **)malloc(top_radix(RESHAPE) * sizeof(tt *));
//     parents[0] = a;
//     reshaped_grads = tt_reshape(a->grads, new_shape_copy);
//   }
//   tt *t = tt_copy(a, a->requires_grad);
//   free(t->grads);
//   t->shape = new_shape_copy;
//   t->parents = parents;
//   t->op = RESHAPE;
//   t->_backwards = &_reshape_backwards;
//   t->grads = reshaped_grads;
//   return t;
// }
// TODO: padding also
