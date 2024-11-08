#include "assert.h"
#include "stdbool.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"

#include "../include/teensygrad.h"
#include <stdint.h>

// TODO: storage: getitem, setitem, incref, decref , logical to physical

tstorage* tstorage_new(uint64_t buffer_length) {
    float* buffer = (float*)malloc(buffer_length);
    tstorage* storage = (tstorage*)malloc(sizeof(tstorage));
    storage->buffer = buffer;
    storage->refcount = 1;
    storage->size = buffer_length;
    return storage;
}

void tstorage_free(tstorage* s) {
    free(s->buffer);
    free(s);
}

float tstorage_getitem(tstorage* s, uint64_t index) {
    assert(index > 0 && index < s->size);
    return s->buffer[index];
}

void tstorage_setitem(tstorage* s, uint64_t index, float val) {
    assert(index > 0 && index < s->size);
    s->buffer[index] = val;
}

void tstorage_inc_refcount(tstorage* s) {
    s->refcount++;
}

void tstorage_dec_refcount(tstorage* s) {
    s->refcount--;
    if (s->refcount <= 0) {
        tstorage_free(s);
    }
}

// TODO: test please
uint64_t tstorage_logical_to_physical(tt* t, ttuple* logical_index) {
    assert(logical_index->size == t->data->size);
    assert(logical_index->size == t->strides->size);

    uint64_t index = 0;
    for (int i = 0; i < logical_index->size; i++) {
        index += logical_index->items[i] * t->strides->items[i];
    }
    return index + t->offset;
}


tt *tt_zeros(ttuple *s, bool requires_grad) {
  uint64_t size = ttuple_mul(s);
  float *buffer = (float *)calloc(size, size * (uint64_t)sizeof(float));
  ttuple *copy = ttuple_copy(s);

  tt *grads = NULL;
  if (requires_grad) {
    grads = tt_zeros(s, false);
  }

  tt *t = (tt *)malloc(sizeof(tt));

  t->shape = copy;
  t->buffer = buffer;
  t->size = size;
  t->requires_grad = requires_grad;
  t->parents = NULL;
  t->op = NOOP;
  t->grads = grads;
  t->_backwards = NULL;
  return t;
}

tt *tt_ones(ttuple *s, bool requires_grad) {
  tt *ones = tt_zeros(s, requires_grad);
  for (size_t i = 0; i < ones->size; i++) {
    ones->buffer[i] = 1.0f;
  }
  return ones;
}

tt *tt_from_buffer(ttuple *s, float *buffer, bool requires_grad) {
  tt *ret = (tt *)malloc(sizeof(tt));
  ttuple *copy = ttuple_copy(s);
  ret->shape = copy;
  uint64_t size = ttuple_mul(s);

  ret->buffer = buffer;
  ret->size = size;

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

tt *tt_fill(ttuple *s, float fill_value, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->size; i++) {
    t->buffer[i] = fill_value;
  }
  return t;
}

tt *tt_linspace(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->size; i++) {
    t->buffer[i] = (max - min) / (float)t->size * i + min; //-fabs(min);
  }
  return t;
}

tt *tt_uniform(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_zeros(s, requires_grad);
  for (uint64_t i = 0; i < t->size; i++) {
    t->buffer[i] = (float)rand() / (float)RAND_MAX * (max - min) + min;
  }
  return t;
}

tt *tt_uniformint(ttuple *s, float min, float max, bool requires_grad) {
  tt *t = tt_uniform(s, min, max, requires_grad);
  for (uint64_t i = 0; i < t->size; i++) {
    t->buffer[i] = (float)(int)t->buffer[i];
  }
  return t;
}

void tt_copy_buffer(tt *dest, tt *src) {
  assert(ttuple_equal(dest->shape, src->shape) &&
         "Tensors are not the same shape.");
  for (uint64_t i = 0; i < dest->size; i++) {
    dest->buffer[i] = src->buffer[i];
  }
}

tt *tt_copy(tt *original, bool requires_grad) {
  ttuple *shape_copy = ttuple_copy(original->shape);
  tt *tensor_copy = tt_zeros(shape_copy, requires_grad);
  tt_copy_buffer(tensor_copy, original);
  return tensor_copy;
}

void tt_to_zeros(tt *t) { memset(t->buffer, 0, t->size * 4); }

void tt_to_n(tt *t, float n) {
  for (uint32_t i = 0; i < t->size; i++) {
    t->buffer[i] = n;
  }
}

void tt_print(tt *t) {
  printf("teensy tensor: \n  ");
  if (!t) {
    printf("values: (null)\n");
    return;
  }
  ttuple_print(t->shape);
  if (t->requires_grad) {
    printf("  op: ");
    print_op_string(t->op);
  }
  printf("  values: [ ");
  for (int i = 0; i < t->size; i++) {
    printf("%f, ", t->buffer[i]);
  }
  printf("]\n");
}

void tt_free(tt *t) {
  ttuple_free(t->shape);
  free(t->buffer);
  free(t->parents);
  if (t->requires_grad) {
    tt_free(t->grads); // make sure grads cant have grads
  }
  free(t);
}

// -----------------------------------------------------------
// Ops + derivatives
// Unary ops
void _neg_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tt *grads = tt_fill(self->shape, -1.0f, false);
  tt *mul_grads = tt_mul(grads, self->grads);
  tt *acc_grads = tt_add(mul_grads, self->parents[0]->grads);
  tt_free(self->parents[0]->grads);
  tt_free(grads);
  tt_free(mul_grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_neg(tt *a) {
  ttuple *copy = ttuple_copy(a->shape);

  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(NEG) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, a->requires_grad);
  t->parents = parents;
  t->op = NEG;
  t->_backwards = &_neg_backwards;
  for (uint64_t i = 0; i < a->size; i++) {
    t->buffer[i] = -a->buffer[i];
  }

  return t;
}

void _relu_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }

  tt *grads = tt_zeros(self->shape, false);
  for (size_t i = 0; i < self->parents[0]->size; i++) {
    if (self->parents[0]->buffer[i] > 0) {
      grads->buffer[i] = 1;
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
  ttuple *copy = ttuple_copy(a->shape);
  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RELU) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(copy, a->requires_grad);
  t->parents = parents;
  t->op = RELU;
  t->_backwards = &_relu_backwards;

  for (uint64_t i = 0; i < a->size; i++) {
    t->buffer[i] = a->buffer[i] * (a->buffer[i] > 0);
  }

  return t;
}
// Binary ops
void _add_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    // self->grads are the grads, must accumulate.
    tt *grads_0 = tt_add(self->grads, self->parents[0]->grads);
    tt_free(self->parents[0]->grads);
    self->parents[0]->grads = grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_add(self->grads, self->parents[1]->grads);
    tt_free(self->parents[1]->grads);
    self->parents[1]->grads = grads_1;
  }
}

tt *tt_add(tt *a, tt *b) {
  assert(ttuple_equal(a->shape, b->shape) && "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;

  tt **parents = NULL;
  // irrelevant if not requires_grad
  if (requires_grad) {
    parents = (tt **)malloc(top_radix(ADD) * sizeof(tt *));
    parents[0] = a;
    parents[1] = b;
  }

  tt *t = tt_zeros(copy, requires_grad);
  t->parents = parents;
  t->op = ADD;
  t->_backwards = &_add_backwards;
  for (uint64_t i = 0; i < a->size; i++) {
    t->buffer[i] = a->buffer[i] + b->buffer[i];
  }

  return t;
}

void _mul_backwards(tt *self) {
  if (self->parents[0]->requires_grad) {
    tt *grads_0 = tt_mul(self->grads, self->parents[1]);
    tt *acc_grads_0 = tt_add(grads_0, self->parents[0]->grads);
    tt_free(self->parents[0]->grads);
    tt_free(grads_0);
    self->parents[0]->grads = acc_grads_0;
  }

  if (self->parents[1]->requires_grad) {
    tt *grads_1 = tt_mul(self->grads, self->parents[0]);
    tt *acc_grads_1 = tt_add(grads_1, self->parents[1]->grads);
    tt_free(self->parents[1]->grads);
    tt_free(grads_1);
    self->parents[1]->grads = acc_grads_1;
  }
}

tt *tt_mul(tt *a, tt *b) {
  assert(ttuple_equal(a->shape, b->shape) && "Tensors are not the same shape.");
  ttuple *copy = ttuple_copy(a->shape);

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

  for (uint64_t i = 0; i < a->size; i++) {
    t->buffer[i] = a->buffer[i] * b->buffer[i];
  }

  return t;
}

// Reduce ops
void _sum_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tt *expanded_grads =
      tt_fill(self->parents[0]->shape, self->grads->buffer[0], false);
  tt *acc_grads = tt_add(self->parents[0]->grads, expanded_grads);

  tt_free(self->parents[0]->grads);
  tt_free(expanded_grads);

  self->parents[0]->grads = acc_grads;
}

tt *tt_sum(tt *a, ttuple *axes) {
  assert(!ttuple_duplicates(axes));
  uint32_t new_size = axes->size - a->shape->size;
  assert(new_size >= 0);
  // new_size = 0 means array of size 1 (scalar)
  ttuple *new_shape = ttuple_build(1, 1);
  tt **parents = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(SUM_REDUCE) * sizeof(tt *));
    parents[0] = a;
  }

  tt *t = tt_zeros(new_shape, a->requires_grad);
  t->parents = parents;
  t->op = SUM_REDUCE;
  t->_backwards = &_sum_backwards;

  double sum = 0.0f;
  for (uint64_t i = 0; i < a->size; i++) {
    sum += a->buffer[i];
  }

  t->buffer[0] = sum;

  return t;
}

// Movement ops
// Permute
// void _permute_backwards( tt* self) {
//
// }
//
//  tt* tt_permute( tt* t,  ttuple* axes) {
//     assert(t->shape->size == axes->size);
//      tt* tensor_copy = tt_copy(t, t->requires_grad);
//      ttuple* permuted_shape = ttuple_permute(t->shape, axes);
//     uint64_t buf_size = buflen(tensor_copy->shape);
//
//     for (uint64_t i = 0; i < buf_size; i++) {
//         size_t new_index = 0;
//         for (int j = 0; j < axes->size; j++) {
//             uint32_t shape_coord = permuted_shape->dims[j];
//             uint32_t axis = axes->dims[i];
//             uint32_t old_coord =
//             new_index += axis*shape_coord*old_coord;
//         }
//         tensor_copy->buffer[new_index] = t->buffer[i];
//
//     }
//     free(tensor_copy->shape);
//     tensor_copy->shape = permuted_shape;
//     return tensor_copy;
// }

// Expand
void _expand_backwards(tt *self) {}

tt *tt_expand(tt *a, ttuple *shape) {
  int diff = shape->size - a->shape->size;
  assert(diff >= 0 && "shape must have higher dimensions");
  tt **parents = NULL;

  for (int i = 0; i < a->shape->size; i++) {
    assert(shape->dims[i + diff] == a->shape->dims[i]);
  }
  tt *expanded_tensor = tt_zeros(shape, a->requires_grad);

  for (uint32_t i = 0; i < expanded_tensor->size; i++) {
    expanded_tensor->buffer[i] = a->buffer[i % a->size];
  }

  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(EXPAND) * sizeof(tt *));
    parents[0] = a;
  }

  expanded_tensor->parents = parents;
  expanded_tensor->_backwards = &_expand_backwards;
  expanded_tensor->op = EXPAND;

  return expanded_tensor;
}

// Reshape
void _reshape_backwards(tt *self) {
  if (!self->parents[0]->requires_grad) {
    return;
  }
  tt *grads = tt_reshape(self->grads, self->parents[0]->shape);
  tt *acc_grads = tt_add(grads, self->parents[0]->grads);

  free(grads);
  self->parents[0]->grads = acc_grads;
}

tt *tt_reshape(tt *a, ttuple *new_shape) {
  ttuple *new_shape_copy = ttuple_copy(new_shape);
  assert(buflen(new_shape) == buflen(a->shape));
  tt **parents = NULL;
  tt *reshaped_grads = NULL;
  if (a->requires_grad) {
    parents = (tt **)malloc(top_radix(RESHAPE) * sizeof(tt *));
    parents[0] = a;
    reshaped_grads = tt_reshape(a->grads, new_shape_copy);
  }
  tt *t = tt_copy(a, a->requires_grad);
  free(t->grads);
  t->shape = new_shape_copy;
  t->parents = parents;
  t->op = RESHAPE;
  t->_backwards = &_reshape_backwards;
  t->grads = reshaped_grads;
  return t;
}
