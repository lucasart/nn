#pragma once
#include <stddef.h>

typedef struct {
    double *inputs;  // vector of inputLen elements
    double *weights;  // vector of (inputLen + 1) * outputLen elements
    double *outputs;  // vector of outputLen elements. this is NOT owned by nn_layer_t.
    size_t inputLen, outputLen;
} nn_layer_t;

nn_layer_t nn_layer_init(size_t inputLen, size_t outputLen, double *outputs);
void nn_layer_destroy(nn_layer_t *nn);
