#pragma once
#include <stddef.h>

typedef struct {
    double *inputs;  // vector of inputCnt elements
    double *weights;  // vector of (inputCnt + 1) * outputCnt elements
    double *outputs;  // vector of outputCnt elements. this is NOT owned by nn_layer_t.
    size_t inputCnt, outputCnt;
} nn_layer_t;

nn_layer_t nn_layer_init(size_t inputCnt, size_t outputCnt, double *outputs);
void nn_layer_destroy(nn_layer_t *nn);

typedef struct {
    nn_layer_t *layers;
    size_t layersCnt;
} nn_network_t;

nn_network_t nn_network_init(size_t layersCnt, size_t *intputCnts, size_t outputCnt);
void nn_network_destroy(nn_network_t *nn);
