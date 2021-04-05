#pragma once
#include <stddef.h>

double nn_linear(double x);
double nn_relu(double x);
double nn_sigmoid(double x);
typedef double(*nn_act_t)(double);

typedef struct {
    double *inputs;  // vector of inputCnt elements
    double *weights;  // vector of (inputCnt + 1) * outputCnt elements
    nn_act_t act;  // activation function (pointer)
    double *outputs;  // vector of outputCnt elements. this is NOT owned by nn_layer_t.
    size_t inputCnt, outputCnt;
} nn_layer_t;

nn_layer_t nn_layer_init(size_t inputCnt, nn_act_t act, size_t outputCnt, double *outputs);
void nn_layer_destroy(nn_layer_t *nn);

typedef struct {
    nn_layer_t *layers;
    size_t layersCnt;
} nn_network_t;

nn_network_t nn_network_init(size_t layersCnt, size_t *intputCnts, nn_act_t *acts,
    size_t outputCnt);
void nn_network_destroy(nn_network_t *nn);
