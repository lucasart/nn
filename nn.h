#pragma once
#include <stdbool.h>
#include <stddef.h>

typedef double(*nn_func_t)(double);

// Activation functions: y = f(x)
double nn_linear(double x);
double nn_relu(double x);
double nn_sigmoid(double x);

// Derivative on the inverse: f'(f^{-1}(y)) = f'(x)
double nn_linear_derinv(double y);
double nn_relu_derinv(double y);
double nn_sigmoid_derinv(double y);

// An layer describes the connexion (input layer, weights) -> output layer
typedef struct {
    // Neurons on this layer
    size_t neuronCnt;
    double *neurons;
    double *deltas;  // derivative of the error wrt each neuron (NULL for input layer)
    nn_func_t act, actDerinv;  // activation function and its derinv (NULL for input layer)

    // Connextion to the next layer
    double *weights;  // (neuronCnt + 1) * nextLayer.neuronCnt (NULL for output layer)
} nn_layer_t;

void nn_layer_print(const nn_layer_t *layer, size_t nextLayerNeuronCnt, const char *what);

typedef struct {
    // For best performance, all in one memory block. Layout is:
    // - double weights[weightCnt]
    // - double neurons[neuronCnt]
    // - double deltas[neuronCnt - layers[0].neuronCnt] (input layer has no deltas)
    double *block;
    size_t weightCnt, neuronCnt;

    // For practicality, organize the above memory block by layer
    nn_layer_t *layers;
    size_t layerCnt;
} nn_network_t;

nn_network_t nn_network_init(size_t layerCnt, size_t *neuronCnts, nn_func_t *acts,
    nn_func_t *actDerinvs);
void nn_network_destroy(nn_network_t *nn);
void nn_network_print(const nn_network_t *nn, const char *what);
void nn_run(const nn_network_t *nn);
