/*
 * nn, a simple neuron network framework in C. Copyright 2021 lucasart.
 *
 * c-chess-cli is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * c-chess-cli is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program. If
 * not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <stdbool.h>
#include <stddef.h>

// Activation functions
typedef double(*nn_func_t)(double);

enum {
    NN_LINEAR,  // y = x
    NN_RELU,  // y = max(0, x)
    NN_SIGMOID  // y = 1 / (1 + exp(-x))
};

typedef struct {
    // Neurons on this layer
    size_t neuronCnt;
    double *neurons;
    double *deltas;  // derivative of the error wrt each neuron's input (NULL for input layer)
    nn_func_t act, actDerinv;  // activation function and its derinv (NULL for input layer)

    // Connextion to the next layer
    double *weights;  // (neuronCnt + 1) * nextLayer.neuronCnt (NULL for output layer)
} nn_layer_t;

void nn_layer_print(const nn_layer_t *layer, size_t nextLayerNeuronCnt, const char *what);

typedef struct {
    // For best performance, allocate everything in one memory block. Layout is:
    // - double weights[weightCnt]
    // - double neurons[neuronCnt]
    // - double deltas[neuronCnt - layers[0].neuronCnt] (input layer has no deltas)
    double *block;
    size_t weightCnt, neuronCnt;

    // For practicality, organize the above memory block by layer
    nn_layer_t *layers;
    size_t layerCnt;
} nn_network_t;

// Create a network (zero initialized):
// - layerCnt: how many layers, including input and output layer (must be >= 2).
// - neuronCnts[layerCnt]: how many neurons per layer.
// - actIds[layerCnt - 1]: activation function for each layer (starts with layer 1, because layer 0
//   is just the input layer which has no activation function).
nn_network_t nn_network_init(size_t layerCnt, size_t *neuronCnts, int *actIds);

// Releases memory (and sets *nn = {0} for good measure)
void nn_network_destroy(nn_network_t *nn);

// Display the network in human readable form on stdout. what is a string of characters, used to say
// what you want to print:
// - 'n': print the neurons
// - 'w': print the weights
// - 'd': print the deltas (computed by backprop)
void nn_network_print(const nn_network_t *nn, const char *what);

// Run the network forward. inputs may be:
// - double[nn->layers[0].neuronCnt]: copies the inputs[] array into nn->layers[0].neurons[].
// - NULL: assumes inputs are already placed in nn->layers[0].neurons[].
void nn_run(const nn_network_t *nn, const double *inputs);

// Compute the deltas based on an expected output. You must first run the network forward:
// - given a training example = (inputs, outputs);
// - step 1: nn_run(nn, inputs);
// - step 2: nn_backprop(nn, output).
void nn_backprop(const nn_network_t *nn, const double *outputs);
