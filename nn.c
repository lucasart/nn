#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

double nn_linear(double x) { return x; }
double nn_relu(double x) { return x > 0 ? x : 0; }
double nn_sigmoid(double x) { return 1 / (1 + exp(-x)); }

double nn_linear_derinv(double y) { (void)y; return 1; }
double nn_relu_derinv(double y) { return y > 0 ? 1 : 0; }
double nn_sigmoid_derinv(double y) { return y * (1 - y); }

static void nn_print_array(size_t n, const double *array)
{
    for (size_t i = 0; i < n; i++)
        printf(i == n - 1 ? "%f\n": "%f,", array[i]);
}

void nn_layer_print(const nn_layer_t *layer, size_t nextLayerNeuronCnt, const char *what)
{
    if (strchr(what, 'n')) {
        printf("neurons[%zu]=", layer->neuronCnt);
        nn_print_array(layer->neuronCnt, layer->neurons);
    }

    if (strchr(what, 'd') && layer->deltas) {
        printf("deltas[%zu]=", layer->neuronCnt);
        nn_print_array(layer->neuronCnt, layer->deltas);
    }

    if (strchr(what, 'w') && layer->weights) {
        printf("weights[%zu][%zu]=\n", nextLayerNeuronCnt, layer->neuronCnt + 1);
        const double *w = layer->weights;

        for (size_t j = 0; j < nextLayerNeuronCnt; j++) {
            printf("%zu:", j);
            nn_print_array(layer->neuronCnt + 1, w);
            w += layer->neuronCnt + 1;
        }
    }
}

nn_network_t nn_network_init(size_t layerCnt, size_t *neuronCnts, nn_func_t *acts,
    nn_func_t *actDerinvs)
{
    nn_network_t nn = {
        .layers = malloc(layerCnt * sizeof(nn_layer_t)),
        .layerCnt = layerCnt
    };

    for (size_t i = 0; i < layerCnt; i++) {
        nn.neuronCnt += neuronCnts[i];

        if (i+1 < layerCnt)
            nn.weightCnt += (neuronCnts[i] + 1) * neuronCnts[i+1];
    }

    nn.block = calloc(nn.weightCnt + 2 * nn.neuronCnt - neuronCnts[0], sizeof(double *));

    // First layer
    nn.layers[0] = (nn_layer_t){
        .neuronCnt = neuronCnts[0],
        .weights = nn.block,
        .neurons = nn.block + nn.weightCnt
        // .deltas .act .actDerinv are NULL
    };

    for (size_t i = 1; i < layerCnt; i++)
        nn.layers[i] = (nn_layer_t){
            .neuronCnt = neuronCnts[i],
            .neurons = nn.layers[i-1].neurons + neuronCnts[i-1],
            .deltas = i > 1
                ? nn.layers[i-1].deltas + neuronCnts[i-1]
                : nn.block + nn.weightCnt + nn.neuronCnt,
            .act = acts[i-1],
            .actDerinv = actDerinvs[i-1],
            .weights = i+1 < layerCnt
                ? nn.layers[i-1].weights + (neuronCnts[i-1] + 1) * neuronCnts[i]
                : NULL
        };

    return nn;
}

void nn_network_destroy(nn_network_t *nn)
{
    free(nn->block);
    free(nn->layers);
    *nn = (nn_network_t){0};
}

void nn_network_print(const nn_network_t *nn, const char *what)
{
    for (size_t i = 0; i < nn->layerCnt; i++) {
        printf("layer #%zu:\n", i);
        nn_layer_print(&nn->layers[i], i+1 < nn->layerCnt ? nn->layers[i+1].neuronCnt : 0, what);
    }
}

void nn_run(const nn_network_t *nn)
{
    for (size_t l = 1; l < nn->layerCnt; l++) {
        const double *w = nn->layers[l-1].weights;

        for (size_t o = 0; o < nn->layers[l].neuronCnt; o++) {
            // start with the biais
            double sum = *w++;

            // then add the sum product
            for (size_t i = 0; i < nn->layers[l-1].neuronCnt; i++)
                sum += nn->layers[l-1].neurons[i] * *w++;

            // apply activation function and store neuron value
            nn->layers[l].neurons[o] = nn->layers[l].act(sum);
        }
    }
}
