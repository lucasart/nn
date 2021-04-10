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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

static double nn_linear(double x) { return x; }
static double nn_relu(double x) { return x > 0 ? x : 0; }
static double nn_sigmoid(double x) { return 1 / (1 + exp(-x)); }

static double nn_linear_derinv(double y) { (void)y; return 1; }
static double nn_relu_derinv(double y) { return y > 0 ? 1 : 0; }
static double nn_sigmoid_derinv(double y) { return y * (1 - y); }

static const struct { int id; nn_func_t func, funcDerinv; } actMap[] = {
    {NN_LINEAR, nn_linear, nn_linear_derinv},
    {NN_RELU, nn_relu, nn_relu_derinv},
    {NN_SIGMOID, nn_sigmoid, nn_sigmoid_derinv}
};

void nn_print_array(size_t n, const double *array)
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

nn_network_t nn_network_init(size_t layerCnt, size_t *neuronCnts, int *actIds)
{
    nn_network_t nn = {
        .layers = malloc(layerCnt * sizeof(nn_layer_t)),
        .layerCnt = layerCnt
    };

    for (size_t i = 0; i < layerCnt; i++) {
        nn.neuronCnt += neuronCnts[i];

        if (i+1 < layerCnt)
            nn.weightCnt += (neuronCnts[i] + 1) * neuronCnts[i + 1];
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
            .neurons = nn.layers[i - 1].neurons + neuronCnts[i - 1],
            .deltas = i > 1
                ? nn.layers[i - 1].deltas + neuronCnts[i - 1]
                : nn.block + nn.weightCnt + nn.neuronCnt,
            .act = actMap[actIds[i - 1]].func,
            .actDerinv = actMap[actIds[i - 1]].funcDerinv,
            .weights = i+1 < layerCnt
                ? nn.layers[i - 1].weights + (neuronCnts[i - 1] + 1) * neuronCnts[i]
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
        nn_layer_print(&nn->layers[i], i+1 < nn->layerCnt ? nn->layers[i + 1].neuronCnt : 0, what);
    }
}

void nn_run(const nn_network_t *nn, const double *inputs)
{
    if (inputs)
        memcpy(nn->layers[0].neurons, inputs, nn->layers[0].neuronCnt * sizeof(double *));

    // Clear deltas[]
    memset(nn->block + nn->weightCnt + nn->neuronCnt, 0,
        (nn->neuronCnt - nn->layers[0].neuronCnt) * sizeof(double *));

    for (size_t l = 1; l < nn->layerCnt; l++) {
        const double *w = nn->layers[l - 1].weights;

        for (size_t o = 0; o < nn->layers[l].neuronCnt; o++) {
            double sum = 0;

            // add the sum product
            for (size_t i = 0; i < nn->layers[l - 1].neuronCnt; i++)
                sum += nn->layers[l - 1].neurons[i] * *w++;

            // add the biais
            sum += *w++;

            // apply activation function and store neuron value
            nn->layers[l].neurons[o] = nn->layers[l].act(sum);
        }
    }
}

static void nn_do_backprop(const nn_network_t *nn, const double *outputs, bool absolute)
{
    // Compute deltas on the output layer
    const nn_layer_t *ol = &nn->layers[nn->layerCnt - 1];

    for (size_t j = 0; j < ol->neuronCnt; j++) {
        const double diff = ol->neurons[j] - outputs[j];  // diff = o - t
        ol->deltas[j] = ol->actDerinv(ol->neurons[j]);
        ol->deltas[j] *= absolute
            ? (diff > 0 ? 1 : diff < 0 ? -1 : 0)  // d/do(|diff|) = sign(diff)
            : diff;  // d/do(.5*diff^2) = diff
    }

    // Compute deltas on inner layer l based on deltas from next layer l+1
    for (size_t l = nn->layerCnt - 2; l > 0; l--) {
        const nn_layer_t *cl = &nn->layers[l], *nl = &nn->layers[l + 1];  // current and next layer

        for (size_t j = 0; j < nl->neuronCnt; j++)
            for (size_t i = 0; i < cl->neuronCnt; i++)
                cl->deltas[i] += cl->weights[j * (cl->neuronCnt + 1) + i] * nl->deltas[j];

        for (size_t i = 0; i < cl->neuronCnt; i++)
            cl->deltas[i] *= cl->actDerinv(cl->neurons[i]);
    }
}

void nn_backprop(const nn_network_t *nn, const double *inputs, const double *outputs, bool absolute)
{
    nn_run(nn, inputs);
    nn_do_backprop(nn, outputs, absolute);
}

static void nn_do_gradient(const nn_network_t *nn, double *gradient)
{
    for (size_t l = 0; l < nn->layerCnt - 1; l++)
        for (size_t j = 0; j < nn->layers[l + 1].neuronCnt; j++) {
            for (size_t i = 0; i < nn->layers[l].neuronCnt; i++)
                *gradient++ = nn->layers[l].neurons[i] * nn->layers[l + 1].deltas[j];

            *gradient++ = nn->layers[l + 1].deltas[j];  // biais
        }
}

void nn_gradient(const nn_network_t *nn, const double *inputs, const double *outputs, bool absolute,
    double *gradient)
{
    nn_backprop(nn, inputs, outputs, absolute);
    nn_do_gradient(nn, gradient);
}
