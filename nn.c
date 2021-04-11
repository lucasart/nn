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
#include <stdlib.h>
#include <string.h>
#include "nn.h"

typedef double(*nn_func_t)(double);

static double nn_linear(double x) { return x; }
static double nn_relu(double x) { return x > 0 ? x : 0; }
static double nn_sigmoid(double x) { return 1 / (1 + exp(-x)); }

static double nn_linear_derinv(double y) { (void)y; return 1; }
static double nn_relu_derinv(double y) { return y > 0 ? 1 : 0; }
static double nn_sigmoid_derinv(double y) { return y * (1 - y); }

static const struct { nn_func_t func, funcDerinv; char name[12]; uint32_t id; } actMap[] = {
    {nn_linear, nn_linear_derinv, "linear", NN_LINEAR},
    {nn_relu, nn_relu_derinv, "relu", NN_RELU},
    {nn_sigmoid, nn_sigmoid_derinv, "sigmoid", NN_SIGMOID}
};

void nn_array_print(size_t n, const double *array)
{
    for (size_t i = 0; i < n; i++)
        printf(i == n - 1 ? "%f\n": "%f,", array[i]);
}

void nn_layer_print(const nn_layer_t *layer, uint32_t nextLayerNeuronCnt, const char *what)
{
    if (strchr(what, 'a') && layer->deltas)
        printf("activation=%s\n", actMap[layer->actId].name);

    if (strchr(what, 'n')) {
        printf("neurons[%" PRIu32 "]=", layer->neuronCnt);
        nn_array_print(layer->neuronCnt, layer->neurons);
    }

    if (strchr(what, 'd') && layer->deltas) {
        printf("deltas[%" PRIu32 "]=", layer->neuronCnt);
        nn_array_print(layer->neuronCnt, layer->deltas);
    }

    if (strchr(what, 'w') && layer->weights) {
        printf("weights[%" PRIu32 "][%" PRIu32 "]=\n", nextLayerNeuronCnt, layer->neuronCnt + 1);
        const double *w = layer->weights;

        for (uint32_t j = 0; j < nextLayerNeuronCnt; j++) {
            printf("    %" PRIu32 ":", j);
            nn_array_print(layer->neuronCnt + 1, w);
            w += layer->neuronCnt + 1;
        }
    }
}

nn_network_t nn_network_init(uint32_t layerCnt, uint32_t *neuronCnts, uint32_t *actIds)
{
    nn_network_t nn = {
        .layers = malloc(layerCnt * sizeof(nn_layer_t)),
        .layerCnt = layerCnt
    };

    for (uint32_t i = 0; i < layerCnt; i++) {
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
    };

    for (uint32_t i = 1; i < layerCnt; i++)
        nn.layers[i] = (nn_layer_t){
            .neuronCnt = neuronCnts[i],
            .neurons = nn.layers[i - 1].neurons + neuronCnts[i - 1],
            .deltas = i > 1
                ? nn.layers[i - 1].deltas + neuronCnts[i - 1]
                : nn.block + nn.weightCnt + nn.neuronCnt,
            .actId = actIds[i - 1],
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
    for (uint32_t i = 0; i < nn->layerCnt; i++) {
        printf("layer #%" PRIu32 ":\n", i);
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

    for (uint32_t l = 1; l < nn->layerCnt; l++) {
        const double *w = nn->layers[l - 1].weights;

        for (uint32_t o = 0; o < nn->layers[l].neuronCnt; o++) {
            double sum = 0;

            // add the sum product
            for (uint32_t i = 0; i < nn->layers[l - 1].neuronCnt; i++)
                sum += nn->layers[l - 1].neurons[i] * *w++;

            // add the biais
            sum += *w++;

            // apply activation function and store neuron value
            nn->layers[l].neurons[o] = actMap[nn->layers[l].actId].func(sum);
        }
    }
}

static void nn_do_backprop(const nn_network_t *nn, const double *outputs, bool absolute)
{
    // Compute deltas on the output layer
    const nn_layer_t *ol = &nn->layers[nn->layerCnt - 1];

    for (uint32_t j = 0; j < ol->neuronCnt; j++) {
        const double diff = ol->neurons[j] - outputs[j];  // diff = o - t
        ol->deltas[j] = actMap[ol->actId].funcDerinv(ol->neurons[j]);
        ol->deltas[j] *= absolute
            ? (diff > 0 ? 1 : diff < 0 ? -1 : 0)  // d/do(|diff|) = sign(diff)
            : diff;  // d/do(.5*diff^2) = diff
    }

    // Compute deltas on inner layer l based on deltas from next layer l+1
    for (uint32_t l = nn->layerCnt - 2; l > 0; l--) {
        const nn_layer_t *cl = &nn->layers[l], *nl = &nn->layers[l + 1];  // current and next layer

        for (uint32_t j = 0; j < nl->neuronCnt; j++)
            for (uint32_t i = 0; i < cl->neuronCnt; i++)
                cl->deltas[i] += cl->weights[j * (cl->neuronCnt + 1) + i] * nl->deltas[j];

        for (uint32_t i = 0; i < cl->neuronCnt; i++)
            cl->deltas[i] *= actMap[cl->actId].funcDerinv(cl->neurons[i]);
    }
}

void nn_backprop(const nn_network_t *nn, const double *inputs, const double *outputs, bool absolute)
{
    nn_run(nn, inputs);
    nn_do_backprop(nn, outputs, absolute);
}

static void nn_do_gradient(const nn_network_t *nn, double *gradient)
{
    for (uint32_t l = 0; l < nn->layerCnt - 1; l++)
        for (uint32_t j = 0; j < nn->layers[l + 1].neuronCnt; j++) {
            for (uint32_t i = 0; i < nn->layers[l].neuronCnt; i++)
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

void nn_save(const nn_network_t *nn, FILE *out)
{
    fwrite(&nn->layerCnt, sizeof(nn->layerCnt), 1, out);

    for (uint32_t l = 0; l < nn->layerCnt; l++)
        fwrite(&nn->layers[l].neuronCnt, sizeof(nn->layers[l].neuronCnt), 1, out);

    for (uint32_t l = 1; l < nn->layerCnt; l++)
        fwrite(&nn->layers[l].actId, sizeof(nn->layers[l].actId), 1, out);

    fwrite(nn->block, sizeof(*nn->block), nn->weightCnt, out);
}

nn_network_t nn_load(FILE *in)
{
    // Read network structure
    uint32_t layerCnt = 0;
    fread(&layerCnt, sizeof(layerCnt), 1, in);

    uint32_t neuronCnts[layerCnt], actIds[layerCnt - 1];
    fread(neuronCnts, sizeof(*neuronCnts), layerCnt, in);
    fread(actIds, sizeof(*actIds), layerCnt - 1, in);

    // Create network for the required structure (zero initialized)
    nn_network_t nn = nn_network_init(layerCnt, neuronCnts, actIds);

    // Read weights from file
    fread(nn.block, sizeof(*nn.block), nn.weightCnt, in);
    return nn;
}
