#include <math.h>
#include <stdlib.h>
#include "nn.h"

double nn_linear(double x) { return x; }
double nn_relu(double x) { return x > 0 ? x : 0; }
double nn_sigmoid(double x) { return 1 / (1 + exp(-x)); }

nn_layer_t nn_layer_init(size_t inputCnt, nn_act_t act, size_t outputCnt,
    double *outputs)
{
    return (nn_layer_t){
        .inputs = calloc(inputCnt, sizeof(double *)),
        .weights = calloc((inputCnt + 1) * outputCnt, sizeof(double *)),
        .act = act,
        .outputs = outputs,
        .inputCnt = inputCnt,
        .outputCnt = outputCnt
    };
}

void nn_layer_destroy(nn_layer_t *layer)
{
    free(layer->inputs);
    free(layer->weights);
    *layer = (nn_layer_t){0};
}

nn_network_t nn_network_init(size_t layersCnt, size_t *intputCnts, nn_act_t *acts,
    size_t outputCnt)
{
    nn_network_t nn = {
        .layers = malloc(layersCnt * sizeof(nn_layer_t)),
        .layersCnt = layersCnt
    };

    // Allocate from layers[layersCnt - 1] to layersCnt[0]
    double *outputs = calloc(outputCnt, sizeof(double *));
    size_t layerOutputCnt = outputCnt;

    for (size_t i = layersCnt - 1; i < layersCnt; i--) {
        nn.layers[i] = nn_layer_init(intputCnts[i], acts[i], layerOutputCnt, outputs);
        outputs = nn.layers[i].inputs;
        layerOutputCnt = nn.layers[i].inputCnt;
    }

    return nn;
}

void nn_network_destroy(nn_network_t *nn)
{
    free(nn->layers[nn->layersCnt - 1].outputs);

    for (size_t i = 0; i < nn->layersCnt; i++)
        nn_layer_destroy(&nn->layers[i]);

    free(nn->layers);
    nn->layers = NULL;
    nn->layersCnt = 0;
}
