#include <stdlib.h>
#include "nn.h"

nn_layer_t nn_layer_init(size_t inputLen, size_t outputLen, double *outputs)
{
    return (nn_layer_t){
        .inputs = calloc(inputLen, sizeof(double *)),
        .weights = calloc((inputLen + 1) * outputLen, sizeof(double *)),
        .outputs = outputs,
        .inputLen = inputLen,
        .outputLen = outputLen
    };
}

void nn_layer_destroy(nn_layer_t *nn)
{
    free(nn->inputs);
    free(nn->weights);
    *nn = (nn_layer_t){0};
}
