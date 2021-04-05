#include <stdlib.h>
#include "nn.h"

int main(void)
{
    double *outputs = calloc(2, sizeof(double *));
    nn_layer_t nn = nn_layer_init(3, 2, outputs);
    nn_layer_destroy(&nn);
}
