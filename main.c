#include <stdlib.h>
#include "nn.h"

int main(void)
{
    nn_network_t nn = nn_network_init(3, (size_t[3]){4, 3, 2}, 1);
    nn_network_destroy(&nn);
}
