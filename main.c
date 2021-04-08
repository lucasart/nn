#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "nn.h"

// libc independant PRNG (SplitMix64)
static uint64_t prng_u64(uint64_t *x)
{
    *x += 0x9e3779b97f4a7c15;
    uint64_t z = (*x ^ (*x >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Uniform [-1,1] hacking the bit representation of a double
static double prng_double(uint64_t *x)
{
    const uint64_t r = prng_u64(x);
    const double v = (r >> 11) * 0x1.0p-53;
    return r & 1024 ? -v : v;
}

int main(void)
{
    nn_network_t nn = nn_network_init(4, (size_t[4]){4, 3, 2, 1},
        (nn_func_t[3]){nn_linear, nn_relu, nn_sigmoid},
        (nn_func_t[3]){nn_linear_derinv, nn_relu_derinv, nn_sigmoid_derinv});

    // Random weights and inputs
    uint64_t seed = 0;

    for (size_t i = 0; i < nn.weightCnt + nn.neuronCnt; i++)
        nn.block[i] = prng_double(&seed);

    nn_run(&nn);
    nn_network_print(&nn, "nw");
    nn_network_destroy(&nn);
}
