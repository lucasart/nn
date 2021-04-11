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
    nn_network_t nn = nn_network_init(4, (uint32_t[4]){4, 3, 2, 1},
        (uint32_t[3]){NN_LINEAR, NN_RELU, NN_SIGMOID});

    // Random weights and inputs
    uint64_t seed = 0;
    for (uint32_t i = 0; i < nn.weightCnt + nn.layers[0].neuronCnt; i++)
        nn.block[i] = prng_double(&seed);

    double gradient[nn.weightCnt];
    nn_gradient(&nn, NULL, (double[1]){0.5}, false, gradient);

    puts("network:");
    nn_network_print(&nn, "anwd");

    puts("\ngradient:");
    nn_array_print(nn.weightCnt, gradient);

    FILE *out = fopen("network.bin", "wb");
    nn_save(&nn, out);
    fclose(out);

    FILE *in = fopen("network.bin", "rb");
    nn_network_t nnVerify = nn_load(in);
    fclose(in);

    puts("\nnetwork reloaded:");
    nn_network_print(&nn, "aw");

    nn_network_destroy(&nnVerify);
    nn_network_destroy(&nn);
}
