#include <stdint.h>

uint64_t branchless_reduce(uint64_t t, uint64_t M) {
    uint64_t mask = (uint64_t)0 - (uint64_t)(t >= M);
    return t - (M & mask);
}
