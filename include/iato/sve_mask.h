#ifndef IATO_SVE_MASK_H
#define IATO_SVE_MASK_H

#include <stddef.h>
#include <stdint.h>

typedef struct sve_mask_state {
    size_t vl_bytes;
    uint64_t last_mask_epoch;
    uint32_t active;
} sve_mask_state;

#endif
