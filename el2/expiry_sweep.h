#ifndef IATO_EXPIRY_SWEEP_H
#define IATO_EXPIRY_SWEEP_H

#include <stdint.h>

#include "smmu_init.h"

typedef struct {
    uint32_t stream_id;
    uint64_t spdm_session_id[2];
    uint8_t status;
    uint64_t expiry_unix_s;
} iato_binding_entry_t;

#define IATO_STATUS_AWAITING_POLICY 0U
#define IATO_STATUS_ACTIVE 1U
#define IATO_STATUS_REVOKED 2U

extern iato_binding_entry_t iato_binding_table[IATO_SMMU_MAX_STREAMS];
int iato_expiry_sweep(uint64_t elapsed_ns);

#endif
