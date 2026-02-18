#ifndef EL2_SMMU_H
#define EL2_SMMU_H

#include "el2_types.h"

#ifndef SMMU_BASE
#define SMMU_BASE 0x09050000UL
#endif

el2_err_t el2_smmu_write_ste(stream_id_t stream_id, pa_range_t *pa_range, uint32_t permissions, el2_time_t expiry);
el2_err_t el2_smmu_fault_ste(stream_id_t stream_id);
const smmu_ste_t *el2_smmu_debug_get_ste(stream_id_t stream_id);
int el2_smmu_debug_fault_count(void);

#endif
