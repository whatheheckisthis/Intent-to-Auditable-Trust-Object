#include "el2_smmu.h"
#include <string.h>

static smmu_ste_t g_ste_table[MAX_BINDINGS];
static int g_fault_count;

static uint64_t el2_s2_alloc_table(pa_range_t *pa_range) {
    return (pa_range->base & ~0xFFFULL) | 0x3ULL;
}

static void cmd_sync(stream_id_t stream_id) { (void)stream_id; }

el2_err_t el2_smmu_write_ste(stream_id_t stream_id, pa_range_t *pa_range, uint32_t permissions, el2_time_t expiry) {
    if (stream_id >= MAX_BINDINGS) return EL2_ERR_SMMU_FAULT;
    smmu_ste_t ste = {0};
    ste[0] = (uint64_t)(STE_WORD0_V | STE_WORD0_CFG_TRANS);
    ste[1] = el2_s2_alloc_table(pa_range);
    ste[2] = pa_range->base;
    ste[3] = pa_range->limit;
    ste[4] = permissions;
    ste[5] = expiry;
    memcpy(g_ste_table[stream_id], ste, sizeof(ste));
    cmd_sync(stream_id);
    return EL2_OK;
}

el2_err_t el2_smmu_fault_ste(stream_id_t stream_id) {
    if (stream_id >= MAX_BINDINGS) return EL2_ERR_SMMU_FAULT;
    memset(g_ste_table[stream_id], 0, sizeof(smmu_ste_t));
    g_ste_table[stream_id][0] = STE_WORD0_FAULT;
    g_fault_count++;
    cmd_sync(stream_id);
    return EL2_OK;
}

const smmu_ste_t *el2_smmu_debug_get_ste(stream_id_t stream_id) {
    if (stream_id >= MAX_BINDINGS) return NULL;
    return &g_ste_table[stream_id];
}

int el2_smmu_debug_fault_count(void) { return g_fault_count; }
