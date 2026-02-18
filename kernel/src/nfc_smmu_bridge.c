#include "nfc_smmu_bridge.h"
#include <stdlib.h>
#include <string.h>

#define EL2_SMC_NFC_BLOB     0xC2000001UL
#define EL2_SMC_QUERY_STATUS 0xC2000003UL

static uint32_t smmu_stream_id = 0;
static int enrollment_mode = 0;

__attribute__((weak)) uint64_t kernel_smc_call(uint64_t fn, uint64_t x1, uint64_t x2, uint64_t x3) {
    (void)fn; (void)x1; (void)x2; (void)x3;
    return 0;
}

int nfc_smmu_forward_blob(const void *payload, uint32_t len, uint32_t stream_id) {
    void *buf = aligned_alloc(64, len ? len : 64);
    if (!buf) return -1;
    memcpy(buf, payload, len);
    uint64_t rc = kernel_smc_call(EL2_SMC_NFC_BLOB, (uint64_t)(uintptr_t)buf, stream_id, 0);
    free(buf);
    (void)smmu_stream_id;
    (void)enrollment_mode;
    return (int)rc;
}

int nfc_smmu_query_status(uint32_t stream_id) {
    return (int)kernel_smc_call(EL2_SMC_QUERY_STATUS, stream_id, 0, 0);
}
