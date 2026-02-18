#ifndef NFC_SMMU_BRIDGE_H
#define NFC_SMMU_BRIDGE_H

#include <stdint.h>

int nfc_smmu_forward_blob(const void *payload, uint32_t len, uint32_t stream_id);
int nfc_smmu_query_status(uint32_t stream_id);

#endif
