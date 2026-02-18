#include "el2_types.h"
#include "el2_nfc_validator.h"
#include "el2_binding_table.h"
#include "el2_spdm.h"
#include "el2_trust_store.h"

#define EL2_SMC_NFC_BLOB       0xC2000001UL
#define EL2_SMC_SPDM_ENUMERATE 0xC2000002UL
#define EL2_SMC_QUERY_STATUS   0xC2000003UL
#define EL2_SMC_ENROLLMENT     0xC2000004UL

typedef struct { uint64_t x0,x1,x2,x3; } smc_args_t;

uint64_t el2_smc_handler(smc_args_t *a) {
    if (!a) return (uint64_t)EL2_ERR_SPDM_FAILED;
    switch (a->x0) {
    case EL2_SMC_NFC_BLOB: {
        nfc_blob_t local;
        nfc_blob_t *in = (nfc_blob_t *)(uintptr_t)a->x1;
        local = *in;
        ste_credential_t cred;
        el2_err_t rc = el2_validate_nfc_blob(&local, &cred);
        if (rc != EL2_OK) return (uint64_t)rc;
        return (uint64_t)el2_binding_set_nfc((stream_id_t)a->x2, &cred);
    }
    case EL2_SMC_SPDM_ENUMERATE:
        if (el2_doe_map((stream_id_t)a->x1, (pa_t)a->x2, a->x3) != EL2_OK) return (uint64_t)EL2_ERR_SPDM_FAILED;
        return (uint64_t)el2_spdm_attest_device((stream_id_t)a->x1);
    case EL2_SMC_QUERY_STATUS: {
        const binding_entry_t *e = el2_binding_get((stream_id_t)a->x1);
        return e ? (uint64_t)e->status : (uint64_t)BINDING_EMPTY;
    }
    case EL2_SMC_ENROLLMENT:
        return (uint64_t)el2_enrollment_begin();
    default:
        return (uint64_t)EL2_ERR_SPDM_FAILED;
    }
}
