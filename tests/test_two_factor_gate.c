#include "../el2/include/el2_binding_table.h"
#include "../el2/include/el2_expiry.h"
#include "../el2/include/el2_trust_store.h"
#include "../el2/include/el2_nfc_validator.h"
#include "../el2/include/el2_smmu.h"
#include "../nfc/include/se_protocol.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void set_hmac_key(void) {
    el2_trust_store_t *ts = (el2_trust_store_t *)(uintptr_t)el2_get_trust_store();
    for (int i = 0; i < 32; i++) ts->se_root_hmac_key[i] = (uint8_t)(i + 1);
}

int main(void) {
    set_hmac_key();
    el2_set_mock_time_ns(1000);

    sha256_t sid_ok = {0xAA};
    sha256_t sid_bad = {0xBB};
    sha256_t cert = {0};
    pa_range_t range = {.base = 0x2000, .limit = 0x2fff, .flags = 3};
    ste_credential_t cred;
    nfc_blob_t blob;

    assert(el2_binding_set_spdm(3, cert, sid_ok) == EL2_OK);
    const binding_entry_t *e = el2_binding_get(3);
    assert(e && e->status == BINDING_SPDM_DONE);
    const smmu_ste_t *ste0 = el2_smmu_debug_get_ste(3);
    assert(ste0 && (*ste0)[0] == 0);

    assert(se_issue_credential(NULL, 4, &range, 3, sid_ok, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_nfc(4, &cred) == EL2_ERR_NO_BINDING);

    assert(se_issue_credential(NULL, 3, &range, 3, sid_bad, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_nfc(3, &cred) == EL2_ERR_SESSION_MISMATCH);

    assert(se_issue_credential(NULL, 3, &range, 3, sid_ok, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_nfc(3, &cred) == EL2_OK);
    e = el2_binding_get(3);
    assert(e && e->status == BINDING_ACTIVE);

    assert(se_issue_credential(NULL, 6, &range, 3, sid_ok, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_spdm(6, cert, sid_ok) == EL2_OK);
    assert(el2_binding_set_nfc(6, &cred) == EL2_OK);
    e = el2_binding_get(6);
    assert(e && e->status == BINDING_ACTIVE);

    puts("test_two_factor_gate passed");
    return 0;
}
