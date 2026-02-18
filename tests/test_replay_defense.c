#include "../el2/include/el2_binding_table.h"
#include "../el2/include/el2_nfc_validator.h"
#include "../el2/include/el2_trust_store.h"
#include "../el2/include/el2_expiry.h"
#include "../nfc/include/se_protocol.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void set_hmac_key(void) {
    el2_trust_store_t *ts = (el2_trust_store_t *)(uintptr_t)el2_get_trust_store();
    for (int i = 0; i < 32; i++) ts->se_root_hmac_key[i] = (uint8_t)(0x55 + i);
}

int main(void) {
    set_hmac_key();
    el2_set_mock_time_ns(1000);

    sha256_t sid = {7};
    sha256_t sid_bad = {9};
    sha256_t cert = {3};
    pa_range_t range = {.base = 0x1000, .limit = 0x1fff, .flags = 3};
    nfc_blob_t blob;
    ste_credential_t cred;

    assert(el2_binding_set_spdm(5, cert, sid) == EL2_OK);

    assert(se_issue_credential(NULL, 5, &range, 3, sid, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_ERR_NONCE_REPLAYED);

    assert(se_issue_credential(NULL, 5, &range, 3, sid, &blob) == 0);
    el2_set_mock_time_ns(400000000000ULL);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_ERR_EXPIRED);
    el2_set_mock_time_ns(1000);

    assert(se_issue_credential(NULL, 5, &range, 3, sid_bad, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_nfc(5, &cred) == EL2_ERR_SESSION_MISMATCH);

    assert(se_issue_credential(NULL, 8, &range, 3, sid_bad, &blob) == 0);
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_OK);
    assert(el2_binding_set_nfc(5, &cred) == EL2_ERR_SESSION_MISMATCH);

    assert(se_issue_credential(NULL, 5, &range, 3, sid, &blob) == 0);
    blob.mac[0] ^= 0xFF;
    assert(el2_validate_nfc_blob(&blob, &cred) == EL2_ERR_BAD_SIGNATURE);

    puts("test_replay_defense passed");
    return 0;
}
