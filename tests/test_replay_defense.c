#include "../el2/include/el2_binding_table.h"
#include "../el2/include/el2_nfc_validator.h"
#include "../el2/include/el2_trust_store.h"
#include "../el2/include/el2_expiry.h"
#include "../nfc/include/se_protocol.h"
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void set_hmac_key(void) {
    el2_trust_store_t *ts = (el2_trust_store_t *)(uintptr_t)el2_get_trust_store();
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
    const uint64_t key_len = sizeof(ts->se_root_hmac_key);
    uint64_t i = 0;

    for (;;) {
        const svbool_t pg = svwhilelt_b8(i, key_len);
        if (!svptest_any(svptrue_b8(), pg)) {
            break;
        }

        const svuint8_t v = svindex_u8((uint8_t)(0x55u + i), 1);
        svst1_u8(pg, &ts->se_root_hmac_key[i], v);
        i = svincp_b8(i, pg);
    }
#else
    for (uint64_t i = 0; i < sizeof(ts->se_root_hmac_key); ++i) {
        ts->se_root_hmac_key[i] = (uint8_t)(0x55u + i);
    }
#endif

#if defined(__aarch64__)
    __asm__ volatile("dsb ish" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
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
