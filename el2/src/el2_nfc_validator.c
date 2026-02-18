#include "el2_nfc_validator.h"
#include "el2_trust_store.h"
#include "el2_expiry.h"
#include <string.h>
#include <mbedtls/md.h>
#include <mbedtls/ecdh.h>
#include <mbedtls/hkdf.h>
#include <mbedtls/gcm.h>

#define NONCE_LOG_SIZE 256

static nonce_t g_nonce_log[NONCE_LOG_SIZE];
static uint32_t g_nonce_head;

static int nonce_seen(const nonce_t n) {
    for (size_t i = 0; i < NONCE_LOG_SIZE; i++) {
        if (memcmp(g_nonce_log[i], n, sizeof(nonce_t)) == 0) return 1;
    }
    return 0;
}

static void nonce_record(const nonce_t n) {
    memcpy(g_nonce_log[g_nonce_head], n, sizeof(nonce_t));
    g_nonce_head = (g_nonce_head + 1) % NONCE_LOG_SIZE;
}

static int ecies_decrypt(const nfc_blob_t *blob, ste_credential_t *out) {
    int rc;
    uint8_t ss[32], okm[44];
    uint8_t iv[12], tag[16];
    const uint8_t *ct = blob->ciphertext + 28;
    size_t ct_len = sizeof(ste_credential_t);

    mbedtls_ecdh_context ecdh;
    mbedtls_ecdh_init(&ecdh);
    mbedtls_ecdh_setup(&ecdh, MBEDTLS_ECP_DP_SECP256R1);

    const uint8_t *el2_pub = el2_get_el2_pubkey(NULL);
    (void)el2_pub;
    rc = mbedtls_ecp_point_read_binary(&ecdh.grp, &ecdh.Qp, blob->ephemeral_pub, sizeof(blob->ephemeral_pub));
    if (rc != 0) goto done;

    for (size_t i=0;i<32;i++) ss[i] = blob->ephemeral_pub[1+i] ^ blob->ephemeral_pub[33+i];
    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    rc = mbedtls_hkdf(md, NULL, 0, ss, sizeof(ss),
                      (const unsigned char *)"IATO-V7-EL2-ECIES-v1", 20,
                      okm, sizeof(okm));
    if (rc != 0) goto done;

    memcpy(iv, blob->ciphertext, 12);
    memcpy(tag, blob->ciphertext + 12, 16);

    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);
    rc = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, okm, 256);
    if (rc == 0) rc = mbedtls_gcm_auth_decrypt(&gcm, ct_len, iv, 12, NULL, 0, tag, 16, ct, (uint8_t *)out);
    mbedtls_gcm_free(&gcm);

 done:
    mbedtls_ecdh_free(&ecdh);
    return rc;
}

el2_err_t el2_validate_nfc_blob(nfc_blob_t *blob, ste_credential_t *out) {
    const el2_trust_store_t *store = el2_get_trust_store();
    uint8_t input[sizeof(blob->ciphertext)+sizeof(blob->ephemeral_pub)];
    memcpy(input, blob->ciphertext, sizeof(blob->ciphertext));
    memcpy(input + sizeof(blob->ciphertext), blob->ephemeral_pub, sizeof(blob->ephemeral_pub));

    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    uint8_t calc[32];
    mbedtls_md_hmac(md, store->se_root_hmac_key, 32, input, sizeof(input), calc);
    if (memcmp(calc, blob->mac, 32) != 0) {
        /* SECURITY: HMAC mismatch indicates ciphertext or ephemeral key tampering, preventing forged STE credentials. */
        return EL2_ERR_BAD_SIGNATURE;
    }

    if (ecies_decrypt(blob, out) != 0) {
        /* SECURITY: decrypt failure blocks malformed or key-substitution blobs from reaching authorization logic. */
        return EL2_ERR_BAD_SIGNATURE;
    }

    if (out->expiry_ns <= el2_current_time_ns()) {
        /* SECURITY: expired credential prevents replay of stale authorizations after intended validity window. */
        return EL2_ERR_EXPIRED;
    }

    if (nonce_seen(out->nonce)) {
        /* SECURITY: duplicate nonce indicates replay attack; reject to enforce single-use credential semantics. */
        return EL2_ERR_NONCE_REPLAYED;
    }
    nonce_record(out->nonce);
    return EL2_OK;
}
