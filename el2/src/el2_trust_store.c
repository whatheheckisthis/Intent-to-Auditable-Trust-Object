#include "el2_trust_store.h"
#include "el2_nfc_validator.h"
#include <string.h>
#include <stdlib.h>
#include <mbedtls/ecp.h>
#include <mbedtls/sha256.h>

#define ENROLLMENT_GPIO_BASE 0x08010000UL

static el2_trust_store_t g_store;
static mbedtls_ecp_keypair g_el2_key;
static int g_key_init;

static int el2_rng(void *ctx, unsigned char *out, size_t len) {
    (void)ctx;
    for (size_t i = 0; i < len; i++) out[i] = (unsigned char)(rand() & 0xFF);
    return 0;
}

__attribute__((weak)) int el2_gpio_enrollment_asserted(void) {
    volatile uint32_t *gpio = (volatile uint32_t *)ENROLLMENT_GPIO_BASE;
    return (*gpio & 1u) ? 1 : 0;
}

__attribute__((weak)) int el2_tpm_pcr7_is_unextended(void) { return 1; }
__attribute__((weak)) int el2_tpm2_pcr_extend7(const uint8_t hash[32]) { (void)hash; return 0; }

static int ensure_el2_key(void) {
    if (g_key_init) return 0;
    mbedtls_ecp_keypair_init(&g_el2_key);
    int rc = mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, &g_el2_key, el2_rng, NULL);
    if (rc == 0) g_key_init = 1;
    return rc;
}

el2_err_t el2_enrollment_begin(void) {
    if (g_store.sealed) return EL2_ERR_TRUST_SEALED;
    if (!el2_tpm_pcr7_is_unextended() || !el2_gpio_enrollment_asserted()) return EL2_ERR_ENROLLMENT_MODE;
    if (ensure_el2_key() != 0) return EL2_ERR_ENROLLMENT_MODE;
    g_store.enrollment_mode = true;
    return EL2_OK;
}

el2_err_t el2_enrollment_receive_blob(const nfc_blob_t *blob) {
    if (!g_store.enrollment_mode || g_store.sealed) return EL2_ERR_ENROLLMENT_MODE;
    if (!blob) return EL2_ERR_BAD_SIGNATURE;
    ste_credential_t cred;
    nfc_blob_t copy = *blob;
    el2_err_t rc = el2_validate_nfc_blob(&copy, &cred);
    if (rc != EL2_OK) return rc;
    memcpy(g_store.se_root_hmac_key, cred.nonce, 32);
    memset(g_store.se_root_verify_key, 0, sizeof(g_store.se_root_verify_key));
    memcpy(g_store.se_root_verify_key, cred.spdm_session_id, 32);
    memcpy(g_store.se_root_verify_key + 32, cred.nonce, 32);
    g_store.cert_count = 1;
    g_store.spdm_ca_chain[0].len = 32;
    memcpy(g_store.spdm_ca_chain[0].der, cred.spdm_session_id, 32);
    return EL2_OK;
}

el2_err_t el2_enrollment_seal(void) {
    if (!g_store.enrollment_mode || g_store.sealed) return EL2_ERR_ENROLLMENT_MODE;
    uint8_t digest[32];
    mbedtls_sha256_context sha;
    mbedtls_sha256_init(&sha);
    mbedtls_sha256_starts(&sha, 0);
    mbedtls_sha256_update(&sha, (const unsigned char *)&g_store, sizeof(g_store));
    mbedtls_sha256_finish(&sha, digest);
    mbedtls_sha256_free(&sha);
    if (el2_tpm2_pcr_extend7(digest) != 0) return EL2_ERR_PCR_EXTEND;
    g_store.sealed = true;
    g_store.enrollment_mode = false;
    return EL2_OK;
}

const uint8_t *el2_get_el2_pubkey(size_t *out_len) {
    static uint8_t pub[65];
    size_t olen = 0;
    if (ensure_el2_key() != 0) return NULL;
    if (mbedtls_ecp_point_write_binary(&g_el2_key.grp, &g_el2_key.Q,
                                       MBEDTLS_ECP_PF_UNCOMPRESSED,
                                       &olen, pub, sizeof(pub)) != 0) {
        return NULL;
    }
    if (out_len) *out_len = olen;
    return pub;
}

const el2_trust_store_t *el2_get_trust_store(void) {
    return &g_store;
}
