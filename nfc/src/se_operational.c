#include "../include/se_protocol.h"
#include "../../el2/include/el2_trust_store.h"
#include <string.h>
#include <stdlib.h>
#include <mbedtls/md.h>
#include <mbedtls/hkdf.h>
#include <mbedtls/gcm.h>

static void fill_random(uint8_t *p, size_t n) { for (size_t i=0;i<n;i++) p[i]=(uint8_t)(rand() & 0xff); }

int se_issue_credential(nfc_handle_t *h,
                        stream_id_t stream_id,
                        pa_range_t *pa_range,
                        uint32_t permissions,
                        sha256_t spdm_session_id,
                        nfc_blob_t *out) {
    (void)h;
    if (!out || !pa_range) return -1;

    ste_credential_t cred;
    memset(&cred, 0, sizeof(cred));
    cred.stream_id = stream_id;
    cred.pa_range = *pa_range;
    cred.permissions = permissions;
    cred.expiry_ns = 300000000000ULL;
    memcpy(cred.spdm_session_id, spdm_session_id, sizeof(sha256_t));
    fill_random(cred.nonce, sizeof(nonce_t));

    memset(out, 0, sizeof(*out));
    out->version = 1;
    out->ephemeral_pub[0] = 0x04;
    fill_random(&out->ephemeral_pub[1], 64);

    uint8_t ss[32], okm[44], iv[12], tag[16];
    for (size_t i=0;i<32;i++) ss[i] = out->ephemeral_pub[1+i] ^ out->ephemeral_pub[33+i];
    const mbedtls_md_info_t *md = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
    mbedtls_hkdf(md, NULL, 0, ss, sizeof(ss), (const unsigned char *)"IATO-V7-EL2-ECIES-v1", 20, okm, sizeof(okm));
    fill_random(iv, sizeof(iv));

    memcpy(out->ciphertext, iv, sizeof(iv));
    mbedtls_gcm_context gcm;
    mbedtls_gcm_init(&gcm);
    mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, okm, 256);
    mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, sizeof(cred), iv, sizeof(iv), NULL, 0,
                              (const unsigned char *)&cred, out->ciphertext + 28, 16, tag);
    memcpy(out->ciphertext + 12, tag, 16);
    mbedtls_gcm_free(&gcm);

    uint8_t input[sizeof(out->ciphertext)+sizeof(out->ephemeral_pub)];
    memcpy(input, out->ciphertext, sizeof(out->ciphertext));
    memcpy(input + sizeof(out->ciphertext), out->ephemeral_pub, sizeof(out->ephemeral_pub));
    const el2_trust_store_t *store = el2_get_trust_store();
    mbedtls_md_hmac(md, store->se_root_hmac_key, 32, input, sizeof(input), out->mac);
    return 0;
}

int se_get_challenge(nfc_handle_t *h, nonce_t *out) {
    (void)h;
    if (!out) return -1;
    fill_random(*out, sizeof(nonce_t));
    return 0;
}
