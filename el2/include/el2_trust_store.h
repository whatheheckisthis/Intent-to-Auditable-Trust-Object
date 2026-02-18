#ifndef EL2_TRUST_STORE_H
#define EL2_TRUST_STORE_H

#include "el2_types.h"
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    uint8_t se_root_hmac_key[32];
    uint8_t se_root_verify_key[64];
    struct {
        uint8_t der[1024];
        size_t len;
    } spdm_ca_chain[MAX_SPDM_CA_CERTS];
    size_t cert_count;
    bool sealed;
    bool enrollment_mode;
} el2_trust_store_t;

el2_err_t el2_enrollment_begin(void);
el2_err_t el2_enrollment_receive_blob(const nfc_blob_t *blob);
el2_err_t el2_enrollment_seal(void);
const uint8_t *el2_get_el2_pubkey(size_t *out_len);
const el2_trust_store_t *el2_get_trust_store(void);

/* dependency hooks */
int el2_gpio_enrollment_asserted(void);
int el2_tpm_pcr7_is_unextended(void);
int el2_tpm2_pcr_extend7(const uint8_t hash[32]);

#endif
