/* SPDX-License-Identifier: Apache-2.0 */

#include "iato/rme_mgmt.h"
#include "iato/sve_mask.h"
#include "pkcs11_signer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef unsigned long C_Session;

enum {
    IATO_HSM_OK = 0,
    IATO_HSM_ERR_ARG = -1,
    IATO_HSM_ERR_PKCS11 = -2,
    IATO_HSM_ERR_ASM = -3,
    IATO_HSM_ERR_SIGN = -4,
};

struct iato_hsm_bridge {
    rme_realm_t realm;
    sve_mask_state mask_state;
    C_Session session;
    volatile uint8_t *mana_mailbox_mmio;
};

extern uint64_t iato_sve2_get_vl(void);
extern void iato_sve2_zero_zregs(void);
extern uint64_t iato_sve2_mask_and_sign(const void *input_ptr,
                                        const void *mask_ptr,
                                        volatile void *mailbox_ptr,
                                        void *digest_out);

static void iato_secure_zero(volatile uint8_t *buf, size_t len) {
    if (buf == NULL) {
        return;
    }
    while (len-- > 0) {
        *buf++ = 0;
    }
}

int iato_hsm_bridge_init(struct iato_hsm_bridge *bridge,
                         volatile void *mana_mailbox_mmio,
                         rme_realm_t realm,
                         C_Session session) {
    if (bridge == NULL || mana_mailbox_mmio == NULL) {
        return IATO_HSM_ERR_ARG;
    }

    memset(bridge, 0, sizeof(*bridge));
    bridge->realm = realm;
    bridge->session = session;
    bridge->mana_mailbox_mmio = (volatile uint8_t *)mana_mailbox_mmio;
    bridge->mask_state.vl_bytes = (size_t)iato_sve2_get_vl();
    bridge->mask_state.active = 1U;

    if (pkcs11_initialize_from_env() != 0) {
        return IATO_HSM_ERR_PKCS11;
    }

    return IATO_HSM_OK;
}

int iato_hsm_bridge_mask_and_csign(struct iato_hsm_bridge *bridge,
                                   const uint8_t *input,
                                   const uint8_t *mask,
                                   uint8_t digest_out[32]) {
    if (bridge == NULL || input == NULL || mask == NULL || digest_out == NULL) {
        return IATO_HSM_ERR_ARG;
    }

    const uint64_t rc = iato_sve2_mask_and_sign(input,
                                                mask,
                                                (volatile void *)bridge->mana_mailbox_mmio,
                                                digest_out);
    if (rc != 0U) {
        return IATO_HSM_ERR_ASM;
    }

    /* C_Sign event: sign the Merkle leaf while blinded payload stays MMIO-side. */
    if (sign_log_hsm(digest_out, 32U) != 0) {
        return IATO_HSM_ERR_SIGN;
    }

    return IATO_HSM_OK;
}

void iato_hsm_bridge_close(struct iato_hsm_bridge *bridge) {
    if (bridge == NULL) {
        return;
    }

    iato_sve2_zero_zregs();
    bridge->mask_state.active = 0U;
    pkcs11_cleanup();

    iato_secure_zero((volatile uint8_t *)bridge, sizeof(*bridge));
}
