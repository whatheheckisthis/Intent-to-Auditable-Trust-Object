#ifndef SE_PROTOCOL_H
#define SE_PROTOCOL_H

#include "../../el2/include/el2_types.h"
#include <stdint.h>

typedef struct { int dummy; } nfc_handle_t;

typedef struct {
    uint8_t encrypted_payload[512];
    uint32_t len;
} se_enrollment_blob_t;

#define APDU_CLA_SECURE 0x80
#define APDU_INS_ENROLL_BEGIN 0xE0
#define APDU_INS_ENROLL_DELIVER_TRUST_ANCHORS 0xE1
#define APDU_INS_ISSUE_CREDENTIAL 0xA0
#define APDU_INS_GET_CHALLENGE_NONCE 0xA1

int se_enrollment_exchange(nfc_handle_t *h, se_enrollment_blob_t *out);
int se_issue_credential(nfc_handle_t *h, stream_id_t stream_id, pa_range_t *pa_range,
                        uint32_t permissions, sha256_t spdm_session_id, nfc_blob_t *out);
int se_get_challenge(nfc_handle_t *h, nonce_t *out);

#endif
