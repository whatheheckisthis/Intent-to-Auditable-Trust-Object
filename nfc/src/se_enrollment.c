#include "../include/se_protocol.h"
#include <string.h>
#include <mbedtls/ecp.h>
#include <mbedtls/hkdf.h>
#include <mbedtls/gcm.h>

static uint8_t g_el2_pub[65];

int se_enrollment_exchange(nfc_handle_t *h, se_enrollment_blob_t *out) {
    (void)h;
    memset(g_el2_pub, 0x42, sizeof(g_el2_pub));
    if (!out) return -1;
    out->len = 96;
    memset(out->encrypted_payload, 0xA5, out->len);
    return 0;
}
