#include "osint_audit_log.h"

#include <arpa/inet.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinycbor.h"

static int encode_osint_map(CborEncoder *map,
                            const char *messy_json_like,
                            const osint_sanitized_payload_t *sanitized) {
    CborError err = CborNoError;

    err = cbor_encode_text_stringz(map, "masked_first_seen_ns");
    if (err != CborNoError) return -1;
    err = cbor_encode_uint(map, sanitized->masked_first_seen_ns);
    if (err != CborNoError) return -1;

    err = cbor_encode_text_stringz(map, "masked_last_seen_ns");
    if (err != CborNoError) return -1;
    err = cbor_encode_uint(map, sanitized->masked_last_seen_ns);
    if (err != CborNoError) return -1;

    char src_txt[INET_ADDRSTRLEN] = {0};
    char dst_txt[INET_ADDRSTRLEN] = {0};
    uint32_t src_be = htonl(sanitized->masked_src_ipv4);
    uint32_t dst_be = htonl(sanitized->masked_dst_ipv4);
    inet_ntop(AF_INET, &src_be, src_txt, sizeof(src_txt));
    inet_ntop(AF_INET, &dst_be, dst_txt, sizeof(dst_txt));

    err = cbor_encode_text_stringz(map, "masked_src_ipv4");
    if (err != CborNoError) return -1;
    err = cbor_encode_text_stringz(map, src_txt);
    if (err != CborNoError) return -1;

    err = cbor_encode_text_stringz(map, "masked_dst_ipv4");
    if (err != CborNoError) return -1;
    err = cbor_encode_text_stringz(map, dst_txt);
    if (err != CborNoError) return -1;

    err = cbor_encode_text_stringz(map, "messy_payload");
    if (err != CborNoError) return -1;
    err = cbor_encode_text_stringz(map, messy_json_like);
    if (err != CborNoError) return -1;

    err = cbor_encode_text_stringz(map, "source_digest_sha256");
    if (err != CborNoError) return -1;
    err = cbor_encode_byte_string(map, sanitized->source_digest, sizeof(sanitized->source_digest));
    if (err != CborNoError) return -1;

    return 0;
}

int cbor_wrap_osint_data(const char *messy_json_like,
                         const osint_sanitized_payload_t *sanitized,
                         uint8_t **out,
                         size_t *out_len) {
    if (messy_json_like == NULL || sanitized == NULL || out == NULL || out_len == NULL) {
        return -1;
    }

    uint8_t *buf = calloc(1, 4096);
    if (buf == NULL) {
        return -1;
    }

    CborEncoder enc;
    CborEncoder map;
    cbor_encoder_init(&enc, buf, 4096, 0);
    if (cbor_encoder_create_map(&enc, &map, 6) != CborNoError) {
        free(buf);
        return -1;
    }
    if (encode_osint_map(&map, messy_json_like, sanitized) != 0) {
        free(buf);
        return -1;
    }
    if (cbor_encoder_close_container(&enc, &map) != CborNoError) {
        free(buf);
        return -1;
    }

    *out_len = cbor_encoder_get_buffer_size(&enc, buf);
    *out = buf;
    return 0;
}
