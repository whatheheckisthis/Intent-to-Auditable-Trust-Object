#ifndef OSINT_AUDIT_LOG_H
#define OSINT_AUDIT_LOG_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t masked_src_ipv4;
    uint32_t masked_dst_ipv4;
    uint64_t masked_first_seen_ns;
    uint64_t masked_last_seen_ns;
    uint8_t source_digest[32];
} osint_sanitized_payload_t;

int mask_v7_sve2(osint_sanitized_payload_t *payload);
int cbor_wrap_osint_data(const char *messy_json_like,
                         const osint_sanitized_payload_t *sanitized,
                         uint8_t **out,
                         size_t *out_len);

#endif
