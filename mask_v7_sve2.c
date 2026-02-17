#include "osint_audit_log.h"

#include <stdint.h>

#if defined(__aarch64__)
extern void mask_v7_tuple_sve2(void);
#endif

int mask_v7_sve2(osint_sanitized_payload_t *payload) {
    if (payload == NULL) {
        return -1;
    }

    payload->masked_src_ipv4 &= 0xFFFFFF00u;
    payload->masked_dst_ipv4 &= 0xFFFFFF00u;

    const uint64_t bucket = 60000000000ULL;
    payload->masked_first_seen_ns = (payload->masked_first_seen_ns / bucket) * bucket;
    payload->masked_last_seen_ns = (payload->masked_last_seen_ns / bucket) * bucket;

#if defined(__aarch64__)
    mask_v7_tuple_sve2();
#endif
    return 0;
}
