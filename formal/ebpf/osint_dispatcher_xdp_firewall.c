// SPDX-License-Identifier: GPL-2.0
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>

/*
 * XDP firewall for OSINT Dispatcher REST logging endpoint.
 * Policy:
 *  - For packets to LOG_ENDPOINT_IP:LOG_ENDPOINT_PORT:
 *      allow only if HTTP payload contains SECURITY_TAG_HEADER.
 *  - For all other traffic: pass.
 */

#define LOG_ENDPOINT_IP __constant_htonl(0xC000020A) /* 192.0.2.10 */
#define LOG_ENDPOINT_PORT __constant_htons(8443)

static const char SECURITY_TAG_HEADER[] = "X-TEE-Security-Tag: TEE_ATTESTED";

static __always_inline int payload_contains_tag(void *payload, void *data_end)
{
    const char *needle = SECURITY_TAG_HEADER;
    const int needle_len = sizeof(SECURITY_TAG_HEADER) - 1;
    char *cursor = payload;

#pragma unroll
    for (int i = 0; i < 256; i++) {
        if ((void *)(cursor + needle_len) > data_end)
            return 0;

        int match = 1;
#pragma unroll
        for (int j = 0; j < 33; j++) {
            if (j >= needle_len)
                break;
            if (cursor[j] != needle[j]) {
                match = 0;
                break;
            }
        }

        if (match)
            return 1;

        if ((void *)(cursor + 1) > data_end)
            return 0;
        cursor++;
    }

    return 0;
}

SEC("xdp")
int xdp_osint_dispatcher_guard(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_ABORTED;

    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_ABORTED;

    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;

    struct tcphdr *tcp = (void *)ip + (ip->ihl * 4);
    if ((void *)(tcp + 1) > data_end)
        return XDP_ABORTED;

    if (ip->daddr != LOG_ENDPOINT_IP || tcp->dest != LOG_ENDPOINT_PORT)
        return XDP_PASS;

    void *payload = (void *)tcp + (tcp->doff * 4);
    if (payload >= data_end)
        return XDP_DROP;

    if (payload_contains_tag(payload, data_end))
        return XDP_PASS;

    return XDP_DROP;
}

char LICENSE[] SEC("license") = "GPL";
