// SPDX-License-Identifier: GPL-2.0
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

// NOTE:
// eBPF cannot safely dereference arbitrary MMIO addresses from XDP context.
// This bridge models MMIO as pinned array maps that are mirrored by a trusted
// user-space daemon performing the real FPGA BAR read/write operations.

#define FPGA_MMIO_WORDS 128
#define WITNESS_WORDS 8
#define ETH_P_IPV4 0x0800

struct osint_packet_meta {
    __u64 ts_ns;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u16 pkt_len;
    __u16 proto;
    __u32 flow_hash;
};

struct xdp_witness_cb {
    __u32 valid;
    __u32 epoch_id;
    __u32 witness[WITNESS_WORDS];
};

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, FPGA_MMIO_WORDS);
    __type(key, __u32);
    __type(value, __u32);
} fpga_mmio_shadow SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct osint_packet_meta);
} packet_meta_scratch SEC(".maps");

static __always_inline int parse_l3l4(void *data, void *data_end,
                                      struct osint_packet_meta *m) {
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) {
        return -1;
    }

    if (bpf_ntohs(eth->h_proto) != ETH_P_IPV4) {
        return -1;
    }

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end) {
        return -1;
    }

    __builtin_memset(m, 0, sizeof(*m));
    m->src_ip = iph->saddr;
    m->dst_ip = iph->daddr;
    m->proto = iph->protocol;

    if (iph->protocol == IPPROTO_UDP) {
        struct udphdr *uh = (void *)iph + (iph->ihl * 4);
        if ((void *)(uh + 1) > data_end) {
            return -1;
        }
        m->src_port = uh->source;
        m->dst_port = uh->dest;
    }

    m->flow_hash = (__u32)m->src_ip ^ (__u32)m->dst_ip ^
                   ((__u32)m->src_port << 16) ^ (__u32)m->dst_port;
    return 0;
}

static __always_inline void push_mmio_packet_meta(const struct osint_packet_meta *m) {
    // Layout example for FPGA command doorbell registers.
    const __u32 keys[] = {0, 1, 2, 3, 4, 5};
    const __u32 vals[] = {
        (__u32)m->src_ip,
        (__u32)m->dst_ip,
        ((__u32)m->src_port << 16) | (__u32)m->dst_port,
        ((__u32)m->pkt_len << 16) | (__u32)m->proto,
        m->flow_hash,
        (__u32)(m->ts_ns & 0xffffffff)
    };

#pragma clang loop unroll(full)
    for (int i = 0; i < 6; i++) {
        bpf_map_update_elem(&fpga_mmio_shadow, &keys[i], &vals[i], BPF_ANY);
    }
}

static __always_inline int pull_witness(struct xdp_witness_cb *cb) {
    __u32 key = 16;
    __u32 status = 0;
    __u32 *status_ptr = bpf_map_lookup_elem(&fpga_mmio_shadow, &key);
    if (!status_ptr) {
        return -1;
    }

    // status bit0 == proof ready
    status = *status_ptr;
    if ((status & 0x1) == 0) {
        cb->valid = 0;
        return 1;
    }

    cb->valid = 1;
    cb->epoch_id = status >> 1;
#pragma clang loop unroll(full)
    for (int i = 0; i < WITNESS_WORDS; i++) {
        __u32 w_key = 32 + i;
        __u32 *w_ptr = bpf_map_lookup_elem(&fpga_mmio_shadow, &w_key);
        cb->witness[i] = w_ptr ? *w_ptr : 0;
    }
    return 0;
}

SEC("xdp")
int xdp_osint_snark_bridge(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    // Reserve metadata room for witness in front of packet.
    if (bpf_xdp_adjust_meta(ctx, -(int)sizeof(struct xdp_witness_cb)) < 0) {
        return XDP_PASS;
    }

    void *data_meta = (void *)(long)ctx->data_meta;
    if (data_meta + sizeof(struct xdp_witness_cb) > data) {
        return XDP_PASS;
    }

    struct xdp_witness_cb *cb = data_meta;
    __builtin_memset(cb, 0, sizeof(*cb));

    __u32 idx = 0;
    struct osint_packet_meta *m = bpf_map_lookup_elem(&packet_meta_scratch, &idx);
    if (!m) {
        return XDP_ABORTED;
    }

    if (parse_l3l4(data, data_end, m) == 0) {
        // Required usage of bpf_probe_read{_kernel}: copy metadata snapshot before MMIO write.
        struct osint_packet_meta snap = {};
        bpf_probe_read_kernel(&snap, sizeof(snap), m);
        snap.ts_ns = bpf_ktime_get_ns();
        snap.pkt_len = (__u16)(data_end - data);
        push_mmio_packet_meta(&snap);
    }

    // Poll proof-ready status in MMIO mirror; witness is appended to metadata.
    pull_witness(cb);

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
