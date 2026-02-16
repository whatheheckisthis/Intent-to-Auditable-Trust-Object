// SPDX-License-Identifier: GPL-2.0
#include <stddef.h>
#include <stdint.h>

/* Minimal BPF types to bypass missing asm/types.h */
typedef uint8_t __u8;
typedef uint16_t __u16;
typedef uint32_t __u32;
typedef uint64_t __u64;
typedef int32_t __s32;

#ifndef SEC
#define SEC(NAME) __attribute__((section(NAME), used))
#endif

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>
#include "snark_fpga_mmio.h"

#define WITNESS_WORDS FPGA_MMIO_WITNESS_WORDS
#define ETH_P_IP 0x0800
#define IPPROTO_UDP 17

struct xdp_md {
    __u32 data;
    __u32 data_end;
    __u32 data_meta;
    __u32 ingress_ifindex;
    __u32 rx_queue_index;
    __u32 egress_ifindex;
};

struct ethhdr {
    __u8 h_dest[6];
    __u8 h_source[6];
    __u16 h_proto;
};

struct iphdr {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    __u8 ihl:4;
    __u8 version:4;
#else
    __u8 version:4;
    __u8 ihl:4;
#endif
    __u8 tos;
    __u16 tot_len;
    __u16 id;
    __u16 frag_off;
    __u8 ttl;
    __u8 protocol;
    __u16 check;
    __u32 saddr;
    __u32 daddr;
};

struct udphdr {
    __u16 source;
    __u16 dest;
    __u16 len;
    __u16 check;
};

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
    __u32 verify_ok;
    __u32 witness[WITNESS_WORDS];
};

struct escalate_event {
    __u64 ts_ns;
    __u32 reason;
    __u32 epoch_id;
    __u32 flow_hash;
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

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(max_entries, 64);
    __type(key, __u32);
    __type(value, __u32);
} escalate_alerts SEC(".maps");

static __always_inline int parse_l3l4(void *data, void *data_end,
                                      struct osint_packet_meta *m)
{
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return -1;

    if (bpf_ntohs(eth->h_proto) != ETH_P_IP)
        return -1;

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end)
        return -1;

    __builtin_memset(m, 0, sizeof(*m));
    m->src_ip = iph->saddr;
    m->dst_ip = iph->daddr;
    m->proto = iph->protocol;

    if (iph->protocol == IPPROTO_UDP) {
        struct udphdr *uh = (void *)iph + (iph->ihl * 4);
        if ((void *)(uh + 1) > data_end)
            return -1;
        m->src_port = uh->source;
        m->dst_port = uh->dest;
    }

    m->flow_hash = (__u32)m->src_ip ^ (__u32)m->dst_ip ^
                   ((__u32)m->src_port << 16) ^ (__u32)m->dst_port;
    return 0;
}

static __always_inline void push_mmio_packet_meta(const struct osint_packet_meta *m)
{
    const __u32 keys[] = {
        FPGA_MMIO_REG_SRC_IP,
        FPGA_MMIO_REG_DST_IP,
        FPGA_MMIO_REG_PORTS,
        FPGA_MMIO_REG_PKT_META,
        FPGA_MMIO_REG_FLOW_HASH,
        FPGA_MMIO_REG_TS_LOW
    };
    const __u32 vals[] = {
        (__u32)m->src_ip,
        (__u32)m->dst_ip,
        ((__u32)m->src_port << 16) | (__u32)m->dst_port,
        ((__u32)m->pkt_len << 16) | (__u32)m->proto,
        m->flow_hash,
        (__u32)(m->ts_ns & 0xffffffff)
    };

#pragma clang loop unroll(full)
    for (int i = 0; i < 6; i++)
        bpf_map_update_elem(&fpga_mmio_shadow, &keys[i], &vals[i], BPF_ANY);
}

static __always_inline int pull_witness(struct xdp_witness_cb *cb)
{
    __u32 key = FPGA_MMIO_REG_WITNESS_STATUS;
    __u32 *status_ptr = bpf_map_lookup_elem(&fpga_mmio_shadow, &key);
    __u32 status;

    if (!status_ptr)
        return -1;

    status = *status_ptr;
    cb->epoch_id = status >> FPGA_WITNESS_EPOCH_SHIFT;

    if (status & FPGA_WITNESS_FAIL_MASK) {
        cb->valid = 1;
        cb->verify_ok = 0;
        return -2;
    }

    if ((status & FPGA_WITNESS_READY_MASK) == 0) {
        cb->valid = 0;
        cb->verify_ok = 0;
        return 1;
    }

    cb->valid = 1;
    cb->verify_ok = 1;
#pragma clang loop unroll(full)
    for (int i = 0; i < WITNESS_WORDS; i++) {
        __u32 w_key = FPGA_MMIO_WITNESS_BASE + i;
        __u32 *w_ptr = bpf_map_lookup_elem(&fpga_mmio_shadow, &w_key);
        cb->witness[i] = w_ptr ? *w_ptr : 0;
    }

    return 0;
}

static __always_inline int ESCALATE(struct xdp_md *ctx,
                                    const struct osint_packet_meta *m,
                                    __u32 epoch_id,
                                    __u32 reason)
{
    struct escalate_event ev = {
        .ts_ns = bpf_ktime_get_ns(),
        .reason = reason,
        .epoch_id = epoch_id,
        .flow_hash = m ? m->flow_hash : 0,
    };

    bpf_perf_event_output(ctx, &escalate_alerts, BPF_F_CURRENT_CPU, &ev, sizeof(ev));
    return XDP_ABORTED;
}

SEC("xdp")
int xdp_osint_snark_bridge(struct xdp_md *ctx)
{
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    if (bpf_xdp_adjust_meta(ctx, -(int)sizeof(struct xdp_witness_cb)) < 0)
        return XDP_PASS;

    void *data_meta = (void *)(long)ctx->data_meta;
    if (data_meta + sizeof(struct xdp_witness_cb) > data)
        return XDP_PASS;

    struct xdp_witness_cb *cb = data_meta;
    __builtin_memset(cb, 0, sizeof(*cb));

    __u32 idx = 0;
    struct osint_packet_meta *m = bpf_map_lookup_elem(&packet_meta_scratch, &idx);
    if (!m)
        return XDP_ABORTED;

    if (parse_l3l4(data, data_end, m) == 0) {
        struct osint_packet_meta snap = {};
        bpf_probe_read_kernel(&snap, sizeof(snap), m);
        snap.ts_ns = bpf_ktime_get_ns();
        snap.pkt_len = (__u16)(data_end - data);
        push_mmio_packet_meta(&snap);
    }

    int rc = pull_witness(cb);
    if (rc == -2)
        return ESCALATE(ctx, m, cb->epoch_id, 1);
    if (rc < 0)
        return ESCALATE(ctx, m, 0, 2);

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
