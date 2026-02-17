#include "vmlinux.h"
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "GPL";

struct flow_id {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
    __u8 pad[3];
};

struct flow_metrics {
    __u64 packets;
    __u64 bytes;
    __u64 last_seen_ns;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 262144);
    __type(key, struct flow_id);
    __type(value, struct flow_metrics);
} flow_map SEC(".maps");

static __always_inline int parse_flow(void *data, void *data_end, struct flow_id *id)
{
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return -1;

    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return -1;

    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end)
        return -1;

    if (iph->ihl < 5)
        return -1;

    void *l4 = (void *)iph + (iph->ihl * 4);
    if (l4 > data_end)
        return -1;

    if (iph->protocol != IPPROTO_TCP && iph->protocol != IPPROTO_UDP)
        return -1;

    id->src_ip = iph->saddr;
    id->dst_ip = iph->daddr;
    id->proto = iph->protocol;

    if (iph->protocol == IPPROTO_TCP) {
        struct tcphdr *th = l4;
        if ((void *)(th + 1) > data_end)
            return -1;
        id->src_port = th->source;
        id->dst_port = th->dest;
    } else {
        struct udphdr *uh = l4;
        if ((void *)(uh + 1) > data_end)
            return -1;
        id->src_port = uh->source;
        id->dst_port = uh->dest;
    }

    return 0;
}

SEC("xdp")
int netflow_filter(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct flow_id id = {};
    if (parse_flow(data, data_end, &id) < 0)
        return XDP_PASS;

    __u64 now = bpf_ktime_get_ns();
    __u64 pkt_len = (__u64)data_end - (__u64)data;

    struct flow_metrics *m = bpf_map_lookup_elem(&flow_map, &id);
    if (m) {
        m->packets += 1;
        m->bytes += pkt_len;
        m->last_seen_ns = now;
    } else {
        struct flow_metrics init = {
            .packets = 1,
            .bytes = pkt_len,
            .last_seen_ns = now,
        };
        bpf_map_update_elem(&flow_map, &id, &init, BPF_ANY);
    }

    return XDP_PASS;
}
