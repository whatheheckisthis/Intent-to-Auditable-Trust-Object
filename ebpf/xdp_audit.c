#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/ipv6.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_FLOWS 1048576
#define NS_PER_SEC 1000000000ULL

struct flow_key_v4 {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
    __u8 pad[3];
};

struct flow_key_v6 {
    __u8 src_ip[16];
    __u8 dst_ip[16];
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
    __u8 pad[3];
};

struct flow_metrics {
    __u64 packets;
    __u64 bytes;
    __u64 first_seen_ns;
    __u64 last_seen_ns;
};

struct l1_packet_stats {
    __u64 packets;
    __u64 bytes;
    __u64 window_start_ns;
    __u64 window_packets;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, MAX_FLOWS);
    __type(key, struct flow_key_v4);
    __type(value, struct flow_metrics);
} flow_v4_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, MAX_FLOWS);
    __type(key, struct flow_key_v6);
    __type(value, struct flow_metrics);
} flow_v6_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct l1_packet_stats);
} l1_stats SEC(".maps");

static __always_inline void update_l1_stats(__u64 now, __u64 pkt_len)
{
    __u32 idx = 0;
    struct l1_packet_stats *stats = bpf_map_lookup_elem(&l1_stats, &idx);
    if (!stats)
        return;

    stats->packets += 1;
    stats->bytes += pkt_len;

    if (stats->window_start_ns == 0 || now - stats->window_start_ns >= NS_PER_SEC) {
        stats->window_start_ns = now;
        stats->window_packets = 1;
    } else {
        stats->window_packets += 1;
    }
}

static __always_inline void update_metrics(void *map, const void *key, __u64 now, __u64 pkt_len)
{
    struct flow_metrics *metrics = bpf_map_lookup_elem(map, key);
    if (metrics) {
        metrics->packets += 1;
        metrics->bytes += pkt_len;
        metrics->last_seen_ns = now;
        return;
    }

    struct flow_metrics init = {
        .packets = 1,
        .bytes = pkt_len,
        .first_seen_ns = now,
        .last_seen_ns = now,
    };
    bpf_map_update_elem(map, key, &init, BPF_ANY);
}

static __always_inline int parse_l4_ports(void *l4, void *data_end, __u8 proto, __u16 *src_port, __u16 *dst_port)
{
    if (proto == IPPROTO_TCP) {
        struct tcphdr *th = l4;
        if ((void *)(th + 1) > data_end)
            return -1;
        *src_port = th->source;
        *dst_port = th->dest;
        return 0;
    }

    if (proto == IPPROTO_UDP) {
        struct udphdr *uh = l4;
        if ((void *)(uh + 1) > data_end)
            return -1;
        *src_port = uh->source;
        *dst_port = uh->dest;
        return 0;
    }

    return -1;
}

SEC("xdp")
int netflow_filter(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    __u64 now = bpf_ktime_get_ns();
    __u64 pkt_len = (__u64)data_end - (__u64)data;
    update_l1_stats(now, pkt_len);

    if (eth->h_proto == bpf_htons(ETH_P_IP)) {
        struct iphdr *iph = (void *)(eth + 1);
        if ((void *)(iph + 1) > data_end || iph->ihl < 5)
            return XDP_PASS;

        void *l4 = (void *)iph + (iph->ihl * 4);
        if (l4 > data_end)
            return XDP_PASS;

        struct flow_key_v4 key = {
            .src_ip = iph->saddr,
            .dst_ip = iph->daddr,
            .proto = iph->protocol,
        };

        if (parse_l4_ports(l4, data_end, iph->protocol, &key.src_port, &key.dst_port) == 0)
            update_metrics(&flow_v4_map, &key, now, pkt_len);

        return XDP_PASS;
    }

    if (eth->h_proto == bpf_htons(ETH_P_IPV6)) {
        struct ipv6hdr *ip6h = (void *)(eth + 1);
        if ((void *)(ip6h + 1) > data_end)
            return XDP_PASS;

        void *l4 = (void *)(ip6h + 1);
        if (l4 > data_end)
            return XDP_PASS;

        struct flow_key_v6 key = {
            .proto = ip6h->nexthdr,
        };
#pragma clang loop unroll(full)
        for (int i = 0; i < 16; i++) {
            key.src_ip[i] = ip6h->saddr.s6_addr[i];
            key.dst_ip[i] = ip6h->daddr.s6_addr[i];
        }

        if (parse_l4_ports(l4, data_end, ip6h->nexthdr, &key.src_port, &key.dst_port) == 0)
            update_metrics(&flow_v6_map, &key, now, pkt_len);
    }

    return XDP_PASS;
}
