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
#define SPIKE_THRESHOLD_PPS 50000

struct flow_key {
    __u8 family;
    __u8 proto;
    __u16 src_port;
    __u16 dst_port;
    __u16 pad;
    __u8 src_addr[16];
    __u8 dst_addr[16];
};

struct flow_metrics {
    __u64 packets;
    __u64 bytes;
    __u64 first_seen_ns;
    __u64 last_seen_ns;
};

struct spike_state {
    __u64 window_start_ns;
    __u64 pkt_count;
};

struct audit_event {
    __u64 ts_ns;
    __u64 window_packets;
    __u32 pkt_len;
    struct flow_key key;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, MAX_FLOWS);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
    __type(key, struct flow_key);
    __type(value, struct flow_metrics);
} flow_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct spike_state);
} spike_guard SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

static __always_inline int parse_ipv4(void *data, void *data_end, struct flow_key *id)
{
    struct iphdr *iph = data;
    if ((void *)(iph + 1) > data_end)
        return -1;
    if (iph->ihl < 5)
        return -1;

    void *l4 = (void *)iph + (iph->ihl * 4);
    if (l4 > data_end)
        return -1;

    id->family = AF_INET;
    id->proto = iph->protocol;
    __builtin_memcpy(id->src_addr, &iph->saddr, sizeof(iph->saddr));
    __builtin_memcpy(id->dst_addr, &iph->daddr, sizeof(iph->daddr));

    if (iph->protocol == IPPROTO_TCP) {
        struct tcphdr *th = l4;
        if ((void *)(th + 1) > data_end)
            return -1;
        id->src_port = th->source;
        id->dst_port = th->dest;
    } else if (iph->protocol == IPPROTO_UDP) {
        struct udphdr *uh = l4;
        if ((void *)(uh + 1) > data_end)
            return -1;
        id->src_port = uh->source;
        id->dst_port = uh->dest;
    } else {
        return -1;
    }

    return 0;
}

static __always_inline int parse_ipv6(void *data, void *data_end, struct flow_key *id)
{
    struct ipv6hdr *ip6h = data;
    if ((void *)(ip6h + 1) > data_end)
        return -1;

    void *l4 = ip6h + 1;
    if (l4 > data_end)
        return -1;

    id->family = AF_INET6;
    id->proto = ip6h->nexthdr;
    __builtin_memcpy(id->src_addr, &ip6h->saddr, sizeof(ip6h->saddr));
    __builtin_memcpy(id->dst_addr, &ip6h->daddr, sizeof(ip6h->daddr));

    if (ip6h->nexthdr == IPPROTO_TCP) {
        struct tcphdr *th = l4;
        if ((void *)(th + 1) > data_end)
            return -1;
        id->src_port = th->source;
        id->dst_port = th->dest;
    } else if (ip6h->nexthdr == IPPROTO_UDP) {
        struct udphdr *uh = l4;
        if ((void *)(uh + 1) > data_end)
            return -1;
        id->src_port = uh->source;
        id->dst_port = uh->dest;
    } else {
        return -1;
    }

    return 0;
}

static __always_inline int parse_flow(void *data, void *data_end, struct flow_key *id)
{
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return -1;

    __builtin_memset(id, 0, sizeof(*id));

    if (eth->h_proto == bpf_htons(ETH_P_IP))
        return parse_ipv4((void *)(eth + 1), data_end, id);

    if (eth->h_proto == bpf_htons(ETH_P_IPV6))
        return parse_ipv6((void *)(eth + 1), data_end, id);

    return -1;
}

static __always_inline void emit_event(const struct flow_key *id, __u64 now, __u64 win_pkts, __u32 pkt_len)
{
    struct audit_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
    if (!evt)
        return;

    evt->ts_ns = now;
    evt->window_packets = win_pkts;
    evt->pkt_len = pkt_len;
    evt->key = *id;
    bpf_ringbuf_submit(evt, 0);
}

SEC("xdp")
int netflow_filter(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct flow_key id = {};
    if (parse_flow(data, data_end, &id) != 0)
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
            .first_seen_ns = now,
            .last_seen_ns = now,
        };
        bpf_map_update_elem(&flow_map, &id, &init, BPF_ANY);
    }

    __u32 idx = 0;
    struct spike_state *spike = bpf_map_lookup_elem(&spike_guard, &idx);
    if (!spike)
        return XDP_PASS;

    if (spike->window_start_ns == 0 || now - spike->window_start_ns >= NS_PER_SEC) {
        spike->window_start_ns = now;
        spike->pkt_count = 1;
    } else {
        spike->pkt_count += 1;
    }

    if (spike->pkt_count >= SPIKE_THRESHOLD_PPS)
        emit_event(&id, now, spike->pkt_count, (__u32)pkt_len);

    return XDP_PASS;
}
