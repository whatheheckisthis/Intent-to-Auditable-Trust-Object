#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <bpf/bpf_endian.h>
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_FLOWS 1048576
#define SPIKE_THRESHOLD_PPS 50000
#define NS_PER_SEC 1000000000ULL

struct flow_key {
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
    __u64 first_seen_ns;
};

struct spike_state {
    __u64 window_start_ns;
    __u64 pkt_count;
};

struct audit_event {
    __u64 ts_ns;
    __u64 window_packets;
    __u64 pkt_delta_ns;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
    __u8 reason;
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

static __always_inline int parse_flow(void *data, void *data_end, struct flow_key *id)
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

    id->src_ip = iph->saddr;
    id->dst_ip = iph->daddr;
    id->proto = iph->protocol;
    id->src_port = 0;
    id->dst_port = 0;

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

static __always_inline void emit_event(const struct flow_key *id, __u64 now, __u64 win_pkts, __u64 delta, __u8 reason)
{
    struct audit_event *evt = bpf_ringbuf_reserve(&events, sizeof(*evt), 0);
    if (!evt)
        return;

    evt->ts_ns = now;
    evt->window_packets = win_pkts;
    evt->pkt_delta_ns = delta;
    evt->src_ip = id->src_ip;
    evt->dst_ip = id->dst_ip;
    evt->src_port = id->src_port;
    evt->dst_port = id->dst_port;
    evt->proto = id->proto;
    evt->reason = reason;
    bpf_ringbuf_submit(evt, 0);
}

SEC("xdp")
int netflow_filter(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct flow_key id = {};
    if (parse_flow(data, data_end, &id) < 0)
        return XDP_PASS;

    __u64 now = bpf_ktime_get_ns();
    __u64 pkt_len = (__u64)data_end - (__u64)data;

    struct flow_metrics *m = bpf_map_lookup_elem(&flow_map, &id);
    __u64 delta_ns = 0;
    if (m) {
        delta_ns = (m->last_seen_ns > 0 && now > m->last_seen_ns) ? (now - m->last_seen_ns) : 0;
        m->packets += 1;
        m->bytes += pkt_len;
        m->last_seen_ns = now;
    } else {
        struct flow_metrics init = {
            .packets = 1,
            .bytes = pkt_len,
            .last_seen_ns = now,
            .first_seen_ns = now,
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
        emit_event(&id, now, spike->pkt_count, delta_ns, 1);

    if (delta_ns > 0 && delta_ns < 100)
        emit_event(&id, now, spike->pkt_count, delta_ns, 2);

    return XDP_PASS;
}
