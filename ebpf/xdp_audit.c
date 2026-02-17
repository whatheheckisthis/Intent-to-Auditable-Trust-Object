#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

char LICENSE[] SEC("license") = "GPL";

#define TVLA_ABS_T_THRESHOLD 4
#define TVLA_MIN_SAMPLES 32
#define EPSILON_NS 1

struct flow_key {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
};

struct flow_state {
    __u64 last_seen_ns;
};

struct tvla_state {
    __u64 sample_count;
    __u64 mean;
    __u64 m2;
};

struct audit_event {
    __u64 ts_ns;
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 proto;
    __u16 pkt_len;
    __u64 delta_ns;
    __u64 mean_ns;
    __u64 variance_ns;
    __u32 tvla_tscore;
    __u8 leak_flag;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 131072);
    __type(key, struct flow_key);
    __type(value, struct flow_state);
} flow_tracker SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_HASH);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct tvla_state);
} timing_stats SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
} audit_events SEC(".maps");

static __always_inline int parse_l4(void *data, void *data_end, struct flow_key *key, __u16 *pkt_len)
{
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) {
        return -1;
    }

    if (eth->h_proto != bpf_htons(ETH_P_IP)) {
        return -1;
    }

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) {
        return -1;
    }

    if (ip->ihl < 5) {
        return -1;
    }

    void *l4 = (void *)ip + (ip->ihl * 4);
    if (l4 > data_end) {
        return -1;
    }

    key->src_ip = ip->saddr;
    key->dst_ip = ip->daddr;
    key->proto = ip->protocol;
    key->src_port = 0;
    key->dst_port = 0;

    if (ip->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp = l4;
        if ((void *)(tcp + 1) > data_end) {
            return -1;
        }
        key->src_port = tcp->source;
        key->dst_port = tcp->dest;
    } else if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = l4;
        if ((void *)(udp + 1) > data_end) {
            return -1;
        }
        key->src_port = udp->source;
        key->dst_port = udp->dest;
    }

    *pkt_len = (__u16)((__u64)data_end - (__u64)data);
    return 0;
}

static __always_inline __u64 isqrt_u64(__u64 x)
{
    __u64 op = x;
    __u64 res = 0;
    __u64 one = 1ULL << 62;

    while (one > op) {
        one >>= 2;
    }

#pragma unroll
    for (int i = 0; i < 32; i++) {
        if (one == 0) {
            break;
        }
        if (op >= res + one) {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    return res;
}

SEC("xdp")
int xdp_audit(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct flow_key key = {};
    __u16 pkt_len = 0;

    if (parse_l4(data, data_end, &key, &pkt_len) < 0) {
        return XDP_PASS;
    }

    __u64 now = bpf_ktime_get_ns();
    __u64 delta_ns = 0;

    struct flow_state *state = bpf_map_lookup_elem(&flow_tracker, &key);
    if (state) {
        if (now > state->last_seen_ns) {
            delta_ns = now - state->last_seen_ns;
        }
        state->last_seen_ns = now;
    } else {
        struct flow_state fresh = {
            .last_seen_ns = now,
        };
        bpf_map_update_elem(&flow_tracker, &key, &fresh, BPF_ANY);
    }

    if (delta_ns == 0) {
        return XDP_PASS;
    }

    __u32 stats_key = 0;
    struct tvla_state *stats = bpf_map_lookup_elem(&timing_stats, &stats_key);
    if (!stats) {
        struct tvla_state init = {
            .sample_count = 1,
            .mean = delta_ns,
            .m2 = 0,
        };
        bpf_map_update_elem(&timing_stats, &stats_key, &init, BPF_ANY);
        return XDP_PASS;
    }

    __u64 old_mean = stats->mean;
    __u64 old_count = stats->sample_count;

    stats->sample_count = old_count + 1;
    __u64 new_mean = old_mean + (delta_ns - old_mean) / stats->sample_count;
    stats->mean = new_mean;

    __u64 diff_old = delta_ns > old_mean ? delta_ns - old_mean : old_mean - delta_ns;
    __u64 diff_new = delta_ns > new_mean ? delta_ns - new_mean : new_mean - delta_ns;
    stats->m2 += diff_old * diff_new;

    __u64 variance = 0;
    __u32 tscore = 0;
    __u8 leak_flag = 0;

    if (stats->sample_count > 1) {
        variance = stats->m2 / (stats->sample_count - 1);
    }

    if (stats->sample_count >= TVLA_MIN_SAMPLES && variance > 0) {
        __u64 sigma = isqrt_u64(variance + EPSILON_NS);
        __u64 abs_delta = delta_ns > stats->mean ? delta_ns - stats->mean : stats->mean - delta_ns;
        if (sigma > 0) {
            tscore = (__u32)(abs_delta / sigma);
        }
        if (tscore >= TVLA_ABS_T_THRESHOLD) {
            leak_flag = 1;
        }
    }

    if (leak_flag) {
        struct audit_event evt = {
            .ts_ns = now,
            .src_ip = key.src_ip,
            .dst_ip = key.dst_ip,
            .src_port = key.src_port,
            .dst_port = key.dst_port,
            .proto = key.proto,
            .pkt_len = pkt_len,
            .delta_ns = delta_ns,
            .mean_ns = stats->mean,
            .variance_ns = variance,
            .tvla_tscore = tscore,
            .leak_flag = leak_flag,
        };

        bpf_perf_event_output(ctx, &audit_events, BPF_F_CURRENT_CPU, &evt, sizeof(evt));
    }

    return XDP_PASS;
}
