#include <math.h>
#include <stdint.h>
#include <string.h>

#define TELEMETRY_DIM 4
#define STABILITY_STEPS 100000000ULL

typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t pkt_len;
    uint8_t proto;
    uint32_t flow_hash;
    uint64_t ts_ns;
} telemetry_packet_t;

typedef struct {
    double x[TELEMETRY_DIM];
} lyapunov_state_t;

typedef struct {
    uint8_t bytes[32];
} witness256_t;

static inline double clamp(double v, double lo, double hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

static lyapunov_state_t map_packet_to_manifold(const telemetry_packet_t *pkt)
{
    lyapunov_state_t s;
    s.x[0] = (double)pkt->flow_hash / 4294967295.0;
    s.x[1] = (double)pkt->pkt_len / 65535.0;
    s.x[2] = (double)pkt->proto / 255.0;
    s.x[3] = (double)(pkt->ts_ns & 0xffffffffULL) / 4294967295.0;
    return s;
}

static double lyapunov_energy(const lyapunov_state_t *s)
{
    // Positive definite energy V(x) = x^T P x, diagonal P.
    const double p[TELEMETRY_DIM] = {3.5, 2.0, 1.25, 2.75};
    double v = 0.0;
    for (int i = 0; i < TELEMETRY_DIM; ++i) {
        v += p[i] * s->x[i] * s->x[i];
    }
    return v;
}

witness256_t telemetry_to_stability_witness(const telemetry_packet_t *pkt,
                                            uint64_t op_count)
{
    witness256_t out;
    memset(&out, 0, sizeof(out));

    lyapunov_state_t s = map_packet_to_manifold(pkt);
    double v0 = lyapunov_energy(&s);

    // Contractive dynamics approximation for recursive fold stability.
    const double alpha = 1.0e-8;
    const uint64_t steps = (op_count == 0) ? STABILITY_STEPS : op_count;
    const double decay = exp(-alpha * (double)steps);
    const double vN = v0 * decay;
    const double margin = clamp(v0 - vN, 0.0, 1.0);

    // Serialize scalar invariants into 256-bit witness words.
    uint64_t *w = (uint64_t *)out.bytes;
    w[0] = (uint64_t)(v0 * 1.0e12);
    w[1] = (uint64_t)(vN * 1.0e12);
    w[2] = (uint64_t)(margin * 1.0e12);
    w[3] = steps;

    return out;
}
