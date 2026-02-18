#include "el2_spdm.h"
#include "el2_binding_table.h"
#include <string.h>
#include <stdlib.h>
#include <mbedtls/sha256.h>

#define DOE_CTRL 0x08
#define DOE_STATUS 0x0C
#define DOE_WDATA 0x10
#define DOE_RDATA 0x14

enum { MSG_GET_VERSION=0x84, MSG_VERSION=0x04, MSG_GET_CAP=0xE1, MSG_CAP=0x61,
       MSG_NEG_ALG=0xE3, MSG_ALG=0x63, MSG_GET_DIGESTS=0x81, MSG_DIGESTS=0x01,
       MSG_GET_CERT=0x82, MSG_CERT=0x02, MSG_CHALLENGE=0x83, MSG_CHALLENGE_AUTH=0x03 };

typedef struct { stream_id_t sid; pa_t pa; uint64_t len; uint8_t valid; } doe_map_t;
static doe_map_t g_map[MAX_BINDINGS];
static el2_spdm_responder_t g_responder;

void el2_spdm_set_responder(el2_spdm_responder_t responder) { g_responder = responder; }

el2_err_t el2_doe_map(stream_id_t stream_id, pa_t bar_pa, uint64_t bar_len) {
    size_t idx = stream_id % MAX_BINDINGS;
    g_map[idx].sid = stream_id;
    g_map[idx].pa = bar_pa;
    g_map[idx].len = bar_len;
    g_map[idx].valid = 1;
    return EL2_OK;
}

uint32_t el2_doe_read(stream_id_t stream_id, uint32_t off) {
    (void)stream_id; (void)off;
    return 0;
}

void el2_doe_write(stream_id_t stream_id, uint32_t off, uint32_t val) {
    (void)stream_id; (void)off; (void)val;
}

static int exchange(const uint8_t *req, size_t req_len, uint8_t *rsp, size_t *rsp_len) {
    if (!g_responder) return -1;
    return g_responder(req, req_len, rsp, rsp_len);
}

el2_err_t el2_spdm_attest_device(stream_id_t stream_id) {
    uint8_t req[64], rsp[512];
    size_t req_len, rsp_len;
    uint8_t transcript[4096];
    size_t tlen = 0;

    uint8_t flow[][2] = {
        {MSG_GET_VERSION, MSG_VERSION}, {MSG_GET_CAP, MSG_CAP}, {MSG_NEG_ALG, MSG_ALG},
        {MSG_GET_DIGESTS, MSG_DIGESTS}, {MSG_GET_CERT, MSG_CERT}, {MSG_CHALLENGE, MSG_CHALLENGE_AUTH}
    };

    for (size_t i = 0; i < sizeof(flow)/2; i++) {
        req[0] = flow[i][0];
        req[1] = (uint8_t)i;
        if (flow[i][0] == MSG_CHALLENGE) {
            for (int n = 0; n < 32; n++) req[2+n] = (uint8_t)(rand() & 0xff);
            req_len = 34;
        } else {
            req_len = 2;
        }
        rsp_len = sizeof(rsp);
        if (exchange(req, req_len, rsp, &rsp_len) != 0) return EL2_ERR_SPDM_FAILED;
        if (rsp_len < 1 || rsp[0] != flow[i][1]) return EL2_ERR_SPDM_FAILED;
        memcpy(transcript + tlen, req, req_len); tlen += req_len;
        memcpy(transcript + tlen, rsp, rsp_len); tlen += rsp_len;
    }

    sha256_t session_id, cert_hash;
    mbedtls_sha256(transcript, tlen, session_id, 0);
    mbedtls_sha256(rsp, 32 < rsp_len ? 32 : rsp_len, cert_hash, 0);
    return el2_binding_set_spdm(stream_id, cert_hash, session_id);
}
