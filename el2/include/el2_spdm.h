#ifndef EL2_SPDM_H
#define EL2_SPDM_H

#include "el2_types.h"
#include <stddef.h>

el2_err_t el2_doe_map(stream_id_t stream_id, pa_t bar_pa, uint64_t bar_len);
el2_err_t el2_spdm_attest_device(stream_id_t stream_id);

uint32_t el2_doe_read(stream_id_t stream_id, uint32_t off);
void el2_doe_write(stream_id_t stream_id, uint32_t off, uint32_t val);

/* test hook */
typedef int (*el2_spdm_responder_t)(const uint8_t *req, size_t req_len, uint8_t *rsp, size_t *rsp_len);
void el2_spdm_set_responder(el2_spdm_responder_t responder);

#endif
