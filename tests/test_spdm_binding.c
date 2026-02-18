#include "../el2/include/el2_spdm.h"
#include "../el2/include/el2_binding_table.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>

static int responder(const uint8_t *req, size_t req_len, uint8_t *rsp, size_t *rsp_len) {
    (void)req_len;
    uint8_t code = 0;
    switch (req[0]) {
        case 0x84: code = 0x04; break;
        case 0xE1: code = 0x61; break;
        case 0xE3: code = 0x63; break;
        case 0x81: code = 0x01; break;
        case 0x82: code = 0x02; break;
        case 0x83: code = 0x03; break;
        default: return -1;
    }
    rsp[0]=code; memset(rsp+1, 0x11, 31); *rsp_len=32; return 0;
}

int main(void) {
    el2_spdm_set_responder(responder);
    assert(el2_doe_map(7, 0x100000, 0x1000) == EL2_OK);
    assert(el2_spdm_attest_device(7) == EL2_OK);
    const binding_entry_t *e = el2_binding_get(7);
    assert(e && e->status == BINDING_SPDM_DONE);
    puts("test_spdm_binding passed");
    return 0;
}
