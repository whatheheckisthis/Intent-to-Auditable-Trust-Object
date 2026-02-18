#include "../el2/include/el2_binding_table.h"
#include "../el2/include/el2_expiry.h"
#include "../el2/include/el2_smmu.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    sha256_t hash = {1};
    ste_credential_t cred;
    memset(&cred, 0, sizeof(cred));
    cred.stream_id = 9;
    cred.expiry_ns = 2000;
    memcpy(cred.spdm_session_id, hash, 32);
    cred.pa_range.base = 0x3000;
    cred.pa_range.limit = 0x3fff;
    cred.permissions = 3;

    assert(el2_binding_set_spdm(9, hash, hash) == EL2_OK);
    assert(el2_binding_set_nfc(9, &cred) == EL2_OK);
    const binding_entry_t *e = el2_binding_get(9);
    assert(e && e->status == BINDING_ACTIVE);

    el2_set_mock_time_ns(3000);
    int pre = el2_smmu_debug_fault_count();
    el2_expiry_handler();
    e = el2_binding_get(9);
    assert(e && e->status == BINDING_EXPIRED);
    assert(el2_smmu_debug_fault_count() == pre + 1);

    assert(el2_binding_set_spdm(9, hash, hash) == EL2_OK);
    cred.expiry_ns = 6000;
    assert(el2_binding_set_nfc(9, &cred) == EL2_OK);
    e = el2_binding_get(9);
    assert(e && e->status == BINDING_ACTIVE);

    puts("test_expiry_sweep passed");
    return 0;
}
