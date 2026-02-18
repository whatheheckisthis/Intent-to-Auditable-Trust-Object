#include "../el2/include/el2_trust_store.h"
#include <assert.h>
#include <stdio.h>

int el2_gpio_enrollment_asserted(void) { return 1; }
int el2_tpm_pcr7_is_unextended(void) { return 1; }
int el2_tpm2_pcr_extend7(const uint8_t hash[32]) { (void)hash; return 0; }

int main(void) {
    assert(el2_enrollment_begin() == EL2_OK);
    assert(el2_enrollment_seal() == EL2_OK);
    assert(el2_enrollment_begin() == EL2_ERR_TRUST_SEALED);
    size_t len = 0;
    assert(el2_get_el2_pubkey(&len) != NULL);
    assert(len > 0);
    puts("test_enrollment passed");
    return 0;
}
