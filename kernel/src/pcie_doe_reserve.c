#include <stdint.h>

#define EL2_SMC_SPDM_ENUMERATE 0xC2000002UL
#define PCI_EXT_CAP_ID_DOE 0x2E

__attribute__((weak)) uint64_t kernel_smc_call(uint64_t fn, uint64_t x1, uint64_t x2, uint64_t x3);

int pcie_doe_probe(uint32_t stream_id, uint64_t bar_pa, uint64_t bar_len, int has_doe_cap) {
    if (!has_doe_cap) return -1;
    (void)PCI_EXT_CAP_ID_DOE;
    return (int)kernel_smc_call(EL2_SMC_SPDM_ENUMERATE, stream_id, bar_pa, bar_len);
}
