#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "smmu_init.h"

volatile uint8_t iato_mock_smmu_mmio[IATO_SMMU_SIZE];

static uint32_t mmio32(uint32_t off) {
    return *(volatile uint32_t *)((uintptr_t)&iato_mock_smmu_mmio[0] + off);
}

int main(void) {
    int rc;
    uint32_t i;

    *(volatile uint32_t *)((uintptr_t)&iato_mock_smmu_mmio[0] + IATO_SMMU_REG_IDR0) = 1U;
    *(volatile uint32_t *)((uintptr_t)&iato_mock_smmu_mmio[0] + IATO_SMMU_REG_IDR1) = 1U;

    rc = iato_smmu_init();
    assert(rc == IATO_SMMU_OK);
    for (i = 0; i < IATO_SMMU_MAX_STREAMS; ++i) {
        assert(iato_smmu_read_ste_word0(i) == 0ULL);
    }
    assert((mmio32(IATO_SMMU_REG_CR0) & 1U) == 1U);

    rc = iato_smmu_write_ste(1U, 0x100000ULL, 0x200000ULL, 3U);
    assert(rc == IATO_SMMU_OK);
    assert((iato_smmu_read_ste_word0(1U) & 1ULL) == 1ULL);
    rc = iato_smmu_write_ste(64U, 0x100000ULL, 0x200000ULL, 3U);
    assert(rc == IATO_SMMU_ERR_RANGE);
    rc = iato_smmu_write_ste(2U, 0x100001ULL, 0x200000ULL, 3U);
    assert(rc == IATO_SMMU_ERR_RANGE);

    rc = iato_smmu_fault_ste(1U);
    assert(rc == IATO_SMMU_OK);
    assert(iato_smmu_read_ste_word0(1U) == 0ULL);
    for (i = 0; i < IATO_SMMU_MAX_STREAMS; ++i) {
        if (i != 1U) {
            assert(iato_smmu_read_ste_word0(i) == 0ULL);
        }
    }

    rc = iato_smmu_write_ste(3U, 0x400000ULL, 0x800000ULL, 1U);
    assert(rc == IATO_SMMU_OK);
    assert(iato_smmu_read_ste_word0(3U) != 0ULL);
    assert(mmio32(IATO_SMMU_REG_CMDQ_PROD) == 6U);

    rc = iato_smmu_init();
    assert(rc == IATO_SMMU_OK);
    assert(iato_smmu_read_ste_word0(3U) != 0ULL);

    puts("smmu_init_test: ok");
    return 0;
}
