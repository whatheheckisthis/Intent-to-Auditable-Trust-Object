#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "smc_handler.h"
#include "smmu_init.h"

volatile uint8_t iato_mock_smmu_mmio[IATO_SMMU_SIZE];
uint8_t iato_mock_guest_ram[IATO_GUEST_RAM_SIZE];
uint64_t iato_mock_cntvct_now;

static int g_validator_rc;
static uint64_t g_pa_base = 0x100000ULL;
static uint64_t g_pa_limit = 0x200000ULL;
static uint8_t g_perms = 3U;

int iato_py_validate_credential(const uint8_t *cred, size_t cred_len, uint32_t stream_id,
                                uint64_t *out_pa_base, uint64_t *out_pa_limit, uint8_t *out_permissions) {
    (void)cred;
    (void)cred_len;
    (void)stream_id;
    *out_pa_base = g_pa_base;
    *out_pa_limit = g_pa_limit;
    *out_permissions = g_perms;
    return g_validator_rc;
}

static void fill_guest(uint64_t gpa, size_t n, uint8_t seed) {
    size_t i;
    uint64_t off = gpa - IATO_GUEST_RAM_BASE;
    for (i = 0; i < n; ++i) {
        iato_mock_guest_ram[off + i] = (uint8_t)(seed + (uint8_t)i);
    }
}

int main(void) {
    uint64_t regs[4] = {IATO_SMC_FUNCTION_ID, IATO_GUEST_RAM_BASE, IATO_CRED_MIN_BYTES, 0};
    const uint8_t *staging;
    size_t i;

    *(volatile uint32_t *)((uintptr_t)&iato_mock_smmu_mmio[0] + IATO_SMMU_REG_IDR0) = 1U;
    *(volatile uint32_t *)((uintptr_t)&iato_mock_smmu_mmio[0] + IATO_SMMU_REG_IDR1) = 1U;
    assert(iato_smmu_init() == IATO_SMMU_OK);
    assert(iato_smc_init() == 0);

    fill_guest(IATO_GUEST_RAM_BASE, IATO_CRED_MIN_BYTES, 0x10U);
    g_validator_rc = 0;
    g_pa_base = 0x100000ULL;
    g_pa_limit = 0x200000ULL;
    assert(iato_smc_handle(regs) == IATO_SMC_OK);

    regs[2] = 120U;
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_LENGTH);
    regs[2] = IATO_CRED_MIN_BYTES;

    regs[1] = IATO_GUEST_RAM_BASE + IATO_GUEST_RAM_SIZE;
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_GPA);
    regs[1] = IATO_GUEST_RAM_BASE;

    regs[3] = 64U;
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_STREAM);
    regs[3] = 0U;

    g_validator_rc = 1;
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_VALIDATION);
    staging = iato_smc_staging_ptr();
    for (i = 0; i < IATO_CRED_MAX_BYTES; ++i) {
        assert(staging[i] == 0U);
    }

    assert(iato_smc_init() == 0);
    g_validator_rc = 0;
    g_pa_base = 0x100001ULL;
    g_pa_limit = 0x200000ULL;
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_SMMU);

    assert(iato_smc_init() == 0);
    g_pa_base = 0x300000ULL;
    g_pa_limit = 0x400000ULL;
    fill_guest(IATO_GUEST_RAM_BASE, IATO_CRED_MIN_BYTES, 0x44U);
    assert(iato_smc_handle(regs) == IATO_SMC_OK);

    iato_mock_cntvct_now = 0ULL;
    for (i = 0; i < 4; ++i) {
        assert(iato_smc_handle(regs) == IATO_SMC_OK);
    }
    assert(iato_smc_handle(regs) == IATO_SMC_ERR_RATE);

    puts("smc_handler_test: ok");
    return 0;
}
