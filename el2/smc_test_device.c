#include "smc_test_device.h"

#include "smc_handler.h"
#include "smmu_init.h"

int iato_smc_test_device_enabled(void) {
#ifdef IATO_TEST_MODE
    return 1;
#else
    return 0;
#endif
}

int iato_smc_test_device_write(const uint8_t *buf, size_t len, uint32_t *resp_code) {
    uint64_t regs[4] = { IATO_SMC_FUNCTION_ID, 0ULL, 149ULL, 0ULL };
    (void)buf;
    if ((len != 153U) || (resp_code == (uint32_t *)0)) {
        return -1;
    }
    *resp_code = (uint32_t)iato_smc_handle(regs);
    return 0;
}

int iato_smc_test_device_read_ste_word0(uint32_t stream_id, uint64_t *word0) {
    if (word0 == (uint64_t *)0) {
        return -1;
    }
    *word0 = iato_smmu_read_ste_word0(stream_id);
    return 0;
}
