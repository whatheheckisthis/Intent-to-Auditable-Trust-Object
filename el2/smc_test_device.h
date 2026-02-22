#ifndef IATO_SMC_TEST_DEVICE_H
#define IATO_SMC_TEST_DEVICE_H

#include <stddef.h>
#include <stdint.h>

int iato_smc_test_device_enabled(void);
int iato_smc_test_device_write(const uint8_t *buf, size_t len, uint32_t *resp_code);
int iato_smc_test_device_read_ste_word0(uint32_t stream_id, uint64_t *word0);

#endif
