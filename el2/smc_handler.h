#ifndef IATO_SMC_HANDLER_H
#define IATO_SMC_HANDLER_H

#include <stddef.h>
#include <stdint.h>

#include "smmu_init.h"

#define IATO_SMC_FUNCTION_ID    0x82000001UL
#define IATO_SMC_OK             0x00000000UL
#define IATO_SMC_ERR_LENGTH     0x00000001UL
#define IATO_SMC_ERR_GPA        0x00000002UL
#define IATO_SMC_ERR_STREAM     0x00000003UL
#define IATO_SMC_ERR_VALIDATION 0x00000004UL
#define IATO_SMC_ERR_SMMU       0x00000005UL
#define IATO_SMC_ERR_RATE       0x00000006UL
#define IATO_SMC_ERR_INTERNAL   0xFFFFFFFFUL

#define IATO_CRED_MAX_BYTES     149U
#define IATO_CRED_MIN_BYTES     109U
#define IATO_GUEST_RAM_BASE     0x40000000ULL
#define IATO_GUEST_RAM_SIZE     0x04000000ULL

int iato_smc_init(void);
uint64_t iato_smc_handle(uint64_t regs[4]);
int iato_smc_copy_from_guest(void *dst, uint64_t guest_pa, size_t length);

#ifdef IATO_HOST_TEST
const uint8_t *iato_smc_staging_ptr(void);
#endif

#endif
