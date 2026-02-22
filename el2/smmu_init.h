#ifndef IATO_SMMU_INIT_H
#define IATO_SMMU_INIT_H

#include <stddef.h>
#include <stdint.h>

#define IATO_SMMU_BASE          0x09050000UL
#define IATO_SMMU_SIZE          0x00010000UL
#define IATO_SMMU_MAX_STREAMS   64U
#define IATO_STE_SIZE_BYTES     64U
#define IATO_STE_SIZE_WORDS     8U

#define IATO_SMMU_OK            0
#define IATO_SMMU_ERR_IDR       -1
#define IATO_SMMU_ERR_TIMEOUT   -2
#define IATO_SMMU_ERR_STRTAB    -3
#define IATO_SMMU_ERR_RANGE     -4

#define IATO_SMMU_REG_IDR0            0x0000U
#define IATO_SMMU_REG_IDR1            0x0004U
#define IATO_SMMU_REG_CR0             0x0020U
#define IATO_SMMU_REG_CR0ACK          0x0024U
#define IATO_SMMU_REG_STRTAB_BASE     0x0080U
#define IATO_SMMU_REG_STRTAB_BASE_CFG 0x0088U
#define IATO_SMMU_REG_CMDQ_BASE       0x0090U
#define IATO_SMMU_REG_CMDQ_PROD       0x0098U
#define IATO_SMMU_REG_CMDQ_CONS       0x009CU

int iato_smmu_init(void);
int iato_smmu_write_ste(uint32_t stream_id, uint64_t pa_base, uint64_t pa_limit, uint8_t permissions);
int iato_smmu_fault_ste(uint32_t stream_id);
uint64_t iato_smmu_read_ste_word0(uint32_t stream_id);

#endif
