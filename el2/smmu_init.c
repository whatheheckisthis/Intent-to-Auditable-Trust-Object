#include "smmu_init.h"

#define IATO_TIMEOUT_ITERS 1000U
#define IATO_CMDQ_ENTRIES 256U

#ifdef IATO_MOCK_MMIO
extern volatile uint8_t iato_mock_smmu_mmio[IATO_SMMU_SIZE];
#define IATO_MMIO_BASE_PTR ((uintptr_t)&iato_mock_smmu_mmio[0])
#else
#define IATO_MMIO_BASE_PTR ((uintptr_t)IATO_SMMU_BASE)
#endif

#if defined(__aarch64__)
#define IATO_DSB_SY() __asm__ volatile("dsb sy" ::: "memory")
#define IATO_ISB() __asm__ volatile("isb" ::: "memory")
#else
#define IATO_DSB_SY() do {} while (0)
#define IATO_ISB() do {} while (0)
#endif

static volatile uint64_t iato_smmu_strtab[IATO_SMMU_MAX_STREAMS][IATO_STE_SIZE_WORDS]
    __attribute__((aligned(IATO_STE_SIZE_BYTES)));

static volatile uint64_t iato_smmu_cmdq[IATO_CMDQ_ENTRIES][2]
    __attribute__((aligned(128)));

static int iato_smmu_initialized;

static inline volatile uint32_t *reg32(uint32_t off) {
    return (volatile uint32_t *)(IATO_MMIO_BASE_PTR + (uintptr_t)off);
}

static inline volatile uint64_t *reg64(uint32_t off) {
    return (volatile uint64_t *)(IATO_MMIO_BASE_PTR + (uintptr_t)off);
}

static uint32_t ilog2_u64(uint64_t v) {
    uint32_t r = 0U;
    while (v > 1U) {
        v >>= 1U;
        r++;
    }
    return r;
}

static int poll_cr0ack(uint32_t target) {
    uint32_t i;
    for (i = 0U; i < IATO_TIMEOUT_ITERS; ++i) {
        if ((*reg32(IATO_SMMU_REG_CR0ACK) & 0x1U) == (target & 0x1U)) {
            return IATO_SMMU_OK;
        }
#ifdef IATO_MOCK_MMIO
        *reg32(IATO_SMMU_REG_CR0ACK) = target;
#endif
    }
    return IATO_SMMU_ERR_TIMEOUT;
}

static int cmdq_issue(uint64_t cmd0, uint64_t cmd1, int wait_sync) {
    uint32_t prod = *reg32(IATO_SMMU_REG_CMDQ_PROD);
    uint32_t idx = prod & (IATO_CMDQ_ENTRIES - 1U);
    iato_smmu_cmdq[idx][0] = cmd0;
    iato_smmu_cmdq[idx][1] = cmd1;
    prod++;
    *reg32(IATO_SMMU_REG_CMDQ_PROD) = prod;
#ifdef IATO_MOCK_MMIO
    *reg32(IATO_SMMU_REG_CMDQ_CONS) = prod;
#endif
    if (wait_sync != 0) {
        uint32_t i;
        for (i = 0U; i < IATO_TIMEOUT_ITERS; ++i) {
            if (*reg32(IATO_SMMU_REG_CMDQ_CONS) == prod) {
                return IATO_SMMU_OK;
            }
        }
        return IATO_SMMU_ERR_TIMEOUT;
    }
    return IATO_SMMU_OK;
}

int iato_smmu_init(void) {
    uint32_t i;
    uint32_t idr0 = *reg32(IATO_SMMU_REG_IDR0);
    uint32_t idr1 = *reg32(IATO_SMMU_REG_IDR1);

    if (iato_smmu_initialized != 0) {
        return IATO_SMMU_OK;
    }
    if ((idr0 == 0U) && (idr1 == 0U)) {
        return IATO_SMMU_ERR_IDR;
    }
    *reg32(IATO_SMMU_REG_CR0) = 0U;
    if (poll_cr0ack(0U) != IATO_SMMU_OK) {
        return IATO_SMMU_ERR_TIMEOUT;
    }

    for (i = 0U; i < IATO_SMMU_MAX_STREAMS; ++i) {
        uint32_t w;
        for (w = 0U; w < IATO_STE_SIZE_WORDS; ++w) {
            iato_smmu_strtab[i][w] = 0ULL;
        }
    }
    IATO_DSB_SY();

    if ((((uintptr_t)&iato_smmu_strtab[0][0]) & (IATO_STE_SIZE_BYTES - 1U)) != 0U) {
        return IATO_SMMU_ERR_STRTAB;
    }

    *reg64(IATO_SMMU_REG_STRTAB_BASE) = (uint64_t)(uintptr_t)&iato_smmu_strtab[0][0];
    *reg32(IATO_SMMU_REG_STRTAB_BASE_CFG) = ilog2_u64(IATO_SMMU_MAX_STREAMS);
    *reg64(IATO_SMMU_REG_CMDQ_BASE) = (uint64_t)(uintptr_t)&iato_smmu_cmdq[0][0];
    *reg32(IATO_SMMU_REG_CMDQ_PROD) = 0U;
    *reg32(IATO_SMMU_REG_CMDQ_CONS) = 0U;

    *reg32(IATO_SMMU_REG_CR0) = 1U;
    if (poll_cr0ack(1U) != IATO_SMMU_OK) {
        return IATO_SMMU_ERR_TIMEOUT;
    }
    IATO_DSB_SY();
    IATO_ISB();
    iato_smmu_initialized = 1;
    return IATO_SMMU_OK;
}

int iato_smmu_write_ste(uint32_t stream_id, uint64_t pa_base, uint64_t pa_limit, uint8_t permissions) {
    uint64_t range;
    uint64_t word0;
    uint64_t word2;
    int rc;
    if ((stream_id >= IATO_SMMU_MAX_STREAMS) || ((pa_base & 0xFFFULL) != 0ULL) || ((pa_limit & 0xFFFULL) != 0ULL) || (pa_limit <= pa_base)) {
        return IATO_SMMU_ERR_RANGE;
    }
    range = pa_limit - pa_base;
    word0 = 1ULL | (4ULL << 1U) | (((uint64_t)(permissions & 0x3U)) << 6U);
    word2 = (1ULL << 10U) | (uint64_t)(64U - ilog2_u64(range));

    iato_smmu_strtab[stream_id][0] = word0;
    iato_smmu_strtab[stream_id][1] = 0ULL;
    iato_smmu_strtab[stream_id][2] = word2;
    iato_smmu_strtab[stream_id][3] = pa_base >> 12U;
    iato_smmu_strtab[stream_id][4] = 0ULL;
    iato_smmu_strtab[stream_id][5] = 0ULL;
    iato_smmu_strtab[stream_id][6] = 0ULL;
    iato_smmu_strtab[stream_id][7] = 0ULL;

    rc = cmdq_issue(0x1ULL, (uint64_t)stream_id, 0);
    if (rc != IATO_SMMU_OK) {
        return rc;
    }
    rc = cmdq_issue(0x2ULL, 0ULL, 1);
    if (rc != IATO_SMMU_OK) {
        return rc;
    }
    IATO_DSB_SY();
    IATO_ISB();
    return IATO_SMMU_OK;
}

int iato_smmu_fault_ste(uint32_t stream_id) {
    uint32_t w;
    int rc;
    if (stream_id >= IATO_SMMU_MAX_STREAMS) {
        return IATO_SMMU_ERR_RANGE;
    }
    for (w = 0U; w < IATO_STE_SIZE_WORDS; ++w) {
        iato_smmu_strtab[stream_id][w] = 0ULL;
    }
    rc = cmdq_issue(0x1ULL, (uint64_t)stream_id, 0);
    if (rc != IATO_SMMU_OK) {
        return rc;
    }
    rc = cmdq_issue(0x2ULL, 0ULL, 1);
    if (rc != IATO_SMMU_OK) {
        return rc;
    }
    IATO_DSB_SY();
    IATO_ISB();
    return IATO_SMMU_OK;
}

uint64_t iato_smmu_read_ste_word0(uint32_t stream_id) {
    if (stream_id >= IATO_SMMU_MAX_STREAMS) {
        return 0ULL;
    }
    return iato_smmu_strtab[stream_id][0];
}
