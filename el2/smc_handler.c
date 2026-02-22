#include "smc_handler.h"

typedef struct {
    uint32_t tokens;
    uint64_t last_refill_ns;
    uint8_t failure_count;
    uint64_t penalty_until_ns;
} iato_rate_state_t;

static uint8_t iato_cred_staging[IATO_CRED_MAX_BYTES] __attribute__((aligned(8)));
static iato_rate_state_t iato_rate[IATO_SMMU_MAX_STREAMS];

extern int iato_py_validate_credential(const uint8_t *cred, size_t cred_len, uint32_t stream_id,
                                       uint64_t *out_pa_base, uint64_t *out_pa_limit, uint8_t *out_permissions);

#ifdef IATO_MOCK_CNTVCT
extern uint64_t iato_mock_cntvct_now;
static uint64_t iato_read_cntvct(void) { return iato_mock_cntvct_now; }
#else
static uint64_t iato_read_cntvct(void) {
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
#endif

static void iato_staging_clear(void) {
    size_t i;
    for (i = 0U; i < IATO_CRED_MAX_BYTES; ++i) {
        iato_cred_staging[i] = 0U;
    }
}

static void rate_refill(iato_rate_state_t *s, uint64_t now) {
    uint64_t elapsed = now - s->last_refill_ns;
    uint32_t add = (uint32_t)(elapsed / 1000000000ULL);
    if (add > 0U) {
        uint32_t nt = s->tokens + add;
        s->tokens = (nt > 5U) ? 5U : nt;
        s->last_refill_ns = now;
    }
}

static int rate_allow(uint32_t sid, uint64_t now) {
    iato_rate_state_t *s = &iato_rate[sid];
    rate_refill(s, now);
    if (s->penalty_until_ns > now) {
        return 0;
    }
    if (s->tokens == 0U) {
        return 0;
    }
    s->tokens--;
    return 1;
}

static void rate_failure(uint32_t sid, uint64_t now) {
    iato_rate_state_t *s = &iato_rate[sid];
    if (s->failure_count < 255U) {
        s->failure_count++;
    }
    if (s->failure_count >= 3U) {
        s->penalty_until_ns = now + 60000000000ULL;
        s->failure_count = 0U;
    }
}

static void rate_success(uint32_t sid) {
    iato_rate[sid].failure_count = 0U;
}

int iato_smc_init(void) {
    uint32_t i;
    iato_staging_clear();
    for (i = 0U; i < IATO_SMMU_MAX_STREAMS; ++i) {
        iato_rate[i].tokens = 5U;
        iato_rate[i].last_refill_ns = 0ULL;
        iato_rate[i].failure_count = 0U;
        iato_rate[i].penalty_until_ns = 0ULL;
    }
    return 0;
}

int iato_smc_copy_from_guest(void *dst, uint64_t guest_pa, size_t length) {
    size_t i;
    uint64_t end = guest_pa + (uint64_t)length;
    volatile uint8_t *d = (volatile uint8_t *)dst;
    if ((guest_pa < IATO_GUEST_RAM_BASE) || (end > (IATO_GUEST_RAM_BASE + IATO_GUEST_RAM_SIZE)) || (end < guest_pa)) {
        return (int)IATO_SMC_ERR_GPA;
    }
#ifdef IATO_MOCK_VALIDATOR
    extern uint8_t iato_mock_guest_ram[IATO_GUEST_RAM_SIZE];
    volatile uint8_t *src = (volatile uint8_t *)&iato_mock_guest_ram[guest_pa - IATO_GUEST_RAM_BASE];
#else
    volatile uint8_t *src = (volatile uint8_t *)(uintptr_t)guest_pa;
#endif
    for (i = 0U; i < length; ++i) {
        d[i] = src[i];
    }
    return 0;
}

uint64_t iato_smc_handle(uint64_t regs[4]) {
    uint32_t sid;
    uint64_t len;
    uint64_t gpa;
    uint64_t pa_base = 0ULL;
    uint64_t pa_limit = 0ULL;
    uint8_t perms = 0U;
    int rc;
    uint64_t now;

    if (regs[0] != IATO_SMC_FUNCTION_ID) {
        return IATO_SMC_ERR_INTERNAL;
    }
    sid = (uint32_t)regs[3];
    if (sid >= IATO_SMMU_MAX_STREAMS) {
        return IATO_SMC_ERR_STREAM;
    }
    now = iato_read_cntvct();
    if (!rate_allow(sid, now)) {
        return IATO_SMC_ERR_RATE;
    }

    len = regs[2];
    if ((len != IATO_CRED_MIN_BYTES) && (len != IATO_CRED_MAX_BYTES)) {
        rate_failure(sid, now);
        iato_staging_clear();
        return IATO_SMC_ERR_LENGTH;
    }
    gpa = regs[1];
    rc = iato_smc_copy_from_guest(iato_cred_staging, gpa, (size_t)len);
    if (rc != 0) {
        rate_failure(sid, now);
        iato_staging_clear();
        return IATO_SMC_ERR_GPA;
    }

    rc = iato_py_validate_credential(iato_cred_staging, (size_t)len, sid, &pa_base, &pa_limit, &perms);
    if (rc != 0) {
        rate_failure(sid, now);
        iato_staging_clear();
        return IATO_SMC_ERR_VALIDATION;
    }

    rc = iato_smmu_write_ste(sid, pa_base, pa_limit, perms);
    if (rc != IATO_SMMU_OK) {
        rate_failure(sid, now);
        iato_staging_clear();
        return IATO_SMC_ERR_SMMU;
    }

    rate_success(sid);
    iato_staging_clear();
    return IATO_SMC_OK;
}

#ifdef IATO_HOST_TEST
const uint8_t *iato_smc_staging_ptr(void) {
    return &iato_cred_staging[0];
}
#endif
