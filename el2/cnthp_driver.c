#include "cnthp_driver.h"

#define IATO_GIC_PPI_CNTHP 26U

static uint64_t iato_cnthp_freq_hz;
static uint64_t iato_cnthp_interval_ns = 30ULL * 1000000000ULL;
static uint64_t iato_cnthp_last_arm_cval;
static iato_sweep_fn_t iato_cnthp_sweep_cb;
static volatile int iato_cnthp_active;

#ifdef IATO_MOCK_CNTVCT
extern uint64_t iato_mock_cntfrq;
extern uint64_t iato_mock_cntpct;
extern uint64_t iato_mock_cnthp_tval;
extern uint64_t iato_mock_cnthp_ctl;
static inline uint64_t read_cntfrq(void) { return iato_mock_cntfrq; }
static inline uint64_t read_cntpct(void) { return iato_mock_cntpct; }
static inline void write_cnthp_tval(uint64_t v) { iato_mock_cnthp_tval = v; }
static inline void write_cnthp_ctl(uint64_t v) { iato_mock_cnthp_ctl = v; }
static inline uint64_t read_cnthp_ctl(void) { return iato_mock_cnthp_ctl; }
#else
static inline uint64_t read_cntfrq(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }
static inline uint64_t read_cntpct(void) { uint64_t v; __asm__ volatile("mrs %0, cntpct_el0" : "=r"(v)); return v; }
static inline void write_cnthp_tval(uint64_t v) { __asm__ volatile("msr cnthp_tval_el2, %0" :: "r"(v)); }
static inline void write_cnthp_ctl(uint64_t v) { __asm__ volatile("msr cnthp_ctl_el2, %0" :: "r"(v)); __asm__ volatile("isb"); }
static inline uint64_t read_cnthp_ctl(void) { uint64_t v; __asm__ volatile("mrs %0, cnthp_ctl_el2" : "=r"(v)); return v; }
#endif

extern int iato_irq_register(uint32_t irq, void (*handler)(void));

int iato_cnthp_init(void) {
    iato_cnthp_freq_hz = read_cntfrq();
    if ((iato_cnthp_freq_hz == 0ULL) || (iato_cnthp_freq_hz > 1000000000ULL)) {
        return IATO_CNTHP_ERR_FREQ;
    }
    write_cnthp_ctl(0x2ULL);
    if (iato_irq_register(IATO_GIC_PPI_CNTHP, iato_cnthp_irq_handler) != 0) {
        return IATO_CNTHP_ERR_IRQ;
    }
    iato_cnthp_active = 0;
    return IATO_CNTHP_OK;
}

void iato_cnthp_set_interval(uint64_t interval_ns) {
    iato_cnthp_interval_ns = interval_ns;
}

void iato_cnthp_arm_ns(uint64_t interval_ns) {
    uint64_t ticks;
    if (interval_ns == 0ULL) {
        iato_cnthp_disarm();
        return;
    }
    ticks = (interval_ns * iato_cnthp_freq_hz) / 1000000000ULL;
    iato_cnthp_last_arm_cval = read_cntpct();
    write_cnthp_tval(ticks);
    write_cnthp_ctl(0x1ULL);
    iato_cnthp_active = 1;
}

void iato_cnthp_disarm(void) {
    write_cnthp_ctl(0x2ULL);
    iato_cnthp_active = 0;
}

uint64_t iato_cnthp_read_elapsed_ns(void) {
    uint64_t delta = read_cntpct() - iato_cnthp_last_arm_cval;
    if (iato_cnthp_freq_hz == 0ULL) {
        return 0ULL;
    }
    return (delta * 1000000000ULL) / iato_cnthp_freq_hz;
}

void iato_cnthp_register_sweep(iato_sweep_fn_t callback) {
    iato_cnthp_sweep_cb = callback;
}

void iato_cnthp_irq_handler(void) {
    uint64_t ctl = read_cnthp_ctl();
    uint64_t elapsed;
    if ((ctl & (1ULL << 2U)) == 0ULL) {
        return;
    }
    write_cnthp_ctl(0x2ULL);
#ifdef IATO_MOCK_CNTVCT
    iato_mock_cnthp_ctl &= ~(1ULL << 2U);
#endif
    elapsed = iato_cnthp_read_elapsed_ns();
    if (iato_cnthp_sweep_cb != (iato_sweep_fn_t)0) {
        iato_cnthp_sweep_cb(elapsed);
    }
    if ((iato_cnthp_active != 0) && (iato_cnthp_interval_ns != 0ULL)) {
        iato_cnthp_arm_ns(iato_cnthp_interval_ns);
    }
}

#ifdef IATO_HOST_TEST
uint64_t iato_cnthp_test_read_ctl(void) { return read_cnthp_ctl(); }
uint64_t iato_cnthp_test_read_tval(void) { return iato_mock_cnthp_tval; }
#endif
