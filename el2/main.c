#include <stdint.h>
#include <stddef.h>

#include "cnthp_driver.h"
#include "expiry_sweep.h"
#include "py_embed.h"
#include "smc_handler.h"
#include "smmu_init.h"
#include "uart.h"

extern void iato_guest_boot(void);
extern int iato_irq_register(uint32_t irq, void (*handler)(void));

#ifndef IATO_MOCK_ALL
static inline uint64_t iato_read_esr_el2(void) { uint64_t v; __asm__ volatile("mrs %0, esr_el2" : "=r"(v)); return v; }
static inline uint64_t iato_read_far_el2(void) { uint64_t v; __asm__ volatile("mrs %0, far_el2" : "=r"(v)); return v; }
static inline uint64_t iato_read_disr_el1(void) { uint64_t v; __asm__ volatile("mrs %0, disr_el1" : "=r"(v)); return v; }
static inline uint64_t iato_read_iar(void) { uint64_t v; __asm__ volatile("mrs %0, ICC_IAR1_EL1" : "=r"(v)); return v; }
static inline void iato_write_eoir(uint64_t v) { __asm__ volatile("msr ICC_EOIR1_EL1, %0" :: "r"(v)); }
#else
uint64_t iato_mock_esr;
uint64_t iato_mock_far;
uint64_t iato_mock_disr;
uint64_t iato_mock_iar;
uint64_t iato_mock_eoir;
static inline uint64_t iato_read_esr_el2(void) { return iato_mock_esr; }
static inline uint64_t iato_read_far_el2(void) { return iato_mock_far; }
static inline uint64_t iato_read_disr_el1(void) { return iato_mock_disr; }
static inline uint64_t iato_read_iar(void) { return iato_mock_iar; }
static inline void iato_write_eoir(uint64_t v) { iato_mock_eoir = v; }
#endif

static void iato_spin(void) { for (;;) { } }
static void iato_expiry_sweep_cb(uint64_t elapsed_ns) { (void)iato_expiry_sweep(elapsed_ns); }

void iato_data_abort_handler(void) {
    iato_uart_puts("[iato][fatal] data abort esr=");
    iato_uart_puthex(iato_read_esr_el2());
    iato_uart_puts(" far=");
    iato_uart_puthex(iato_read_far_el2());
    iato_uart_puts("\n");
    iato_spin();
}

void iato_unhandled_sync(void) {
    iato_uart_puts("[iato][fatal] unhandled sync\n");
    iato_spin();
}

void iato_serror_handler(void) {
    iato_uart_puts("[iato][fatal] serror esr=");
    iato_uart_puthex(iato_read_esr_el2());
    iato_uart_puts(" disr=");
    iato_uart_puthex(iato_read_disr_el1());
    iato_uart_puts("\n");
    iato_spin();
}

void iato_sync_handler(uint64_t regs[4]) {
    uint64_t ec = (iato_read_esr_el2() >> 26U) & 0x3FULL;
    if (ec == 0x17U) {
        regs[0] = iato_smc_handle(regs);
    } else if (ec == 0x15U) {
        regs[0] = IATO_SMC_ERR_INTERNAL;
    } else if (ec == 0x24U) {
        iato_data_abort_handler();
    } else {
        iato_unhandled_sync();
    }
}

void iato_irq_handler(void) {
    uint64_t iar = iato_read_iar();
    uint32_t intid = (uint32_t)(iar & 0x3FFU);
    if (intid == 1023U) {
        return;
    }
    if (intid == 26U) {
        iato_cnthp_irq_handler();
    }
    iato_write_eoir(iar);
}

void iato_main(void) {
    iato_uart_init();
    iato_uart_puts("[iato] EL2 hypervisor starting\n");
    if (iato_smmu_init() != IATO_SMMU_OK) { iato_uart_puts("[iato][fatal] smmu_init failed\n"); iato_spin(); }
    if (iato_smc_init() != 0) { iato_uart_puts("[iato][fatal] smc_init failed\n"); iato_spin(); }
    if (iato_cnthp_init() != IATO_CNTHP_OK) { iato_uart_puts("[iato][fatal] cnthp_init failed\n"); iato_spin(); }
    if (iato_py_embed_init() != IATO_PY_OK) { iato_uart_puts("[iato][fatal] py_embed_init failed\n"); iato_spin(); }
    iato_cnthp_register_sweep(iato_expiry_sweep_cb);
    iato_cnthp_arm_ns(30ULL * 1000000000ULL);
    iato_uart_puts("[iato] EL2 hypervisor ready\n");
    iato_guest_boot();
    iato_spin();
}
