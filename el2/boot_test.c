#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "smc_handler.h"
#include "uart.h"

extern uint64_t iato_mock_esr;
extern uint64_t iato_mock_iar;
extern uint64_t iato_mock_eoir;
extern volatile uint32_t iato_mock_uart_mmio[];
static int g_smc_called;
static int g_irq_called;

uint64_t iato_smc_handle(uint64_t regs[4]) { g_smc_called++; return regs[0] + 1U; }
void iato_cnthp_irq_handler(void) { g_irq_called++; }

int iato_smmu_init(void){ return 0; }
int iato_smc_init(void){ return 0; }
int iato_cnthp_init(void){ return 0; }
int iato_py_embed_init(void){ return 0; }
int iato_expiry_sweep(uint64_t elapsed_ns){ (void)elapsed_ns; return 0; }
void iato_cnthp_arm_ns(uint64_t interval_ns){ (void)interval_ns; }
void iato_cnthp_register_sweep(void *cb){ (void)cb; }
void iato_guest_boot(void){}

void iato_sync_handler(uint64_t regs[4]);
void iato_irq_handler(void);

int main(void) {
    uint64_t regs[4] = { IATO_SMC_FUNCTION_ID, 0, 0, 0 };

    iato_mock_esr = (0x17ULL << 26U);
    iato_sync_handler(regs);
    assert(g_smc_called == 1);

    iato_mock_esr = (0x24ULL << 26U);
    /* data abort spins in production; host variant uses mock symbol below */

    iato_mock_iar = 26U;
    iato_irq_handler();
    assert(g_irq_called == 1);
    assert(iato_mock_eoir == 26U);

    iato_mock_iar = 1023U;
    iato_irq_handler();
    assert(iato_mock_eoir == 26U);

    iato_uart_puts("abc");
    assert((char)iato_mock_uart_mmio[0] == 'c');

    iato_uart_puthex(0x1234ULL);
    assert((char)iato_mock_uart_mmio[0] == '4');

    puts("boot_test: ok");
    return 0;
}
