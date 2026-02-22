#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "cnthp_driver.h"
#include "expiry_sweep.h"

#include "smmu_init.h"

volatile uint8_t iato_mock_smmu_mmio[IATO_SMMU_SIZE];

uint64_t iato_mock_cntfrq = 62500000ULL;
uint64_t iato_mock_cntpct;
uint64_t iato_mock_cnthp_tval;
uint64_t iato_mock_cnthp_ctl;

static void (*g_irq_handler)(void);
static uint64_t g_wall;
static uint64_t g_sweep_calls;
static uint64_t g_last_elapsed;

int iato_irq_register(uint32_t irq, void (*handler)(void)) {
    (void)irq;
    g_irq_handler = handler;
    return 0;
}

uint64_t iato_wall_time_s(void) { return g_wall; }

int iato_mock_smmu_fault_ste(uint32_t stream_id) {
    (void)stream_id;
    g_sweep_calls++;
    return 0;
}

static void test_cb(uint64_t elapsed_ns) {
    g_last_elapsed = elapsed_ns;
}

int main(void) {
    int rc;
    rc = iato_cnthp_init();
    assert(rc == IATO_CNTHP_OK);

    iato_cnthp_arm_ns(30000000000ULL);
    assert(iato_cnthp_test_read_tval() == 1875000000ULL);
    assert((iato_cnthp_test_read_ctl() & 0x1ULL) == 1ULL);

    iato_cnthp_disarm();
    assert(iato_cnthp_test_read_ctl() == 0x2ULL);

    iato_cnthp_register_sweep(test_cb);
    iato_cnthp_arm_ns(1000000000ULL);
    iato_mock_cntpct += 62500000ULL;
    iato_mock_cnthp_ctl |= (1ULL << 2U);
    g_irq_handler();
    assert(g_last_elapsed == 1000000000ULL);

    iato_binding_table[1].stream_id = 1U;
    iato_binding_table[1].status = IATO_STATUS_ACTIVE;
    iato_binding_table[1].expiry_unix_s = 10U;
    g_wall = 11U;
    assert(iato_expiry_sweep(0ULL) == 1);
    assert(g_sweep_calls == 1ULL);

    puts("cnthp_driver_test: ok");
    return 0;
}
