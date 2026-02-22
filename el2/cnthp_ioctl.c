#include "cnthp_ioctl.h"

extern int iato_wq_wait(void *wq);
extern void iato_wq_wake(void *wq);

static volatile uint32_t iato_timer_wq;

int iato_cnthp_ioctl(uint32_t cmd, uint64_t arg) {
    if (cmd == IATO_TIMER_ARM) {
        if (arg == 0ULL) {
            return -22;
        }
        iato_cnthp_set_interval(arg);
        iato_cnthp_arm_ns(arg);
        return 0;
    }
    if (cmd == IATO_TIMER_DISARM) {
        iato_cnthp_disarm();
        return 0;
    }
    return -22;
}

int iato_cnthp_read_elapsed(uint64_t *out_elapsed_ns) {
    int rc;
    rc = iato_wq_wait((void *)&iato_timer_wq);
    if (rc != 0) {
        return -4;
    }
    *out_elapsed_ns = iato_cnthp_read_elapsed_ns();
    return 8;
}

void iato_cnthp_signal_irq(void) {
    iato_wq_wake((void *)&iato_timer_wq);
}
