#ifndef IATO_CNTHP_DRIVER_H
#define IATO_CNTHP_DRIVER_H

#include <stdint.h>

#define IATO_CNTHP_OK       0
#define IATO_CNTHP_ERR_FREQ -1
#define IATO_CNTHP_ERR_IRQ  -2

typedef void (*iato_sweep_fn_t)(uint64_t elapsed_ns);

int iato_cnthp_init(void);
void iato_cnthp_arm_ns(uint64_t interval_ns);
void iato_cnthp_disarm(void);
void iato_cnthp_irq_handler(void);
void iato_cnthp_register_sweep(iato_sweep_fn_t callback);
void iato_cnthp_set_interval(uint64_t interval_ns);
uint64_t iato_cnthp_read_elapsed_ns(void);

#ifdef IATO_HOST_TEST
uint64_t iato_cnthp_test_read_ctl(void);
uint64_t iato_cnthp_test_read_tval(void);
#endif

#endif
