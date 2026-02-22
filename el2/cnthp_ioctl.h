#ifndef IATO_CNTHP_IOCTL_H
#define IATO_CNTHP_IOCTL_H

#include <stdint.h>

#include "cnthp_driver.h"

#ifndef _IO
#define _IO(type, nr) (((type) << 8) | (nr))
#define _IOW(type, nr, data_type) (((type) << 8) | (nr) | 0x10000U)
#endif

#define IATO_TIMER_ARM    _IOW('A', 1, uint64_t)
#define IATO_TIMER_DISARM _IO('A', 2)

int iato_cnthp_ioctl(uint32_t cmd, uint64_t arg);
int iato_cnthp_read_elapsed(uint64_t *out_elapsed_ns);
void iato_cnthp_signal_irq(void);

#endif
