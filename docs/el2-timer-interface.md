# EL2 timer device contract

`/dev/iato-el2-timer` provides:
- `ioctl(IATO_TIMER_ARM, &interval_ns)`
- `ioctl(IATO_TIMER_DISARM)`
- `read(fd, buf, 8)` blocks until CNTHP interrupt.

Calls from EL0/EL1 must fail with `-EPERM`.
