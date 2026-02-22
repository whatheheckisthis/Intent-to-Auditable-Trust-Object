#ifndef IATO_UART_H
#define IATO_UART_H

#include <stdint.h>

void iato_uart_init(void);
void iato_uart_putc(char c);
void iato_uart_puts(const char *s);
void iato_uart_puthex(uint64_t v);

#endif
