#include "uart.h"

#define IATO_UART_BASE 0x09000000UL
#define IATO_UARTDR    0x000U
#define IATO_UARTFR    0x018U
#define IATO_UARTIBRD  0x024U
#define IATO_UARTFBRD  0x028U
#define IATO_UARTLCR_H 0x02CU
#define IATO_UARTCR    0x030U

#ifdef IATO_MOCK_UART
volatile uint32_t iato_mock_uart_mmio[0x1000U / sizeof(uint32_t)];
#define UART_BASE_PTR ((uintptr_t)&iato_mock_uart_mmio[0])
#else
#define UART_BASE_PTR ((uintptr_t)IATO_UART_BASE)
#endif

static inline volatile uint32_t *uart_reg(uint32_t off) {
    return (volatile uint32_t *)(UART_BASE_PTR + (uintptr_t)off);
}

void iato_uart_init(void) {
    *uart_reg(IATO_UARTCR) = 0U;
    *uart_reg(IATO_UARTIBRD) = 1U;
    *uart_reg(IATO_UARTFBRD) = 40U;
    *uart_reg(IATO_UARTLCR_H) = (3U << 5U);
    *uart_reg(IATO_UARTCR) = (1U << 0U) | (1U << 8U);
}

void iato_uart_putc(char c) {
    uint32_t timeout;
    for (timeout = 0U; timeout < 100000U; ++timeout) {
        if (((*uart_reg(IATO_UARTFR)) & (1U << 5U)) == 0U) {
            break;
        }
    }
    *uart_reg(IATO_UARTDR) = (uint32_t)(uint8_t)c;
}

void iato_uart_puts(const char *s) {
    while ((s != (const char *)0) && (*s != '\0')) {
        iato_uart_putc(*s);
        s++;
    }
}

void iato_uart_puthex(uint64_t v) {
    static const char h[] = "0123456789abcdef";
    int i;
    for (i = 15; i >= 0; --i) {
        uint8_t n = (uint8_t)((v >> (i * 4)) & 0xFULL);
        iato_uart_putc(h[n]);
    }
}
