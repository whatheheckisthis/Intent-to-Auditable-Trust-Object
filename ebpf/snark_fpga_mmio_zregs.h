#ifndef SNARK_FPGA_MMIO_ZREGS_H
#define SNARK_FPGA_MMIO_ZREGS_H

#include <stdint.h>

/*
 * Blind-write MMIO map for 256 coefficient lanes (Z-register file) in FPGA fabric.
 * Dispatcher writes 16-bit coefficients as 32-bit aligned words:
 *   write32(FPGA_MMIO_ZREG_WORD(i), coeff_i);
 */

#define FPGA_MMIO_PHYS_BASE          0x90000000ULL
#define FPGA_MMIO_WINDOW_BYTES       0x00010000ULL

#define FPGA_MMIO_CTRL_OFFSET        0x0000U
#define FPGA_MMIO_STATUS_OFFSET      0x0004U
#define FPGA_MMIO_START_OFFSET       0x0008U

#define FPGA_MMIO_ZREG_BASE_OFFSET   0x1000U
#define FPGA_MMIO_ZREG_STRIDE_BYTES  0x0004U
#define FPGA_MMIO_ZREG_COUNT         256U

#define FPGA_MMIO_ZREG_ADDR(i) \
    (FPGA_MMIO_PHYS_BASE + FPGA_MMIO_ZREG_BASE_OFFSET + ((uint64_t)(i) * FPGA_MMIO_ZREG_STRIDE_BYTES))

#define FPGA_MMIO_ZREG_WORD(i) \
    ((FPGA_MMIO_ZREG_BASE_OFFSET >> 2) + (uint32_t)(i))

#define FPGA_MMIO_CTRL_WORD          (FPGA_MMIO_CTRL_OFFSET >> 2)
#define FPGA_MMIO_STATUS_WORD        (FPGA_MMIO_STATUS_OFFSET >> 2)
#define FPGA_MMIO_START_WORD         (FPGA_MMIO_START_OFFSET >> 2)

#define FPGA_CTRL_START_MASK         0x00000001U
#define FPGA_STATUS_BUSY_MASK        0x00000001U
#define FPGA_STATUS_DONE_MASK        0x00000002U

static inline uint64_t fpga_mmio_zreg_phys(uint32_t idx) {
    return FPGA_MMIO_ZREG_ADDR(idx);
}

#endif /* SNARK_FPGA_MMIO_ZREGS_H */
