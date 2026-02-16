#ifndef SNARK_FPGA_MMIO_H
#define SNARK_FPGA_MMIO_H

/*
 * MMIO register map (32-bit words) mirrored by userspace into fpga_mmio_shadow.
 * eBPF writes packet metadata into command-doorbell registers and reads witness
 * status/data registers after FPGA completion.
 */

#define FPGA_MMIO_REG_SRC_IP            0U
#define FPGA_MMIO_REG_DST_IP            1U
#define FPGA_MMIO_REG_PORTS             2U
#define FPGA_MMIO_REG_PKT_META          3U
#define FPGA_MMIO_REG_FLOW_HASH         4U
#define FPGA_MMIO_REG_TS_LOW            5U

#define FPGA_MMIO_REG_WITNESS_STATUS    16U
#define FPGA_MMIO_WITNESS_BASE          32U

#define FPGA_MMIO_WITNESS_WORDS         8U
#define FPGA_MMIO_WORDS                 128U

/* Witness status register format */
#define FPGA_WITNESS_READY_MASK         0x1U
#define FPGA_WITNESS_FAIL_MASK          0x2U
#define FPGA_WITNESS_EPOCH_SHIFT        2U

#endif /* SNARK_FPGA_MMIO_H */
