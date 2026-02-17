/*
 * mask_sve2.s
 *
 * SVE2 masking primitives used by the virtualized dispatcher running under
 * QEMU Neoverse V2 emulation.
 *
 * Tuple lane layout (one lane per flow):
 *   z0.s = src IPv4
 *   z1.s = dst IPv4
 *   z2.h = src port
 *   z3.h = dst port
 *   z4.b = L4 proto
 *
 * The routine preserves protocol and aggressively masks direct identifiers:
 *   - IPv4 addresses are truncated to /24.
 *   - Ports are bucketed to /8 granularity.
 */

    .text
    .align 4
    .global mask_v7_tuple_sve2
    .type mask_v7_tuple_sve2, %function
    .arch armv9-a+sve2

mask_v7_tuple_sve2:
    ptrue       p0.b

    /* src/dst IPv4 -> /24 */
    mov         w9, #0xFF00
    movk        w9, #0xFFFF, lsl #16      // w9 = 0xFFFFFF00
    dup         z31.s, w9
    and         z0.s, z0.s, z31.s
    and         z1.s, z1.s, z31.s

    /* src/dst ports -> /8 buckets (upper byte only) */
    mov         w10, #0xFF00
    dup         z30.h, w10
    and         z2.h, z2.h, z30.h
    and         z3.h, z3.h, z30.h

    /* protocol left intact in z4.b for policy correlation */
    ret

    .size mask_v7_tuple_sve2, .-mask_v7_tuple_sve2
