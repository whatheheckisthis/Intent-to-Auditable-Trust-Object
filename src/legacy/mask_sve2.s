/*
 * SVE2 masking primitive for OSINT sink sanitization.
 * z0.s/z1.s carry IPv4 fields and z5.d/z6.d carry timestamps.
 */

    .text
    .align 4
    .global mask_v7_tuple_sve2
    .global mask_v7_sve2
    .type mask_v7_tuple_sve2, %function
    .type mask_v7_sve2, %function
    .arch armv9-a+sve2

mask_v7_sve2:
mask_v7_tuple_sve2:
    ptrue       p0.b

    /* IPv4 addresses -> /24 */
    mov         w9, #0xFF00
    movk        w9, #0xFFFF, lsl #16
    dup         z31.s, w9
    and         z0.s, z0.s, z31.s
    and         z1.s, z1.s, z31.s

    /* Nanosecond timestamps -> minute buckets. */
    mov         x10, #60000000000
    dup         z30.d, x10
    udiv        z5.d, p0/m, z5.d, z30.d
    mul         z5.d, z5.d, z30.d
    udiv        z6.d, p0/m, z6.d, z30.d
    mul         z6.d, z6.d, z30.d
    ret

    .size mask_v7_tuple_sve2, .-mask_v7_tuple_sve2
    .size mask_v7_sve2, .-mask_v7_sve2
