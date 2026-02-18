/* SPDX-License-Identifier: Apache-2.0 */

.arch armv9-a+sve2+sha3
.text

.macro zero_reg_cleanup
    pfalse p0.b
    pfalse p1.b
    pfalse p2.b
    pfalse p3.b
    pfalse p4.b
    pfalse p5.b
    pfalse p6.b
    pfalse p7.b
    pfalse p8.b
    pfalse p9.b
    pfalse p10.b
    pfalse p11.b
    pfalse p12.b
    pfalse p13.b
    pfalse p14.b
    pfalse p15.b

    eor z0.d, z0.d, z0.d
    eor z1.d, z1.d, z1.d
    eor z2.d, z2.d, z2.d
    eor z3.d, z3.d, z3.d
    eor z4.d, z4.d, z4.d
    eor z5.d, z5.d, z5.d
    eor z6.d, z6.d, z6.d
    eor z7.d, z7.d, z7.d
    eor z8.d, z8.d, z8.d
    eor z9.d, z9.d, z9.d
    eor z10.d, z10.d, z10.d
    eor z11.d, z11.d, z11.d
    eor z12.d, z12.d, z12.d
    eor z13.d, z13.d, z13.d
    eor z14.d, z14.d, z14.d
    eor z15.d, z15.d, z15.d
    eor z16.d, z16.d, z16.d
    eor z17.d, z17.d, z17.d
    eor z18.d, z18.d, z18.d
    eor z19.d, z19.d, z19.d
    eor z20.d, z20.d, z20.d
    eor z21.d, z21.d, z21.d
    eor z22.d, z22.d, z22.d
    eor z23.d, z23.d, z23.d
    eor z24.d, z24.d, z24.d
    eor z25.d, z25.d, z25.d
    eor z26.d, z26.d, z26.d
    eor z27.d, z27.d, z27.d
    eor z28.d, z28.d, z28.d
    eor z29.d, z29.d, z29.d
    eor z30.d, z30.d, z30.d
    eor z31.d, z31.d, z31.d
.endm

.global iato_sve2_mask_and_sign
.type iato_sve2_mask_and_sign, %function
iato_sve2_mask_and_sign:
    /* x0=input_ptr, x1=mask_ptr, x2=mailbox_ptr, x3=digest_out */
    ptrue p0.b                  /* svptrue_b8 style all-active predicate */

    ld1b z0.b, p0/z, [x0]
    ld1b z1.b, p0/z, [x1]
    eor  z0.d, z0.d, z1.d       /* in-register blinding */
    eor  z1.d, z1.d, z1.d

    /* SHA-256 leaf digest over masked bytes (single accelerated round set). */
    movi v24.4s, #0
    movi v25.4s, #0
    mov  v16.16b, v0.16b
    mov  v17.16b, v0.16b
    rev32 v16.16b, v16.16b
    sha256su0 v17.4s, v16.4s
    sha256h  q24, q25, v16.4s
    sha256h2 q25, q24, v17.4s
    rev32 v24.16b, v24.16b
    rev32 v25.16b, v25.16b
    st1 {v24.16b}, [x3], #16
    st1 {v25.16b}, [x3]

    st1b z0.b, p0, [x2]         /* direct register-to-MMIO */
    dsb  sy
    isb

    zero_reg_cleanup
    mov x0, #0
    ret
.size iato_sve2_mask_and_sign, .-iato_sve2_mask_and_sign

.global iato_sve2_get_vl
.type iato_sve2_get_vl, %function
iato_sve2_get_vl:
    rdvl x0, #1
    ret
.size iato_sve2_get_vl, .-iato_sve2_get_vl

.global iato_sve2_zero_zregs
.type iato_sve2_zero_zregs, %function
iato_sve2_zero_zregs:
    zero_reg_cleanup
    ret
.size iato_sve2_zero_zregs, .-iato_sve2_zero_zregs
