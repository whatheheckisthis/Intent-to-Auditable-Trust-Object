    .text
    .globl branchless_reduce_asm
    .type branchless_reduce_asm, @function
branchless_reduce_asm:
    mov %rdi, %rax        # result starts as t
    cmp %rsi, %rdi        # compare t vs M
    sbb %rdx, %rdx        # rdx = 0x00..00 or 0xff..ff (mask)
    and %rsi, %rdx        # either M or 0
    sub %rdx, %rax        # t - (M & mask)
    ret
