    .text
    .global branchless_reduce_arm64_asm
    .type branchless_reduce_arm64_asm, %function
branchless_reduce_arm64_asm:
    // x0=t, x1=M
    subs x9, x0, x1        // set NZCV from t-M
    ngc  x10, xzr          // x10 = 0 - C  => all-ones when t>=M, else 0
    and  x11, x1, x10      // x11 = M & mask
    sub  x0, x0, x11       // result
    ret
