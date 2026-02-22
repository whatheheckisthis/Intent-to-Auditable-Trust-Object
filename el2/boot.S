.section .text.boot, "ax"
.align 11
.global iato_reset_vector
.global iato_vectors
.extern iato_main
.extern iato_sync_handler
.extern iato_irq_handler
.extern iato_serror_handler

iato_reset_vector:
    mrs x0, CurrentEL
    lsr x0, x0, #2
    cmp x0, #2
    b.ne iato_wrong_el_loop

    ldr x0, =__stack_top
    mov sp, x0

    ldr x0, =__bss_start
    ldr x1, =__bss_end
1:
    cmp x0, x1
    b.hs 2f
    str xzr, [x0], #8
    b 1b
2:
    mov x0, #0
    msr sctlr_el2, x0

    mov x0, #(1 << 31)
    orr x0, x0, #(1 << 4)
    orr x0, x0, #(1 << 3)
    msr hcr_el2, x0

    ldr x0, =iato_vectors
    msr vbar_el2, x0

    mov x0, #0x9
    msr ICC_SRE_EL2, x0
    isb
    mov x0, #0xFF
    msr ICC_PMR_EL1, x0

    msr daifclr, #2

    bl iato_main

iato_halt_loop:
    wfe
    b iato_halt_loop

iato_wrong_el_loop:
    wfe
    b iato_wrong_el_loop

.align 11
.section .vectors, "ax"
iato_vectors:
    b vector_sp0_sync
    .space 0x80 - 4
    b vector_sp0_irq
    .space 0x80 - 4
    b vector_sp0_fiq
    .space 0x80 - 4
    b vector_sp0_serror
    .space 0x80 - 4
    b vector_spx_sync
    .space 0x80 - 4
    b vector_spx_irq
    .space 0x80 - 4
    b vector_spx_fiq
    .space 0x80 - 4
    b vector_spx_serror
    .space 0x80 - 4
    b vector_l64_sync
    .space 0x80 - 4
    b vector_l64_irq
    .space 0x80 - 4
    b vector_l64_fiq
    .space 0x80 - 4
    b vector_l64_serror
    .space 0x80 - 4
    b vector_l32_sync
    .space 0x80 - 4
    b vector_l32_irq
    .space 0x80 - 4
    b vector_l32_fiq
    .space 0x80 - 4
    b vector_l32_serror
    .space 0x80 - 4

.macro SAVE4
    sub sp, sp, #32
    stp x0, x1, [sp]
    stp x2, x3, [sp, #16]
.endm
.macro REST4
    ldp x0, x1, [sp]
    ldp x2, x3, [sp, #16]
    add sp, sp, #32
.endm

vector_sp0_sync: b iato_unhandled_exception
vector_sp0_irq: b iato_unhandled_exception
vector_sp0_fiq: b iato_unhandled_exception
vector_sp0_serror: b iato_unhandled_exception

vector_spx_sync:
    SAVE4
    mov x0, sp
    bl iato_sync_handler
    REST4
    eret
vector_spx_irq:
    SAVE4
    bl iato_irq_handler
    REST4
    eret
vector_spx_fiq: b iato_unhandled_exception
vector_spx_serror:
    SAVE4
    bl iato_serror_handler
    REST4
    eret

vector_l64_sync:
    SAVE4
    mov x0, sp
    bl iato_sync_handler
    REST4
    eret
vector_l64_irq:
    SAVE4
    bl iato_irq_handler
    REST4
    eret
vector_l64_fiq: b iato_unhandled_exception
vector_l64_serror:
    SAVE4
    bl iato_serror_handler
    REST4
    eret

vector_l32_sync: b iato_unhandled_exception
vector_l32_irq: b iato_unhandled_exception
vector_l32_fiq: b iato_unhandled_exception
vector_l32_serror: b iato_unhandled_exception

iato_unhandled_exception:
    mrs x0, esr_el2
    mrs x1, far_el2
    mov x2, #0x09000000
    str x0, [x2]
    str x1, [x2, #8]
1:  wfe
    b 1b
