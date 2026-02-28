; fast_mask.asm
; SIMD masking helper for IPv4/IPv6 address pairs.
; SysV ABI:
;   void fast_mask_ip_pair(void *src16, void *dst16, uint8_t family)
; family: 4 => keep /24 of IPv4-mapped low dword, 6 => keep /64 prefix.

default rel
global fast_mask_ip_pair

section .rodata align=16
ipv4_mask: db 0xff,0xff,0xff,0x00,0,0,0,0,0,0,0,0,0,0,0,0
ipv6_mask: db 0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0,0,0,0,0,0,0,0

section .text
fast_mask_ip_pair:
    test rdi, rdi
    jz .done
    test rsi, rsi
    jz .done

    cmp dl, 4
    je .v4

.v6:
    movdqu xmm2, [rel ipv6_mask]
    jmp .apply

.v4:
    movdqu xmm2, [rel ipv4_mask]

.apply:
    movdqu xmm0, [rdi]
    movdqu xmm1, [rsi]
    pand xmm0, xmm2
    pand xmm1, xmm2
    movdqu [rdi], xmm0
    movdqu [rsi], xmm1

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
