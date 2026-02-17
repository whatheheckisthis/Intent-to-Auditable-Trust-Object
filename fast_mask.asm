; fast_mask.asm
; SysV ABI:
;   void fast_mask_ip_pairs(unsigned char *buf, const unsigned char *mask, size_t blocks16)
;
; Applies SIMD mask to each 16-byte block: buf[i] &= mask[i]
; Uses movdqu + pand to support unaligned flow records.

global fast_mask_ip_pairs
section .text

fast_mask_ip_pairs:
    ; rdi=buf, rsi=mask, rdx=blocks16
    test rdi, rdi
    jz .done
    test rsi, rsi
    jz .done
    test rdx, rdx
    jz .done

.loop:
    movdqu xmm0, [rdi]
    movdqu xmm1, [rsi]
    pand xmm0, xmm1
    movdqu [rdi], xmm0

    add rdi, 16
    dec rdx
    jnz .loop

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
