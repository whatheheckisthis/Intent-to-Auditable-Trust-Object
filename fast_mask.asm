; fast_mask.asm
; NASM x86_64 routine to zero sensitive buffers quickly.
; SysV ABI: void fast_mask(void *buf, size_t len)

global fast_mask
section .text

fast_mask:
    ; rdi = buffer pointer, rsi = length
    test rdi, rdi
    jz .done
    test rsi, rsi
    jz .done

    xor eax, eax          ; value to store
    mov rcx, rsi          ; byte count
    rep stosb             ; memset(buf, 0, len)

.done:
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
