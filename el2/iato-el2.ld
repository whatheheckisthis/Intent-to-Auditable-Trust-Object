ENTRY(iato_reset_vector)

MEMORY {
  RAM (rwx) : ORIGIN = 0x40000000, LENGTH = 128M
}

SECTIONS {
  . = ORIGIN(RAM);

  .text.boot ALIGN(2048) : {
    __text_boot_start = .;
    KEEP(*(.text.boot))
    KEEP(*(.vectors))
    __text_boot_end = .;
  } > RAM

  ASSERT((ADDR(.text.boot) % 2048) == 0, ".text.boot must be 2KB aligned")

  .text ALIGN(64) : {
    *(.text*)
  } > RAM

  .rodata ALIGN(64) : {
    *(.rodata*)
  } > RAM

  .data ALIGN(64) : {
    *(.data*)
  } > RAM

  .bss ALIGN(64) (NOLOAD) : {
    __bss_start = .;
    *(.bss*)
    *(COMMON)
    __bss_end = .;
  } > RAM

  .stack ALIGN(64) (NOLOAD) : {
    . = . + 0x4000;
    __stack_top = .;
  } > RAM

  ASSERT((iato_smmu_strtab - ORIGIN(RAM)) < 0x400000, "iato_smmu_strtab beyond first 4MB")
}
