;;! target = "riscv64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       addi    sp, sp, -0x10
;;       sd      ra, 8(sp)
;;       sd      s0, 0(sp)
;;       mv      s0, sp
;;       ld      a4, 0x68(a0)
;;       ld      a1, 0x60(a0)
;;       sltu    a0, a4, a2
;;       add     a1, a1, a2
;;       lui     a2, 1
;;       add     a1, a1, a2
;;       neg     a4, a0
;;       not     a0, a4
;;       and     a2, a1, a0
;;       sb      a3, 0(a2)
;;       ld      ra, 8(sp)
;;       ld      s0, 0(sp)
;;       addi    sp, sp, 0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       addi    sp, sp, -0x10
;;       sd      ra, 8(sp)
;;       sd      s0, 0(sp)
;;       mv      s0, sp
;;       ld      a1, 0x68(a0)
;;       ld      a3, 0x60(a0)
;;       sltu    a1, a1, a2
;;       add     a2, a3, a2
;;       lui     a3, 1
;;       add     a2, a2, a3
;;       neg     a5, a1
;;       not     a1, a5
;;       and     a3, a2, a1
;;       lb      a1, 0(a3)
;;       andi    a0, a1, 0xff
;;       ld      ra, 8(sp)
;;       ld      s0, 0(sp)
;;       addi    sp, sp, 0x10
;;       ret
