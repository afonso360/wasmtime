;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -W memory64 -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     x8, #0
;;       ldr     x9, [x2, #0x60]
;;       add     x9, x9, x4
;;       orr     x7, xzr, #0xffffffff
;;       cmp     x4, x7
;;       csel    x10, x8, x9, hi
;;       csdb
;;       strb    w5, [x10]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     x9, #0
;;       ldr     x10, [x2, #0x60]
;;       add     x10, x10, x4
;;       orr     x8, xzr, #0xffffffff
;;       cmp     x4, x8
;;       csel    x11, x9, x10, hi
;;       csdb
;;       ldrb    w11, [x11]
;;       uxtb    w2, w11
;;       ldp     x29, x30, [sp], #0x10
;;       ret
