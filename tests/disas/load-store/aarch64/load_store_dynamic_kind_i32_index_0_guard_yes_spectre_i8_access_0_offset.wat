;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x9, [x2, #0x68]
;;       ldr     x10, [x2, #0x60]
;;       mov     w11, w4
;;       mov     x12, #0
;;       add     x10, x10, w4, uxtw
;;       cmp     x11, x9
;;       csel    x10, x12, x10, hs
;;       csdb
;;       strb    w5, [x10]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       ldr     x10, [x2, #0x68]
;;       ldr     x11, [x2, #0x60]
;;       mov     w12, w4
;;       mov     x13, #0
;;       add     x11, x11, w4, uxtw
;;       cmp     x12, x10
;;       csel    x11, x13, x11, hs
;;       csdb
;;       ldrb    w12, [x11]
;;       uxtb    w2, w12
;;       ldp     x29, x30, [sp], #0x10
;;       ret
