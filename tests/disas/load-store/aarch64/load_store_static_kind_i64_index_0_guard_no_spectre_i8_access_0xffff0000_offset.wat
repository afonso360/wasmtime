;;! target = "aarch64"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -W memory64 -O static-memory-forced -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     x7, #0xffff
;;       cmp     x4, x7
;;       b.hi    #0x2c
;;   14: ldr     x9, [x2, #0x60]
;;       add     x9, x9, x4
;;       mov     x10, #0xffff0000
;;       strb    w5, [x9, x10]
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   2c: .byte   0x1f, 0xc1, 0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     x8, #0xffff
;;       cmp     x4, x8
;;       b.hi    #0x70
;;   54: ldr     x10, [x2, #0x60]
;;       add     x10, x10, x4
;;       mov     x11, #0xffff0000
;;       ldrb    w10, [x10, x11]
;;       uxtb    w2, w10
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   70: .byte   0x1f, 0xc1, 0x00, 0x00
