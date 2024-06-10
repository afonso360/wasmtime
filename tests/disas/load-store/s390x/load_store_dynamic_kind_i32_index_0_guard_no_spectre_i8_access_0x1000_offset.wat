;;! target = "s390x"
;;! test = "compile"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=0 -O dynamic-memory-guard-size=0"

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i32 1)

  (func (export "do_store") (param i32 i32)
    local.get 0
    local.get 1
    i32.store8 offset=0x1000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0x1000))

;; wasm[0]::function[0]:
;;       lg      %r1, 8(%r2)
;;       lg      %r1, 0(%r1)
;;       la      %r1, 0xa0(%r1)
;;       clgrtle %r15, %r1
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lg      %r3, 0x68(%r2)
;;       llgfr   %r4, %r4
;;       aghi    %r3, -0x1001
;;       clgr    %r4, %r3
;;       jgh     0x56
;;       ag      %r4, 0x60(%r2)
;;       lghi    %r2, 0x1000
;;       stc     %r5, 0(%r2, %r4)
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
;;       .byte   0x00, 0x00
;;
;; wasm[0]::function[1]:
;;       lg      %r1, 8(%r2)
;;       lg      %r1, 0(%r1)
;;       la      %r1, 0xa0(%r1)
;;       clgrtle %r15, %r1
;;       stmg    %r14, %r15, 0x70(%r15)
;;       lgr     %r1, %r15
;;       aghi    %r15, -0xa0
;;       stg     %r1, 0(%r15)
;;       lg      %r3, 0x68(%r2)
;;       llgfr   %r5, %r4
;;       aghi    %r3, -0x1001
;;       clgr    %r5, %r3
;;       jgh     0xb4
;;       ag      %r5, 0x60(%r2)
;;       lghi    %r2, 0x1000
;;       llc     %r5, 0(%r2, %r5)
;;       llcr    %r2, %r5
;;       lmg     %r14, %r15, 0x110(%r15)
;;       br      %r14
;;       .byte   0x00, 0x00
