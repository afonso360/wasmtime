;;! target = "x86_64"
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
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %r8
;;       movl    %edx, %r10d
;;       subq    $0x1001, %r8
;;       cmpq    %r8, %r10
;;       ja      0x2c
;;   1b: movq    0x60(%rdi), %rsi
;;       movb    %cl, 0x1000(%rsi, %r10)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   2c: ud2
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %r9
;;       movl    %edx, %r11d
;;       subq    $0x1001, %r9
;;       cmpq    %r9, %r11
;;       ja      0x71
;;   5b: movq    0x60(%rdi), %rdi
;;       movzbq  0x1000(%rdi, %r11), %rdi
;;       movzbl  %dil, %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   71: ud2
