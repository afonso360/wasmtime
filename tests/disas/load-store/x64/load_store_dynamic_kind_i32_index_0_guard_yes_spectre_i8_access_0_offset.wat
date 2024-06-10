;;! target = "x86_64"
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
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %r11
;;       movl    %edx, %eax
;;       xorq    %rsi, %rsi
;;       movq    %rax, %r10
;;       addq    0x60(%rdi), %r10
;;       cmpq    %r11, %rax
;;       cmovaeq %rsi, %r10
;;       movb    %cl, (%r10)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movq    0x68(%rdi), %rsi
;;       movl    %edx, %ecx
;;       xorq    %rax, %rax
;;       movq    %rcx, %r11
;;       addq    0x60(%rdi), %r11
;;       cmpq    %rsi, %rcx
;;       cmovaeq %rax, %r11
;;       movzbq  (%r11), %rdi
;;       movzbl  %dil, %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
