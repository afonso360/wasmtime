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
    i32.store8 offset=0xffff0000)

  (func (export "do_load") (param i32) (result i32)
    local.get 0
    i32.load8_u offset=0xffff0000))

;; wasm[0]::function[0]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r8d
;;       movq    %r8, %r11
;;       addq    0x2f(%rip), %r11
;;       jb      0x38
;;   17: movq    0x68(%rdi), %rsi
;;       cmpq    %rsi, %r11
;;       ja      0x36
;;   24: addq    0x60(%rdi), %r8
;;       movl    $0xffff0000, %eax
;;       movb    %cl, (%r8, %rax)
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   36: ud2
;;   38: ud2
;;   3a: addb    %al, (%rax)
;;   3c: addb    %al, (%rax)
;;   3e: addb    %al, (%rax)
;;   40: addl    %eax, (%rax)
;;
;; wasm[0]::function[1]:
;;       pushq   %rbp
;;       movq    %rsp, %rbp
;;       movl    %edx, %r9d
;;       movq    %r9, %rsi
;;       addq    0x2f(%rip), %rsi
;;       jb      0x9c
;;   77: movq    0x68(%rdi), %rax
;;       cmpq    %rax, %rsi
;;       ja      0x9a
;;   84: addq    0x60(%rdi), %r9
;;       movl    $0xffff0000, %ecx
;;       movzbq  (%r9, %rcx), %rcx
;;       movzbl  %cl, %eax
;;       movq    %rbp, %rsp
;;       popq    %rbp
;;       retq
;;   9a: ud2
;;   9c: ud2
;;   9e: addb    %al, (%rax)
;;   a0: addl    %eax, (%rax)
