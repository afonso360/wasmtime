;;! target = "riscv64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=false']
;;!
;;! compile = true
;;!
;;! [globals.vmctx]
;;! type = "i64"
;;! vmctx = true
;;!
;;! [globals.heap_base]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 0, readonly = true }
;;!
;;! # (no heap_bound global for static heaps)
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i64"
;;! style = { kind = "static", bound = 0x10000000 }

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

;; function u0:0:
;; block0:
;;   lui a3,65535
;;   addi a3,a3,-1
;;   ugt a3,a0,a3##ty=i64
;;   bne a3,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld a3,0(a2)
;;   add a3,a3,a0
;;   sb a1,4096(a3)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   lui a2,65535
;;   addi a3,a2,-1
;;   ugt a2,a0,a3##ty=i64
;;   bne a2,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld a3,0(a1)
;;   add a3,a3,a0
;;   lbu a0,4096(a3)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
