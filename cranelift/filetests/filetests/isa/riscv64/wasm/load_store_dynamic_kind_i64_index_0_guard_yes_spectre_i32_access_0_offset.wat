;;! target = "riscv64"
;;!
;;! settings = ['enable_heap_access_spectre_mitigation=true']
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
;;! [globals.heap_bound]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 8, readonly = true }
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0
;;! index_type = "i64"
;;! style = { kind = "dynamic", bound = "heap_bound" }

;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
;; !!! GENERATED BY 'make-load-store-tests.sh' DO NOT EDIT !!!
;; !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(module
  (memory i64 1)

  (func (export "do_store") (param i64 i32)
    local.get 0
    local.get 1
    i32.store offset=0)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0))

;; function u0:0:
;; block0:
;;   ld a7,8(a2)
;;   addi a7,a7,-4
;;   ld t3,0(a2)
;;   add t3,t3,a0
;;   sltu a6,a7,a0
;;   li t4,0
;;   selectif_spectre_guard a7,t4,t3##test=a6
;;   sw a1,0(a7)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   ld a7,8(a1)
;;   addi a7,a7,-4
;;   ld t3,0(a1)
;;   add t3,t3,a0
;;   sltu a6,a7,a0
;;   li t4,0
;;   selectif_spectre_guard a7,t4,t3##test=a6
;;   lw a0,0(a7)
;;   j label1
;; block1:
;;   ret
