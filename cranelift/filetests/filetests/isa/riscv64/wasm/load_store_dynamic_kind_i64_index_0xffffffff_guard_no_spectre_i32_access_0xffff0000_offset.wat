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
;;! [globals.heap_bound]
;;! type = "i64"
;;! load = { base = "vmctx", offset = 8, readonly = true }
;;!
;;! [[heaps]]
;;! base = "heap_base"
;;! min_size = 0x10000
;;! offset_guard_size = 0xffffffff
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
    i32.store offset=0xffff0000)

  (func (export "do_load") (param i64) (result i32)
    local.get 0
    i32.load offset=0xffff0000))

;; function u0:0:
;; block0:
;;   ld a6,8(a2)
;;   sltu a6,a6,a0
;;   bne a6,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld a7,0(a2)
;;   add a7,a7,a0
;;   auipc a6,0; ld a6,12(a6); j 12; .8byte 0xffff0000
;;   add t3,a7,a6
;;   sw a1,0(t3)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
;;
;; function u0:1:
;; block0:
;;   ld a6,8(a1)
;;   sltu a6,a6,a0
;;   bne a6,zero,taken(label3),not_taken(label1)
;; block1:
;;   ld a7,0(a1)
;;   add a7,a7,a0
;;   auipc a6,0; ld a6,12(a6); j 12; .8byte 0xffff0000
;;   add t3,a7,a6
;;   lw a0,0(t3)
;;   j label2
;; block2:
;;   ret
;; block3:
;;   udf##trap_code=heap_oob
