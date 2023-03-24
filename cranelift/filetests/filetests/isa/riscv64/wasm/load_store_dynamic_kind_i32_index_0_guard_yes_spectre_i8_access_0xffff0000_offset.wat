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
;;! index_type = "i32"
;;! style = { kind = "dynamic", bound = "heap_bound" }

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

;; function u0:0:
;; block0:
;;   slli t2,a0,32
;;   srli a3,t2,32
;;   auipc t1,0; ld t1,12(t1); j 12; .8byte 0xffff0001
;;   add a0,a3,t1
;;   sltu a4,a0,a3
;;   trap_if a4,heap_oob
;;   ld a4,8(a2)
;;   ld a2,0(a2)
;;   add a2,a2,a3
;;   auipc a3,0; ld a3,12(a3); j 12; .8byte 0xffff0000
;;   add a2,a2,a3
;;   sltu a0,a4,a0
;;   li a3,0
;;   selectif_spectre_guard a4,a3,a2##test=a0
;;   sb a1,0(a4)
;;   j label1
;; block1:
;;   ret
;;
;; function u0:1:
;; block0:
;;   slli t2,a0,32
;;   srli a2,t2,32
;;   auipc t1,0; ld t1,12(t1); j 12; .8byte 0xffff0001
;;   add a0,a2,t1
;;   sltu a3,a0,a2
;;   trap_if a3,heap_oob
;;   ld a3,8(a1)
;;   ld a1,0(a1)
;;   add a1,a1,a2
;;   auipc a2,0; ld a2,12(a2); j 12; .8byte 0xffff0000
;;   add a2,a1,a2
;;   sltu a0,a3,a0
;;   li a3,0
;;   selectif_spectre_guard a1,a3,a2##test=a0
;;   lbu a0,0(a1)
;;   j label1
;; block1:
;;   ret
