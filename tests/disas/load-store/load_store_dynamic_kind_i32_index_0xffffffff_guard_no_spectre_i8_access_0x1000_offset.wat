;;! target = "x86_64"
;;! test = "clif"
;;! flags = " -C cranelift-enable-heap-access-spectre-mitigation=false -O static-memory-maximum-size=0 -O static-memory-guard-size=4294967295 -O dynamic-memory-guard-size=4294967295"

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

;; function u0:0(i64 vmctx, i64, i32, i32) tail {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0+8
;;     gv2 = load.i64 notrap aligned gv1
;;     gv3 = vmctx
;;     gv4 = load.i64 notrap aligned gv3+104
;;     gv5 = load.i64 notrap aligned checked gv3+96
;;     stack_limit = gv2
;;
;;                                 block0(v0: i64, v1: i64, v2: i32, v3: i32):
;; @0040                               v4 = ireduce.i8 v3
;; @0040                               v5 = uextend.i64 v2
;; @0040                               v6 = global_value.i64 gv4
;; @0040                               v7 = icmp ugt v5, v6
;; @0040                               trapnz v7, heap_oob
;; @0040                               v8 = global_value.i64 gv5
;; @0040                               v9 = iadd v8, v5
;; @0040                               v10 = iconst.i64 4096
;; @0040                               v11 = iadd v9, v10  ; v10 = 4096
;; @0040                               store little heap v4, v11
;; @0044                               jump block1
;;
;;                                 block1:
;; @0044                               return
;; }
;;
;; function u0:1(i64 vmctx, i64, i32) -> i32 tail {
;;     gv0 = vmctx
;;     gv1 = load.i64 notrap aligned readonly gv0+8
;;     gv2 = load.i64 notrap aligned gv1
;;     gv3 = vmctx
;;     gv4 = load.i64 notrap aligned gv3+104
;;     gv5 = load.i64 notrap aligned checked gv3+96
;;     stack_limit = gv2
;;
;;                                 block0(v0: i64, v1: i64, v2: i32):
;; @0049                               v4 = uextend.i64 v2
;; @0049                               v5 = global_value.i64 gv4
;; @0049                               v6 = icmp ugt v4, v5
;; @0049                               trapnz v6, heap_oob
;; @0049                               v7 = global_value.i64 gv5
;; @0049                               v8 = iadd v7, v4
;; @0049                               v9 = iconst.i64 4096
;; @0049                               v10 = iadd v8, v9  ; v9 = 4096
;; @0049                               v11 = uload8.i32 little heap v10
;; @004d                               jump block1(v11)
;;
;;                                 block1(v3: i32):
;; @004d                               return v3
;; }
