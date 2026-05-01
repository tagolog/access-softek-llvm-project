// RUN: mlir-translate --import-dxsa-bin %S/inputs/dcl_indexable_temp.bin | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   dxsa.dcl_indexable_temp x0[1], x
// CHECK-NEXT:   dxsa.dcl_indexable_temp x1[23], xy
// CHECK-NEXT:   dxsa.dcl_indexable_temp x2[16], xyzw
// CHECK-NEXT:   dxsa.dcl_indexable_temp x7[4096], xyzw
// CHECK-NEXT: }
