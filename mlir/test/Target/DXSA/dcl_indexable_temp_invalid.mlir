// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error@+1 {{attribute 'register_count' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive whose maximum value is 4096}}
dxsa.dcl_indexable_temp x0[0], xy

// -----

// expected-error@+1 {{attribute 'register_count' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive whose maximum value is 4096}}
dxsa.dcl_indexable_temp x0[4097], xy

// -----

// expected-error@+1 {{expected one of 'x', 'xy', 'xyz', 'xyzw'}}
dxsa.dcl_indexable_temp x0[16], xz

// -----

// expected-error@+1 {{expected register prefix 'x'}}
dxsa.dcl_indexable_temp r0[16], xyzw

// -----

// expected-error@+1 {{expected integer after register prefix 'x'}}
dxsa.dcl_indexable_temp xfoo[16], xyzw
