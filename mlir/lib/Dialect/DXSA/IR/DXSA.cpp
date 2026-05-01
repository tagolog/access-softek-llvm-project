//===--------------- DXSA.cpp - MLIR DXSA Operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DXSA/IR/DXSA.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dxsa;

#include "mlir/Dialect/DXSA/IR/DXSAOpsDialect.cpp.inc"
#include "mlir/Dialect/DXSA/IR/DXSAOpsEnums.cpp.inc"

void DXSADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/DXSA/IR/DXSAOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/DXSA/IR/DXSAOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/DXSA/IR/DXSAOpsAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DclGlobalFlags
//===----------------------------------------------------------------------===//

LogicalResult DclGlobalFlags::verify() {
  if (getFlags() == GlobalFlags::none)
    return emitOpError("expected at least one global flag to be set");
  return success();
}

//===----------------------------------------------------------------------===//
// Custom assembly format helpers
//===----------------------------------------------------------------------===//

namespace {

/// Parse a register identifier of the form `<prefix><N>` (e.g. `x0`, `r3`).
/// The whole identifier must be a single token (no whitespace between
/// prefix and number).
ParseResult parseRegisterName(OpAsmParser &parser, StringRef registerPrefix,
                              IntegerAttr &registerIndex) {
  StringRef name;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&name))
    return failure();
  if (!name.consume_front(registerPrefix))
    return parser.emitError(loc)
           << "expected register prefix '" << registerPrefix << "'";
  unsigned value;
  if (name.getAsInteger(10, value))
    return parser.emitError(loc) << "expected integer after register prefix '"
                                 << registerPrefix << "'";
  registerIndex = parser.getBuilder().getI32IntegerAttr(value);
  return success();
}

/// Print a register identifier of the form `<prefix><N>`.
void printRegisterName(OpAsmPrinter &printer, StringRef registerPrefix,
                       IntegerAttr registerIndex) {
  printer << registerPrefix << registerIndex.getInt();
}

//===----------------------------------------------------------------------===//
// Custom parser/printer for indexable temp registers (dcl_indexable_temp op)
//===----------------------------------------------------------------------===//

ParseResult parseIndexableTempRegister(OpAsmParser &parser,
                                       IntegerAttr &registerIndex) {
  return parseRegisterName(parser, "x", registerIndex);
}

void printIndexableTempRegister(OpAsmPrinter &printer, Operation *op,
                                IntegerAttr registerIndex) {
  printRegisterName(printer, "x", registerIndex);
}

//===----------------------------------------------------------------------===//
// Custom parser/printer for contiguous component masks (xyzw-style)
//===----------------------------------------------------------------------===//

static constexpr StringRef contiguousMasks[] = {"x", "xy", "xyz", "xyzw"};

ParseResult parseContiguousMask(OpAsmParser &parser,
                                IntegerAttr &numComponents) {
  StringRef name;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&name))
    return failure();
  for (auto [i, contiguousMask] : llvm::enumerate(contiguousMasks)) {
    if (name == contiguousMask) {
      numComponents = parser.getBuilder().getI32IntegerAttr(i + 1);
      return success();
    }
  }
  return parser.emitError(loc) << "expected one of 'x', 'xy', 'xyz', 'xyzw'";
}

void printContiguousMask(OpAsmPrinter &printer, Operation *op,
                         IntegerAttr numComponents) {
  unsigned n = numComponents.getInt();
  assert(n >= 1 && n <= std::size(contiguousMasks) &&
         "num_components out of range");
  printer << contiguousMasks[n - 1];
}

} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/DXSA/IR/DXSAOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd attribute method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/DXSA/IR/DXSAOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/DXSA/IR/DXSAOpsTypes.cpp.inc"
