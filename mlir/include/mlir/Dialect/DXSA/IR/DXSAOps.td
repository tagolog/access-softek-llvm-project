//===- DXSAOps.td - DXSA operations definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DXSA_OPS
#define DXSA_OPS

include "mlir/Dialect/DXSA/IR/DXSADialect.td"
include "mlir/Dialect/DXSA/IR/DXSATypes.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DXSA enum definitions
//===----------------------------------------------------------------------===//

// Global shader flags for 'dcl_globalFlags' instruction.
def DXSA_GlobalFlags_None               : I32BitEnumAttrCaseNone<"none">;
// D3D10_SB_GLOBAL_FLAG_REFACTORING_ALLOWED
def DXSA_GlobalFlags_RefactoringAllowed : I32BitEnumAttrCaseBit<"refactoringAllowed", 11>;
// D3D11_SB_GLOBAL_FLAG_ENABLE_DOUBLE_PRECISION_FLOAT_OPS
def DXSA_GlobalFlags_DoublePrecision    : I32BitEnumAttrCaseBit<"enableDoublePrecision", 12>;
// D3D11_SB_GLOBAL_FLAG_FORCE_EARLY_DEPTH_STENCIL
def DXSA_GlobalFlags_ForceEarlyDepth    : I32BitEnumAttrCaseBit<"forceEarlyDepth", 13>;
// D3D11_SB_GLOBAL_FLAG_ENABLE_RAW_AND_STRUCTURED_BUFFERS
def DXSA_GlobalFlags_RawStructured      : I32BitEnumAttrCaseBit<"enableRawStructured", 14>;
// D3D11_1_SB_GLOBAL_FLAG_SKIP_OPTIMIZATION
def DXSA_GlobalFlags_SkipOptimization   : I32BitEnumAttrCaseBit<"skipOptimization", 15>;
// D3D11_1_SB_GLOBAL_FLAG_ENABLE_MINIMUM_PRECISION
def DXSA_GlobalFlags_MinPrecision       : I32BitEnumAttrCaseBit<"enableMinPrecision", 16>;
// D3D11_1_SB_GLOBAL_FLAG_ENABLE_DOUBLE_EXTENSIONS
def DXSA_GlobalFlags_DoubleExtensions   : I32BitEnumAttrCaseBit<"enableDoubleExtensions", 17>;
// D3D11_1_SB_GLOBAL_FLAG_ENABLE_SHADER_EXTENSIONS
def DXSA_GlobalFlags_ShaderExtensions   : I32BitEnumAttrCaseBit<"enableShaderExtensions", 18>;
// D3D12_SB_GLOBAL_FLAG_ALL_RESOURCES_BOUND
def DXSA_GlobalFlags_AllResourcesBound  : I32BitEnumAttrCaseBit<"allResourcesBound", 19>;

def DXSA_GlobalFlags : I32BitEnumAttr<
    "GlobalFlags", "global shader flags", [
      DXSA_GlobalFlags_None,
      DXSA_GlobalFlags_RefactoringAllowed,
      DXSA_GlobalFlags_DoublePrecision,
      DXSA_GlobalFlags_ForceEarlyDepth,
      DXSA_GlobalFlags_RawStructured,
      DXSA_GlobalFlags_SkipOptimization,
      DXSA_GlobalFlags_MinPrecision,
      DXSA_GlobalFlags_DoubleExtensions,
      DXSA_GlobalFlags_ShaderExtensions,
      DXSA_GlobalFlags_AllResourcesBound
    ]> {
  let separator = "|";
  let cppNamespace = "::mlir::dxsa";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def DXSA_GlobalFlagsAttr :
    EnumAttr<DXSADialect, DXSA_GlobalFlags, "global_flags"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// DXSA op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class DXSA_Op<string mnemonic, list<Trait> traits = []> :
    Op<DXSADialect, mnemonic, traits>;

def DXSA_Operand : DXSA_Op<"operand"> {
  let summary = "defines an operand of an instruction";
  let description = [{
    TODO
  }];

  let arguments = (ins Variadic<DXSA_IndexType>:$operands,
                       I32Attr:$type,
                       I32Attr:$num_components,
		       OptionalAttr<I32Attr>:$one,
		       OptionalAttr<I32Attr>:$mask,
		       OptionalAttr<I32ElementsAttr>:$swizzle,
		       OptionalAttr<I32Attr>:$modifier,
		       OptionalAttr<I32Attr>:$min_precision,
		       OptionalAttr<I32Attr>:$non_uniform);
  let results = (outs DXSA_OperandType:$operand);
  let assemblyFormat = "$operands attr-dict";
}

def DXSA_OperandImm : DXSA_Op<"operand.imm"> {
  let summary = "defines an immediate operand of an instruction";
  let description = [{
    TODO
  }];

  let arguments = (ins AnyAttrOf<[I32ElementsAttr, I64ElementsAttr]>:$imm);
  let results = (outs DXSA_OperandType:$operand);
  let assemblyFormat = "attr-dict";
}

def DXSA_IndexImm : DXSA_Op<"index.imm"> {
  let summary = "defines an immediate index";
  let description = [{
    TODO
  }];

  let arguments = (ins AnyAttrOf<[I32Attr, I64Attr]>:$imm);
  let results = (outs DXSA_IndexType:$index);
  let assemblyFormat = "attr-dict";
}

def DXSA_IndexRel : DXSA_Op<"index.rel"> {
  let summary = "defines an operand index";
  let description = [{
    TODO
  }];

  let arguments = (ins DXSA_OperandType:$operand);
  let results = (outs DXSA_IndexType:$index);
  let assemblyFormat = "$operand attr-dict";
}

def DXSA_IndexRelImm : DXSA_Op<"index.rel.imm"> {
  let summary = "defines an operand index with an immediate attribute";
  let description = [{
    TODO
  }];

  let arguments = (ins DXSA_OperandType:$operand, StrAttr:$op, I64Attr:$imm);
  let results = (outs DXSA_IndexType:$index);
  let assemblyFormat = "$operand attr-dict";
}

def DXSA_Instruction : DXSA_Op<"instruction"> {
  let summary = "defines an instruction with zero or more operands";
  let description = [{
    TODO
  }];

  let arguments = (ins Variadic<DXSA_OperandType>:$operands, StrAttr:$mnemonic);
  let results = (outs);
  let assemblyFormat = "$mnemonic $operands attr-dict";
}

def DXSA_DclGlobalFlags : DXSA_Op<"dcl_global_flags"> {
  let summary = "declares global shader flags";
  let arguments = (ins DXSA_GlobalFlagsAttr:$flags);
  let assemblyFormat = "$flags attr-dict";
  let hasVerifier = 1;
}

def DXSA_DclTemps : DXSA_Op<"dcl_temps"> {
  let summary = "declares the number of temporary registers";
  let arguments = (ins ConfinedAttr<I32Attr,
      [IntPositive, IntMaxValue<4096>]>:$count);
  let assemblyFormat = [{ $count attr-dict }];
}

def DXSA_DclIndexableTemp : DXSA_Op<"dcl_indexable_temp"> {
  let summary = "declare a temporary register";
  let arguments = (ins I32Attr:$register_index,
      ConfinedAttr<I32Attr, [IntPositive, IntMaxValue<4096>]>:$register_count,
      ConfinedAttr<I32Attr, [IntMinValue<1>, IntMaxValue<4>]>:$num_components);
  let assemblyFormat = [{
    custom<IndexableTempRegister>($register_index) `[` $register_count `]` `,`
    custom<ContiguousMask>($num_components) attr-dict
  }];
}

#endif // DXSA_OPS
