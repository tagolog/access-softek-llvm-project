//===----------- BinaryParser.cpp - Parse DXSA binary to MLIR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/DXSA/BinaryParser.h"
#include "mlir/Dialect/DXSA/IR/DXSA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

#include "d3d12TokenizedProgramFormat.hpp"

#define DEBUG_TYPE "import-dxsa-bin"

using namespace mlir;
using namespace llvm;

enum OpcodeClass {
  D3D10_SB_FLOAT_OP,
  D3D10_SB_INT_OP,
  D3D10_SB_UINT_OP,
  D3D10_SB_BIT_OP,
  D3D10_SB_FLOW_OP,
  D3D10_SB_TEX_OP,
  D3D10_SB_DCL_OP,
  D3D11_SB_ATOMIC_OP,
  D3D11_SB_MEM_OP,
  D3D11_SB_DOUBLE_OP,
  D3D11_SB_FLOAT_TO_DOUBLE_OP,
  D3D11_SB_DOUBLE_TO_FLOAT_OP,
  D3D11_SB_DEBUG_OP,
};

struct InstructionInfo {
  unsigned numOperands;
  StringRef name;
  OpcodeClass opClass;
  uint32_t precisionFromOutMask;
};

static void initInstructionInfo(MutableArrayRef<InstructionInfo> instructions) {
#define SET(OpCode, Name, NumOperands, PrecMask, OpClass)                      \
  instructions[OpCode] = InstructionInfo{NumOperands, Name, OpClass, PrecMask};
  // clang-format off
  SET(D3D10_SB_OPCODE_ADD, "add", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_AND, "and", 3, 0x06, D3D10_SB_BIT_OP);
  SET(D3D10_SB_OPCODE_BREAK, "break", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_BREAKC, "breakc", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CALL, "call", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CALLC, "callc", 2, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CONTINUE, "continue", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CONTINUEC, "continuec", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CASE, "case", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_CUT, "cut", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_DEFAULT, "default", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_DISCARD, "discard", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_DIV, "div", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_DP2, "dp2", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_DP3, "dp3", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_DP4, "dp4", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_ELSE, "else", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_EMIT, "emit", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_EMITTHENCUT, "emit_then_cut", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_ENDIF, "endif", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_ENDLOOP, "endloop", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_ENDSWITCH, "endswitch", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_EQ, "eq", 3, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_EXP, "exp", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_FRC, "frc", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_FTOI, "ftoi", 2, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_FTOU, "ftou", 2, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_GE, "ge", 3, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_DERIV_RTX, "deriv_rtx", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_DERIV_RTY, "deriv_rty", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_IADD, "iadd", 3, 0x06, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IF, "if", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_IEQ, "ieq", 3, 0x00, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IGE, "ige", 3, 0x00, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_ILT, "ilt", 3, 0x00, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IMAD, "imad", 4, 0x0e, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IMAX, "imax", 3, 0x06, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IMIN, "imin", 3, 0x06, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_IMUL, "imul", 4, 0x0c, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_INE, "ine", 3, 0x00, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_INEG, "ineg", 2, 0x02, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_ISHL, "ishl", 3, 0x02, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_ISHR, "ishr", 3, 0x02, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_ITOF, "itof", 2, 0x00, D3D10_SB_INT_OP);
  SET(D3D10_SB_OPCODE_LABEL, "label", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_LD, "ld", 3, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_LD_MS, "ldms", 4, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_LOG, "log", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_LOOP, "loop", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_LT, "lt", 3, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MAD, "mad", 4, 0x0e, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MAX, "max", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MIN, "min", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MOV, "mov", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MOVC, "movc", 4, 0x0c, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_MUL, "mul", 3, 0x06, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_NE, "ne", 3, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_NOP, "nop", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_NOT, "not", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D10_SB_OPCODE_OR, "or", 3, 0x06, D3D10_SB_BIT_OP);
  SET(D3D10_SB_OPCODE_RESINFO, "resinfo", 3, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_RET, "ret", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_RETC, "retc", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_ROUND_NE, "round_ne", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_ROUND_NI, "round_ni", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_ROUND_PI, "round_pi", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_ROUND_Z, "round_z", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_RSQ, "rsq", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_SAMPLE, "sample", 4, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SAMPLE_B, "sample_b", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SAMPLE_L, "sample_l", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SAMPLE_D, "sample_d", 6, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SAMPLE_C, "sample_c", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SAMPLE_C_LZ, "sample_c_lz", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_SB_OPCODE_SQRT, "sqrt", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_SWITCH, "switch", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_SINCOS, "sincos", 3, 0x04, D3D10_SB_FLOAT_OP);
  SET(D3D10_SB_OPCODE_UDIV, "udiv", 4, 0x0c, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_ULT, "ult", 3, 0x00, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UGE, "uge", 3, 0x00, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UMAX, "umax", 3, 0x06, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UMIN, "umin", 3, 0x06, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UMUL, "umul", 4, 0x0c, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UMAD, "umad", 4, 0x0e, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_USHR, "ushr", 3, 0x02, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_UTOF, "utof", 2, 0x00, D3D10_SB_UINT_OP);
  SET(D3D10_SB_OPCODE_XOR, "xor", 3, 0x06, D3D10_SB_BIT_OP);
  SET(D3D10_SB_OPCODE_RESERVED0, "jmp", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT, "dcl_input", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_OUTPUT, "dcl_output", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT_SGV, "dcl_input_sgv", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT_PS_SGV, "dcl_input_ps_sgv", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_GS_INPUT_PRIMITIVE, "dcl_inputprimitive", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_GS_OUTPUT_PRIMITIVE_TOPOLOGY, "dcl_outputtopology", 0,
      0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_MAX_OUTPUT_VERTEX_COUNT, "dcl_maxout", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT_PS, "dcl_input_ps", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER, "dcl_constantbuffer", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_SAMPLER, "dcl_sampler", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_RESOURCE, "dcl_resource", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT_SIV, "dcl_input_siv", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INPUT_PS_SIV, "dcl_input_ps_siv", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_OUTPUT_SIV, "dcl_output_siv", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_OUTPUT_SGV, "dcl_output_sgv", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_TEMPS, "dcl_temps", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP, "dcl_indexableTemp", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_INDEX_RANGE, "dcl_indexrange", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS, "dcl_globalFlags", 0, 0x00,
      D3D10_SB_DCL_OP);

  SET(D3D10_1_SB_OPCODE_SAMPLE_INFO, "sampleinfo", 2, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_1_SB_OPCODE_SAMPLE_POS, "samplepos", 3, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_1_SB_OPCODE_GATHER4, "gather4", 4, 0x00, D3D10_SB_TEX_OP);
  SET(D3D10_1_SB_OPCODE_LOD, "lod", 4, 0x00, D3D10_SB_TEX_OP);

  SET(D3D11_SB_OPCODE_EMIT_STREAM, "emit_stream", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D11_SB_OPCODE_CUT_STREAM, "cut_stream", 1, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D11_SB_OPCODE_EMITTHENCUT_STREAM, "emit_then_cut_stream", 1, 0x00,
      D3D10_SB_FLOW_OP);
  SET(D3D11_SB_OPCODE_INTERFACE_CALL, "fcall", 1, 0x00, D3D10_SB_FLOW_OP);

  SET(D3D11_SB_OPCODE_DCL_STREAM, "dcl_stream", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_FUNCTION_BODY, "dcl_function_body", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_FUNCTION_TABLE, "dcl_function_table", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_INTERFACE, "dcl_interface", 0, 0x00, D3D10_SB_DCL_OP);

  SET(D3D11_SB_OPCODE_BUFINFO, "bufinfo", 2, 0x00, D3D10_SB_TEX_OP);
  SET(D3D11_SB_OPCODE_DERIV_RTX_COARSE, "deriv_rtx_coarse", 2, 0x02,
      D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_DERIV_RTX_FINE, "deriv_rtx_fine", 2, 0x02,
      D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_DERIV_RTY_COARSE, "deriv_rty_coarse", 2, 0x02,
      D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_DERIV_RTY_FINE, "deriv_rty_fine", 2, 0x02,
      D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_GATHER4_C, "gather4_c", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D11_SB_OPCODE_GATHER4_PO, "gather4_po", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3D11_SB_OPCODE_GATHER4_PO_C, "gather4_po_c", 6, 0x00, D3D10_SB_TEX_OP);
  SET(D3D11_SB_OPCODE_RCP, "rcp", 2, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_F32TOF16, "f32tof16", 2, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_F16TOF32, "f16tof32", 2, 0x00, D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_UADDC, "uaddc", 4, 0x0c, D3D10_SB_UINT_OP);
  SET(D3D11_SB_OPCODE_USUBB, "usubb", 4, 0x0c, D3D10_SB_UINT_OP);
  SET(D3D11_SB_OPCODE_COUNTBITS, "countbits", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_FIRSTBIT_HI, "firstbit_hi", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_FIRSTBIT_LO, "firstbit_lo", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_FIRSTBIT_SHI, "firstbit_shi", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_UBFE, "ubfe", 4, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_IBFE, "ibfe", 4, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_BFI, "bfi", 5, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_BFREV, "bfrev", 2, 0x02, D3D10_SB_BIT_OP);
  SET(D3D11_SB_OPCODE_SWAPC, "swapc", 5, 0x02, D3D10_SB_FLOAT_OP);

  SET(D3D11_SB_OPCODE_HS_DECLS, "hs_decls", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_HS_CONTROL_POINT_PHASE, "hs_control_point_phase", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_HS_FORK_PHASE, "hs_fork_phase", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_HS_JOIN_PHASE, "hs_join_phase", 0, 0x00, D3D10_SB_DCL_OP);

  SET(D3D11_SB_OPCODE_DCL_INPUT_CONTROL_POINT_COUNT,
      "dcl_input_control_point_count", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_OUTPUT_CONTROL_POINT_COUNT,
      "dcl_output_control_point_count", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_TESS_DOMAIN, "dcl_tessellator_domain", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_TESS_PARTITIONING, "dcl_tessellator_partitioning", 0,
      0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_TESS_OUTPUT_PRIMITIVE,
      "dcl_tessellator_output_primitive", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_HS_MAX_TESSFACTOR, "dcl_hs_max_tessfactor", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_HS_FORK_PHASE_INSTANCE_COUNT,
      "dcl_hs_fork_phase_instance_count", 0, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_HS_JOIN_PHASE_INSTANCE_COUNT,
      "dcl_hs_join_phase_instance_count", 0, 0x00, D3D10_SB_DCL_OP);

  SET(D3D11_SB_OPCODE_DCL_THREAD_GROUP, "dcl_thread_group", 0, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_TYPED, "dcl_uav_typed", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_RAW, "dcl_uav_raw", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_UNORDERED_ACCESS_VIEW_STRUCTURED,
      "dcl_uav_structured", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_RAW, "dcl_tgsm_raw", 1,
      0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_THREAD_GROUP_SHARED_MEMORY_STRUCTURED,
      "dcl_tgsm_structured", 1, 0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_RESOURCE_RAW, "dcl_resource_raw", 1, 0x00,
      D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED, "dcl_resource_structured", 1,
      0x00, D3D10_SB_DCL_OP);
  SET(D3D11_SB_OPCODE_LD_UAV_TYPED, "ld_uav_typed", 3, 0x00, D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_STORE_UAV_TYPED, "store_uav_typed", 3, 0x00,
      D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_LD_RAW, "ld_raw", 3, 0x00, D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_STORE_RAW, "store_raw", 3, 0x00, D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_LD_STRUCTURED, "ld_structured", 4, 0x00, D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_STORE_STRUCTURED, "store_structured", 4, 0x00,
      D3D11_SB_MEM_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_AND, "atomic_and", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_OR, "atomic_or", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_XOR, "atomic_xor", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_CMP_STORE, "atomic_cmp_store", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_IADD, "atomic_iadd", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_IMAX, "atomic_imax", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_IMIN, "atomic_imin", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_UMAX, "atomic_umax", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_ATOMIC_UMIN, "atomic_umin", 3, 0x00, D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_ALLOC, "imm_atomic_alloc", 2, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_CONSUME, "imm_atomic_consume", 2, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_IADD, "imm_atomic_iadd", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_AND, "imm_atomic_and", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_OR, "imm_atomic_or", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_XOR, "imm_atomic_xor", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_EXCH, "imm_atomic_exch", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_CMP_EXCH, "imm_atomic_cmp_exch", 5, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_IMAX, "imm_atomic_imax", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_IMIN, "imm_atomic_imin", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_UMAX, "imm_atomic_umax", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_IMM_ATOMIC_UMIN, "imm_atomic_umin", 4, 0x00,
      D3D11_SB_ATOMIC_OP);
  SET(D3D11_SB_OPCODE_SYNC, "sync", 0, 0x00, D3D10_SB_FLOW_OP);
  SET(D3D11_SB_OPCODE_EVAL_SNAPPED, "eval_snapped", 3, 0x02, D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_EVAL_SAMPLE_INDEX, "eval_sample_index", 3, 0x02,
      D3D10_SB_FLOAT_OP);
  SET(D3D11_SB_OPCODE_EVAL_CENTROID, "eval_centroid", 2, 0x02,
      D3D10_SB_FLOAT_OP);

  SET(D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT, "dcl_gsinstances", 0, 0x00,
      D3D10_SB_DCL_OP);

  SET(D3D11_SB_OPCODE_DADD, "dadd", 3, 0x06, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DMAX, "dmax", 3, 0x06, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DMIN, "dmin", 3, 0x06, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DMUL, "dmul", 3, 0x06, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DEQ, "deq", 3, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DGE, "dge", 3, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DLT, "dlt", 3, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DNE, "dne", 3, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DMOV, "dmov", 2, 0x02, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DMOVC, "dmovc", 4, 0x0c, D3D11_SB_DOUBLE_OP);
  SET(D3D11_SB_OPCODE_DTOF, "dtof", 2, 0x02, D3D11_SB_DOUBLE_TO_FLOAT_OP);
  SET(D3D11_SB_OPCODE_FTOD, "ftod", 2, 0x00, D3D11_SB_FLOAT_TO_DOUBLE_OP);

  SET(D3D11_SB_OPCODE_ABORT, "abort", 0, 0x00, D3D11_SB_DEBUG_OP);
  SET(D3D11_SB_OPCODE_DEBUG_BREAK, "debug_break", 0, 0x00, D3D11_SB_DEBUG_OP);

  SET(D3D11_1_SB_OPCODE_DDIV, "ddiv", 3, 0x06, D3D11_SB_DOUBLE_OP);
  SET(D3D11_1_SB_OPCODE_DFMA, "dfma", 4, 0x0e, D3D11_SB_DOUBLE_OP);
  SET(D3D11_1_SB_OPCODE_DRCP, "drcp", 2, 0x02, D3D11_SB_DOUBLE_OP);

  SET(D3D11_1_SB_OPCODE_MSAD, "msad", 4, 0x0e, D3D10_SB_UINT_OP);

  SET(D3D11_1_SB_OPCODE_DTOI, "dtoi", 2, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_1_SB_OPCODE_DTOU, "dtou", 2, 0x00, D3D11_SB_DOUBLE_OP);
  SET(D3D11_1_SB_OPCODE_ITOD, "itod", 2, 0x00, D3D10_SB_INT_OP);
  SET(D3D11_1_SB_OPCODE_UTOD, "utod", 2, 0x00, D3D10_SB_UINT_OP);

  SET(D3DWDDM1_3_SB_OPCODE_GATHER4_FEEDBACK, "gather4_s", 5, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_GATHER4_C_FEEDBACK, "gather4_c_s", 6, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_GATHER4_PO_FEEDBACK, "gather4_po_s", 6, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_GATHER4_PO_C_FEEDBACK, "gather4_po_c_s", 7, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_LD_FEEDBACK, "ld_s", 4, 0x00, D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_LD_MS_FEEDBACK, "ldms_s", 5, 0x00, D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_LD_UAV_TYPED_FEEDBACK, "ld_uav_typed_s", 4, 0x00,
      D3D11_SB_MEM_OP);
  SET(D3DWDDM1_3_SB_OPCODE_LD_RAW_FEEDBACK, "ld_raw_s", 4, 0x00,
      D3D11_SB_MEM_OP);
  SET(D3DWDDM1_3_SB_OPCODE_LD_STRUCTURED_FEEDBACK, "ld_structured_s", 5, 0x00,
      D3D11_SB_MEM_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_L_FEEDBACK, "sample_l_s", 6, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_C_LZ_FEEDBACK, "sample_c_lz_s", 6, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_CLAMP_FEEDBACK, "sample_cl_s", 6, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_B_CLAMP_FEEDBACK, "sample_b_cl_s", 7, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_D_CLAMP_FEEDBACK, "sample_d_cl_s", 8, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_SAMPLE_C_CLAMP_FEEDBACK, "sample_c_cl_s", 7, 0x00,
      D3D10_SB_TEX_OP);
  SET(D3DWDDM1_3_SB_OPCODE_CHECK_ACCESS_FULLY_MAPPED,
      "check_access_fully_mapped", 2, 0x00, D3D10_SB_TEX_OP);
  // clang-format on
}

struct InstructionModifier {
  uint32_t preciseMask{0};
  uint32_t saturate{0};
};

struct OperandModifier {
  uint32_t modifier{0};
  uint32_t minPrecision{0};
  uint32_t nonUniform{0};
};

enum class OperandComponentsKind {
  None,
  Mask,
  Swizzle,
  One,
};

struct OperandComponents {
  unsigned num;
  OperandComponentsKind kind;
  union {
    uint32_t mask;
    uint32_t swizzle[4];
    uint32_t one;
  };
};

class DXBuilder {
public:
  DXBuilder(MLIRContext *context, StringAttr name)
      : context(context),
        module(ModuleOp::create(builder, FileLineColLoc::get(name, 0, 0))),
        builder(module.getRegion()) {}

  using Index = mlir::Value;
  using Operand = mlir::Value;
  using Instruction = mlir::Operation *;
  using Module = mlir::ModuleOp;

  Index buildIndexImm32(int32_t imm, FileLineColLoc loc) {
    Operation *op =
        dxsa::IndexImm::create(builder, loc, builder.getType<dxsa::IndexType>(),
                               builder.getI32IntegerAttr(imm));
    return op->getResults()[0];
  }

  Index buildIndexImm64(int64_t imm, FileLineColLoc loc) {
    Operation *op =
        dxsa::IndexImm::create(builder, loc, builder.getType<dxsa::IndexType>(),
                               builder.getI64IntegerAttr(imm));
    return op->getResults()[0];
  }

  Index buildIndexRelative(Operand operand, FileLineColLoc loc) {
    Operation *op = dxsa::IndexRel::create(
        builder, loc, builder.getType<dxsa::IndexType>(), operand);
    return op->getResults()[0];
  }

  Index buildIndexImm32PlusRelative(int32_t imm, Operand operand,
                                    FileLineColLoc loc) {
    Operation *op = dxsa::IndexRelImm::create(
        builder, loc, builder.getType<dxsa::IndexType>(), operand,
        builder.getStringAttr("add"), builder.getI32IntegerAttr(imm));
    return op->getResults()[0];
  }

  Operand buildOperandImm32(ArrayRef<int32_t> values, FileLineColLoc loc) {
    Operation *op = dxsa::OperandImm::create(
        builder, loc, builder.getType<dxsa::OperandType>(),
        builder.getI32VectorAttr(values));
    return op->getResults()[0];
  }

  Operand buildOperandImm64(ArrayRef<int64_t> values, FileLineColLoc loc) {
    Operation *op = dxsa::OperandImm::create(
        builder, loc, builder.getType<dxsa::OperandType>(),
        builder.getI64VectorAttr(values));
    return op->getResults()[0];
  }

  Operand buildOperand(uint32_t opType, const OperandComponents &components,
                       ArrayRef<Index> indices,
                       const std::optional<OperandModifier> &modifier,
                       FileLineColLoc loc) {
    NamedAttrList attrs;
    attrs.append("type", builder.getI32IntegerAttr(opType));
    if (modifier) {
      const OperandModifier &mod = modifier.value();
      if (uint32_t modifier = mod.modifier) {
        attrs.append("modifier", builder.getI32IntegerAttr(modifier));
      }
      if (uint32_t minPrecision = mod.minPrecision) {
        attrs.append("min_precision", builder.getI32IntegerAttr(minPrecision));
      }
      if (uint32_t nonUniform = mod.nonUniform) {
        attrs.append("non_uniform", builder.getI32IntegerAttr(nonUniform));
      }
    }
    attrs.append("num_components", builder.getI32IntegerAttr(components.num));
    switch (components.kind) {
    case OperandComponentsKind::Mask: {
      attrs.append("mask", builder.getI32IntegerAttr(components.mask));
      break;
    }
    case OperandComponentsKind::Swizzle: {
      SmallVector<int32_t, 4> values;
      for (uint32_t i = 0; i < components.num; ++i) {
        values.push_back(components.swizzle[i]);
      }
      attrs.append("swizzle", builder.getI32VectorAttr(values));
      break;
    }
    case OperandComponentsKind::One: {
      attrs.append("one", builder.getI32IntegerAttr(components.one));
      break;
    }
    case OperandComponentsKind::None:
      break;
    }
    Operation *op = dxsa::Operand::create(
        builder, loc, builder.getType<dxsa::OperandType>(), indices, attrs);
    return op->getResults()[0];
  }

  Instruction buildInstruction(StringRef name, ArrayRef<Operand> operands,
                               const InstructionModifier &modifier,
                               FileLineColLoc loc) {
    return dxsa::Instruction::create(builder, loc, operands,
                                     builder.getStringAttr(name));
  }

  Module buildModule(ArrayRef<Instruction> instructions, FileLineColLoc loc) {
    return module;
  }

  Instruction buildDclGlobalFlags(uint32_t flags, Location loc) {
    auto flagsAttr = dxsa::GlobalFlagsAttr::get(
        builder.getContext(), static_cast<dxsa::GlobalFlags>(flags));
    return dxsa::DclGlobalFlags::create(builder, loc, flagsAttr);
  }

  Instruction buildDclTemps(uint32_t count, Location loc) {
    return dxsa::DclTemps::create(builder, loc,
                                  builder.getI32IntegerAttr(count));
  }

  Instruction buildDclIndexableTemp(uint32_t registerIndex,
                                    uint32_t registerCount,
                                    uint32_t numComponents, Location loc) {
    return dxsa::DclIndexableTemp::create(
        builder, loc, builder.getI32IntegerAttr(registerIndex),
        builder.getI32IntegerAttr(registerCount),
        builder.getI32IntegerAttr(numComponents));
  }

private:
  MLIRContext *context;
  ModuleOp module;
  OpBuilder builder;
};

class Parser {
public:
  Parser(DXBuilder &builder, StringAttr name, StringRef buffer)
      : builder(builder), name(name), buffer(buffer) {
    initInstructionInfo(instrInfo);
  }

  using Token = FailureOr<uint32_t>;
  using Index = DXBuilder::Index;
  using Operand = DXBuilder::Operand;
  using Instruction = DXBuilder::Instruction;
  using Module = DXBuilder::Module;

  /// Parse the current token and move the cursor to the next one.
  Token parseToken() {
    constexpr size_t tokenSize = sizeof(uint32_t);
    if ((currentTokenOffset + tokenSize) > buffer.size()) {
      emitError(getLocation(), "unexpected end of file");
      return failure();
    }

    uint32_t value = support::endian::read<uint32_t>(
        buffer.begin() + currentTokenOffset, endianness::little);
    currentTokenOffset += tokenSize;

    return value;
  }

  /// Returns location where the last parsed token begins (at offset
  /// -4 from the currentTokenOffset).
  FileLineColLoc getLocation(int offset = -4) const {
    return FileLineColLoc::get(name, 0, currentTokenOffset + offset);
  }

  bool isImmOperand(uint32_t token) {
    switch (DECODE_D3D10_SB_OPERAND_TYPE(token)) {
    case D3D10_SB_OPERAND_TYPE_IMMEDIATE32:
    case D3D10_SB_OPERAND_TYPE_IMMEDIATE64:
      return true;
    default:
      return false;
    }
  }

  FailureOr<OperandComponents> parseOperandComponents(uint32_t token) {
    OperandComponents components;
    switch (DECODE_D3D10_SB_OPERAND_NUM_COMPONENTS(token)) {
    case D3D10_SB_OPERAND_0_COMPONENT: {
      components.num = 0;
      break;
    }
    case D3D10_SB_OPERAND_1_COMPONENT: {
      components.num = 1;
      break;
    }
    case D3D10_SB_OPERAND_4_COMPONENT: {
      components.num = 4;
      break;
    }
    default:
      emitError(getLocation(), "unexpected number of components");
      return failure();
    }

    if (components.num != 4 || isImmOperand(token))
      return components;

    switch (DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(token)) {
    case D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE: {
      components.kind = OperandComponentsKind::Mask;
      components.mask = DECODE_D3D10_SB_OPERAND_4_COMPONENT_MASK(token);
      break;
    }
    case D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE: {
      components.kind = OperandComponentsKind::Swizzle;
      components.swizzle[0] =
          DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(token, 0);
      components.swizzle[1] =
          DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(token, 1);
      components.swizzle[2] =
          DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(token, 2);
      components.swizzle[3] =
          DECODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_SOURCE(token, 3);
      break;
    }
    case D3D10_SB_OPERAND_4_COMPONENT_SELECT_1_MODE: {
      components.kind = OperandComponentsKind::One;
      components.one = DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECT_1(token);
      break;
    }
    default:
      emitError(getLocation(), "unexpected component selection");
      return failure();
    }

    return components;
  }

  using OperandIndexTypes = SmallVector<uint32_t, 3>;

  FailureOr<OperandIndexTypes> parseOperandIndexTypes(uint32_t token) {
    SmallVector<uint32_t, 3> indexTypes;
    if (isImmOperand(token))
      return indexTypes; // none

    uint32_t indexDimension = DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(token);
    if (indexDimension > 3) {
      emitError(getLocation(),
                "invalid operand index dimension (must be <= 3)");
      return failure();
    }

    if (indexDimension == D3D10_SB_OPERAND_INDEX_0D)
      return indexTypes; // none

    indexTypes.resize(indexDimension);
    for (unsigned i = 0; i < indexTypes.size(); ++i) {
      indexTypes[i] = DECODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(i, token);
    }

    return indexTypes;
  }

  FailureOr<Index> parseIndex(uint32_t indexType) {
    switch (indexType) {
    case D3D10_SB_OPERAND_INDEX_IMMEDIATE32: {
      Token value = parseToken();
      if (failed(value)) {
        emitError(getLocation(), "expected an operand index imm32");
        return failure();
      }
      return builder.buildIndexImm32(*value, getLocation());
    }
    case D3D10_SB_OPERAND_INDEX_IMMEDIATE64: {
      Token value0 = parseToken();
      if (failed(value0)) {
        emitError(getLocation(), "expected an operand index imm64");
        return failure();
      }

      FileLineColLoc loc = getLocation();

      Token value1 = parseToken();
      if (failed(value1)) {
        emitError(getLocation(),
                  "expected an operand index imm64 (second token)");
        return failure();
      }

      // TODO: check the order of tokens (MSB or LSB?)
      return builder.buildIndexImm64((((uint64_t)*value0) << 32) | *value1,
                                     loc);
    }
    case D3D10_SB_OPERAND_INDEX_RELATIVE: {
      FailureOr<Operand> operand = parseOperand();
      if (failed(operand)) {
        emitError(getLocation(), "expected an index operand");
        return failure();
      }
      return builder.buildIndexRelative(*operand, getLocation());
    }
    case D3D10_SB_OPERAND_INDEX_IMMEDIATE32_PLUS_RELATIVE: {
      Token imm = parseToken();
      if (failed(imm)) {
        emitError(getLocation(), "expected an operand index relative (imm)");
        return failure();
      }

      FileLineColLoc loc = getLocation();

      FailureOr<Operand> operand = parseOperand();
      if (failed(operand)) {
        emitError(getLocation(),
                  "expected an operand index relative (operand)");
        return failure();
      }

      return builder.buildIndexImm32PlusRelative(*imm, *operand, loc);
    }
    default:
      emitError(getLocation(), "invalid operand index type");
      return failure();
    }
  }

  FailureOr<std::optional<OperandModifier>>
  parseOperandExtendedModifier(uint32_t extToken) {
    std::optional<OperandModifier> none;
    if (D3D10_SB_EXTENDED_OPERAND_MODIFIER !=
        DECODE_D3D10_SB_EXTENDED_OPERAND_TYPE(extToken))
      return none;

    OperandModifier modifier;
    modifier.modifier = DECODE_D3D10_SB_OPERAND_MODIFIER(extToken);
    modifier.minPrecision = DECODE_D3D11_SB_OPERAND_MIN_PRECISION(extToken);
    modifier.nonUniform = DECODE_D3D12_SB_OPERAND_NON_UNIFORM(extToken);
    return std::make_optional(modifier);
  }

  FailureOr<Operand> parseOperand() {
    Token token = parseToken();
    if (failed(token))
      return failure();

    FileLineColLoc loc = getLocation();

    uint32_t opType = DECODE_D3D10_SB_OPERAND_TYPE(*token);
    bool isExtended = DECODE_IS_D3D10_SB_OPERAND_EXTENDED(*token);

    FailureOr<OperandComponents> components = parseOperandComponents(*token);
    if (failed(components))
      return failure();

    FailureOr<OperandIndexTypes> indexTypes = parseOperandIndexTypes(*token);
    if (failed(indexTypes))
      return failure();

    std::optional<OperandModifier> modifier;
    if (isExtended) {
      Token extToken = parseToken();
      if (failed(extToken)) {
        emitError(getLocation(), "unexpected an extended operand token");
        return failure();
      }
      auto failureOrModifier = parseOperandExtendedModifier(*extToken);
      if (failed(failureOrModifier))
        return failure();
      modifier = *failureOrModifier;
    }

    if (isImmOperand(*token)) {
      switch (opType) {
      case D3D10_SB_OPERAND_TYPE_IMMEDIATE32: {
        SmallVector<int32_t, 4> values;
        for (unsigned i = 0; i < components->num; ++i) {
          Token value = parseToken();
          if (failed(value)) {
            emitError(getLocation(), "expected an immediate operand (imm32)");
            return failure();
          }
          values.push_back(*value);
        }
        return builder.buildOperandImm32(values, loc);
      }
      case D3D10_SB_OPERAND_TYPE_IMMEDIATE64: {
        if (components->num != 4) {
          emitError(getLocation(), "imm64 operand must have 4 components");
          return failure();
        }
        SmallVector<int64_t, 2> values;
        for (unsigned i = 0; i < 2; ++i) {
          Token high = parseToken();
          if (failed(high)) {
            emitError(getLocation(),
                      "expected an immediate operand (imm64 high)");
            return failure();
          }

          Token low = parseToken();
          if (failed(low)) {
            emitError(getLocation(),
                      "expected an immediate operand (imm64 low)");
            return failure();
          }

          values.push_back((((int64_t)*high) << 32) | *low);
        }
        return builder.buildOperandImm64(values, loc);
      }
      }
      emitError(getLocation(), "unhandled immediate type");
      return failure();
    }

    // Operand indices
    SmallVector<Index, 3> indices;
    for (uint32_t indexType : *indexTypes) {
      FailureOr<Index> index = parseIndex(indexType);
      if (failed(index))
        return failure();
      indices.push_back(*index);
    }
    return builder.buildOperand(opType, *components, indices, modifier,
                                getLocation());
  }

  FailureOr<Instruction> parseDclGlobalFlags(uint32_t opcodeToken,
                                             Location loc) {
    return builder.buildDclGlobalFlags(
        DECODE_D3D10_SB_GLOBAL_FLAGS(opcodeToken), loc);
  }

  FailureOr<Instruction> parseDclTemps(Location loc) {
    auto countToken = parseToken();
    if (failed(countToken))
      return failure();
    auto count = *countToken;
    if (count == 0) {
      emitError(getLocation(), "temp register count cannot be zero");
      return failure();
    }
    if (count > 4096) {
      emitError(getLocation(), "invalid temp register count: ")
          << count << " (max 4096)";
      return failure();
    }
    return builder.buildDclTemps(count, loc);
  }

  FailureOr<Instruction> parseDclIndexableTemp(Location loc) {
    auto registerIndexToken = parseToken();
    if (failed(registerIndexToken))
      return failure();
    auto registerCountToken = parseToken();
    if (failed(registerCountToken))
      return failure();
    auto numComponentsToken = parseToken();
    if (failed(numComponentsToken))
      return failure();

    auto registerCount = *registerCountToken;
    if (registerCount == 0) {
      emitError(getLocation(), "indexable temp array size cannot be zero");
      return failure();
    }
    if (registerCount > 4096) {
      emitError(getLocation(), "invalid indexable temp array size: ")
          << registerCount << " (max 4096)";
      return failure();
    }

    auto numComponents = *numComponentsToken;
    if (numComponents < 1 || numComponents > 4) {
      emitError(getLocation(), "indexable temp num components must be in 1..4, "
                               "got ")
          << numComponents;
      return failure();
    }

    return builder.buildDclIndexableTemp(*registerIndexToken, registerCount,
                                         numComponents, loc);
  }

  OptionalParseResult parseDclInstruction(uint32_t opcodeToken, Location loc,
                                          Instruction &out) {
    FailureOr<Instruction> result;
    switch (DECODE_D3D10_SB_OPCODE_TYPE(opcodeToken)) {
    case D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS:
      result = parseDclGlobalFlags(opcodeToken, loc);
      break;
    case D3D10_SB_OPCODE_DCL_TEMPS:
      result = parseDclTemps(loc);
      break;
    case D3D10_SB_OPCODE_DCL_INDEXABLE_TEMP:
      result = parseDclIndexableTemp(loc);
      break;
    default:
      return std::nullopt;
    }
    if (failed(result))
      return failure();
    out = *result;
    return success();
  }

  FailureOr<Instruction> parseInstruction() {
    size_t beginOffset = currentTokenOffset;
    Token token = parseToken();
    if (failed(token))
      return failure();

    FileLineColLoc loc = getLocation();

    uint32_t opcode = DECODE_D3D10_SB_OPCODE_TYPE(*token);
    InstructionModifier modifier;
    modifier.preciseMask = DECODE_D3D11_SB_INSTRUCTION_PRECISE_VALUES(*token);
    modifier.saturate = DECODE_IS_D3D10_SB_INSTRUCTION_SATURATE_ENABLED(*token);

    uint32_t length = DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(*token);

    // TODO: extended instructions:
    // BOOL b51PlusShader =
    // BOOL bExtended = DECODE_IS_D3D10_SB_OPCODE_EXTENDED(Token)
    // ...

    if (opcode >= D3D10_SB_NUM_OPCODES) {
      emitError(getLocation(), "unknown opcode");
      return failure();
    }

    auto opcodeToken = *token;

    Instruction dclInstruction;
    auto parseResult = parseDclInstruction(opcodeToken, loc, dclInstruction);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return failure();
      if (failed(verifyInstructionLength(beginOffset, length)))
        return failure();
      return dclInstruction;
    }

    unsigned numOperands = instrInfo[opcode].numOperands;

    SmallVector<Operand, 8> operands;
    for (unsigned i = 0; i < numOperands; ++i) {
      FailureOr<Operand> operand = parseOperand();
      if (failed(operand))
        return failure();
      operands.push_back(*operand);
    }

    if (failed(verifyInstructionLength(beginOffset, length)))
      return failure();

    return builder.buildInstruction(instrInfo[opcode].name, operands, modifier,
                                    loc);
  }

  FailureOr<Module> parseModule() {
    FileLineColLoc loc = getLocation(0);
    std::vector<Instruction> instructions;
    while (currentTokenOffset < buffer.size()) {
      FailureOr<Instruction> inst = parseInstruction();
      if (failed(inst)) {
        return failure();
      }
      instructions.push_back(*inst);
    }
    return builder.buildModule(instructions, loc);
  }

  LogicalResult verifyInstructionLength(size_t beginOffset, uint32_t length) {
    if (((currentTokenOffset - beginOffset) / 4) != length) {
      emitError(getLocation(), "instruction length mismatch");
      return failure();
    }
    return success();
  }

private:
  DXBuilder &builder;
  StringAttr name;
  StringRef buffer;
  size_t currentTokenOffset{0};
  InstructionInfo instrInfo[D3D10_SB_NUM_OPCODES];
};

namespace mlir::dxsa {
OwningOpRef<ModuleOp> importDxsaBinaryToModule(llvm::SourceMgr &source,
                                               MLIRContext *context) {

  if (source.getNumBuffers() != 1) {
    emitError(UnknownLoc::get(context), "one source file should be provided");
    return nullptr;
  }

  uint32_t sourceBufId = source.getMainFileID();
  StringRef buffer = source.getMemoryBuffer(sourceBufId)->getBuffer();
  StringAttr name = StringAttr::get(
      context, source.getMemoryBuffer(sourceBufId)->getBufferIdentifier());

  // FIXME:
  context->allowUnregisteredDialects();
  context->loadAllAvailableDialects();

  DXBuilder builder(context, name);
  Parser parser(builder, name, buffer);
  FailureOr<ModuleOp> mod = parser.parseModule();
  if (failed(mod))
    return {nullptr};
  return {*mod};
}
} // namespace mlir::dxsa
