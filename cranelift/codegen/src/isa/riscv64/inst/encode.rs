//! Contains the RISC-V instruction encoding logic.
//!
//! These formats are specified in the RISC-V specification in section 2.2.
//! See: https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf
//!
//! Some instructions especially in extensions have slight variations from
//! the base RISC-V specification.

use super::{UImm5, VType};

/// Encode an R-type instruction.
///
/// Layout:
/// 0-------6-7-------11-12------14-15------19-20------24-25-------31
/// | Opcode |   rd     |  funct3  |   rs1    |   rs2    |   funct7  |
pub fn encode_r_type(opcode: u32, rd: u32, funct3: u32, rs1: u32, rs2: u32, funct7: u32) -> u32 {
    let mut bits = 0;
    bits |= opcode;
    bits |= rd << 7;
    bits |= funct3 << 12;
    bits |= rs1 << 15;
    bits |= rs2 << 20;
    bits |= funct7 << 25;
    bits
}

/// Encodes a Vector ALU instruction.
///
/// Fields:
/// - opcode (7 bits)
/// - vd     (5 bits)
/// - funct3 (3 bits)
/// - vs1    (5 bits)
/// - vs2    (5 bits)
/// - vm     (1 bit)
/// - funct6 (6 bits)
///
/// See: https://github.com/riscv/riscv-v-spec/blob/master/valu-format.adoc
pub fn encode_valu(
    opcode: u32,
    vd: u32,
    funct3: u32,
    vs1: u32,
    vs2: u32,
    vm: u32,
    funct6: u32,
) -> u32 {
    let funct7 = vm << 6 | funct6;
    encode_r_type(opcode, vd, funct3, vs1, vs2, funct7)
}

/// Encodes a Vector CFG Imm instruction.
///
/// See: https://github.com/riscv/riscv-v-spec/blob/master/vcfg-format.adoc
// TODO: Check if this is any of the known instruction types in the spec.
pub fn encode_vcfg_imm(opcode: u32, rd: u32, imm: UImm5, vtype: &VType) -> u32 {
    let mut bits = 0;
    bits |= opcode;
    bits |= rd << 7;
    bits |= 0b111 << 12;
    bits |= imm.bits() << 15;
    bits |= vtype.encode() << 20;
    bits |= 0b11 << 30;
    bits
}

/// Encodes a Vector Mem Unit Stride Load instruction.
///
/// See: https://github.com/riscv/riscv-v-spec/blob/master/vmem-format.adoc
/// TODO: These instructions share opcode space with LOAD-FP and STORE-FP
pub fn encode_vmem_load(
    opcode: u32,
    vd: u32,
    width: u32,
    rs1: u32,
    lumop: u32,
    vm: u32,
    mop: u32,
    mew: u32,
    nf: u32,
) -> u32 {
    let mut bits = 0;
    bits |= opcode & 0b1111111;
    bits |= (vd & 0b11111) << 7;
    bits |= (width & 0b111) << 12;
    bits |= (rs1 & 0b11111) << 15;
    bits |= (lumop & 0b11111) << 20;
    bits |= (vm & 0b1) << 25;
    bits |= (mop & 0b11) << 26;
    bits |= (mew & 0b1) << 28;
    bits |= (nf & 0b111) << 29;
    bits
}

/// Encodes a Vector Mem Unit Stride Load instruction.
///
/// See: https://github.com/riscv/riscv-v-spec/blob/master/vmem-format.adoc
/// TODO: These instructions share opcode space with LOAD-FP and STORE-FP
pub fn encode_vmem_store(
    opcode: u32,
    vs3: u32,
    width: u32,
    rs1: u32,
    sumop: u32,
    vm: u32,
    mop: u32,
    mew: u32,
    nf: u32,
) -> u32 {
    // This is pretty much the same as the load instruction, just
    // with different names on the fields.
    encode_vmem_load(opcode, vs3, width, rs1, sumop, vm, mop, mew, nf)
}
