use crate::isa::riscv64::lower::isle::generated_code::{
    VecAvl, VecLmul, VecMaskMode, VecSew, VecTailMode,
};

// TODO: Can we tell ISLE to derive this?
impl PartialEq for VecAvl {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VecAvl::Static { size: lhs }, VecAvl::Static { size: rhs }) => lhs == rhs,
            (VecAvl::Dynamic { size: lhs }, VecAvl::Dynamic { size: rhs }) => lhs == rhs,
            _ => false,
        }
    }
}

/// Vector Type (VType)
///
/// vtype provides the default type used to interpret the contents of the vector register file.
#[derive(Clone, Debug, PartialEq)]
pub struct VType {
    sew: VecSew,
    lmul: VecLmul,
    tail_mode: VecTailMode,
    mask_mode: VecMaskMode,
}

/// Vector State (VState)
///
/// VState represents the state of the vector unit that each instruction expects before execution.
/// Unlike VType or any of the other types here, VState is not a part of the RISC-V ISA. It is
/// used by our instruction emission code to ensure that the vector unit is in the correct state.
#[derive(Clone, Debug, PartialEq)]
pub struct VState {
    avl: VecAvl,
    vtype: VType,
}
