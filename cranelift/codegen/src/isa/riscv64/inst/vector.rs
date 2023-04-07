use crate::isa::riscv64::lower::isle::generated_code::{
    VecAluOpRRR, VecAvl, VecLmul, VecMaskMode, VecSew, VecTailMode,
};
use core::fmt;

use super::UImm5;

impl VecAvl {
    pub fn _static(size: u32) -> Self {
        VecAvl::Static {
            size: UImm5::maybe_from_u8(size as u8).expect("Invalid size for AVL"),
        }
    }
}

// TODO: Can we tell ISLE to derive this?
impl PartialEq for VecAvl {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VecAvl::Static { size: lhs }, VecAvl::Static { size: rhs }) => lhs == rhs,
        }
    }
}

impl fmt::Display for VecAvl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VecAvl::Static { size } => write!(f, "{}", size),
        }
    }
}

impl VecSew {
    pub fn from_bits(bits: u32) -> Self {
        match bits {
            8 => VecSew::E8,
            16 => VecSew::E16,
            32 => VecSew::E32,
            64 => VecSew::E64,
            _ => panic!("Invalid number of bits for VecSew: {}", bits),
        }
    }

    pub fn bits(&self) -> u32 {
        match self {
            VecSew::E8 => 8,
            VecSew::E16 => 16,
            VecSew::E32 => 32,
            VecSew::E64 => 64,
        }
    }
}

impl fmt::Display for VecSew {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "e{}", self.bits())
    }
}

impl fmt::Display for VecLmul {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VecLmul::Lmul1_8 => write!(f, "m1/8"),
            VecLmul::Lmul1_4 => write!(f, "m1/4"),
            VecLmul::Lmul1_2 => write!(f, "m1/2"),
            VecLmul::Lmul1 => write!(f, "m1"),
            VecLmul::Lmul2 => write!(f, "m2"),
            VecLmul::Lmul4 => write!(f, "m4"),
            VecLmul::Lmul8 => write!(f, "m8"),
        }
    }
}

impl fmt::Display for VecTailMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VecTailMode::Agnostic => write!(f, "ta"),
            VecTailMode::Undisturbed => write!(f, "tu"),
        }
    }
}

impl fmt::Display for VecMaskMode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VecMaskMode::Agnostic => write!(f, "ma"),
            VecMaskMode::Undisturbed => write!(f, "mu"),
        }
    }
}

/// Vector Type (VType)
///
/// vtype provides the default type used to interpret the contents of the vector register file.
#[derive(Clone, Debug, PartialEq)]
pub struct VType {
    pub sew: VecSew,
    pub lmul: VecLmul,
    pub tail_mode: VecTailMode,
    pub mask_mode: VecMaskMode,
}

impl fmt::Display for VType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}",
            self.sew, self.lmul, self.tail_mode, self.mask_mode
        )
    }
}

/// Vector State (VState)
///
/// VState represents the state of the vector unit that each instruction expects before execution.
/// Unlike VType or any of the other types here, VState is not a part of the RISC-V ISA. It is
/// used by our instruction emission code to ensure that the vector unit is in the correct state.
#[derive(Clone, Debug, PartialEq)]
pub struct VState {
    pub avl: VecAvl,
    pub vtype: VType,
}

impl fmt::Display for VState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#avl={}, #vtype=({})", self.avl, self.vtype)
    }
}

impl VecAluOpRRR {
    pub fn opcode(&self) -> u32 {
        match self {
            VecAluOpRRR::Vadd => 0b00000,
        }
    }
    pub fn funct3(&self) -> u32 {
        match self {
            VecAluOpRRR::Vadd => 0b00000,
        }
    }
    pub fn funct6(&self) -> u32 {
        match self {
            VecAluOpRRR::Vadd => 0b00000,
        }
    }
}

impl fmt::Display for VecAluOpRRR {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VecAluOpRRR::Vadd => write!(f, "vadd.vv"),
        }
    }
}
