//! Unwind information for System V ABI (Riscv64).

use crate::isa::riscv64::inst::regs;
use crate::isa::unwind::systemv::RegisterMappingError;
use crate::machinst::Reg;
use gimli::{write::CommonInformationEntry, Encoding, Format, Register};
use regalloc2::RegClass;

/// Creates a new riscv64 common information entry (CIE).
pub fn create_cie() -> CommonInformationEntry {
    use gimli::write::CallFrameInstruction;

    let mut entry = CommonInformationEntry::new(
        Encoding {
            address_size: 8,
            format: Format::Dwarf32,
            version: 1,
        },
        4,  // Code alignment factor
        -8, // Data alignment factor
        Register(regs::link_reg().to_real_reg().unwrap().hw_enc() as u16),
    );

    // Every frame will start with the call frame address (CFA) at SP
    let sp = Register(regs::stack_reg().to_real_reg().unwrap().hw_enc().into());
    entry.add_instruction(CallFrameInstruction::Cfa(sp, 0));

    entry
}

/// Map Cranelift registers to their corresponding Gimli registers.
pub fn map_reg(reg: Reg) -> Result<Register, RegisterMappingError> {
    match reg.class() {
        RegClass::Int => {
            let reg = reg.to_real_reg().unwrap().hw_enc() as u16;
            Ok(Register(reg))
        }
        RegClass::Float => {
            let reg = reg.to_real_reg().unwrap().hw_enc() as u16;
            Ok(Register(32 + reg))
        }
    }
}

pub(crate) struct RegisterMapper;

impl crate::isa::unwind::systemv::RegisterMapper<Reg> for RegisterMapper {
    fn map(&self, reg: Reg) -> Result<u16, RegisterMappingError> {
        Ok(map_reg(reg)?.0)
    }
    fn sp(&self) -> u16 {
        regs::stack_reg().to_real_reg().unwrap().hw_enc() as u16
    }
    fn fp(&self) -> Option<u16> {
        Some(regs::fp_reg().to_real_reg().unwrap().hw_enc() as u16)
    }
    fn lr(&self) -> Option<u16> {
        Some(regs::link_reg().to_real_reg().unwrap().hw_enc() as u16)
    }
    fn lr_offset(&self) -> Option<u32> {
        Some(8)
    }
}

#[cfg(test)]
mod tests {
    use crate::cursor::{Cursor, FuncCursor};
    use crate::ir::types;
    use crate::ir::AbiParam;
    use crate::ir::{ExternalName, Function, InstBuilder, Signature, StackSlotData, StackSlotKind};
    use crate::isa::{lookup, CallConv};
    use crate::settings::{builder, Flags};
    use crate::Context;
    use gimli::write::Address;
    use std::str::FromStr;
    use target_lexicon::triple;

    #[test]
    fn test_simple_func() {
        let isa = lookup(triple!("riscv64"))
            .expect("expect riscv64 ISA")
            .finish(Flags::new(builder()))
            .expect("Creating compiler backend");

        let mut context = Context::for_function(create_function(
            CallConv::SystemV,
            Some(StackSlotData::new(StackSlotKind::ExplicitSlot, 64)),
        ));

        context.compile(&*isa).expect("expected compilation");

        let fde = match context
            .create_unwind_info(isa.as_ref())
            .expect("can create unwind info")
        {
            Some(crate::isa::unwind::UnwindInfo::SystemV(info)) => {
                info.to_fde(Address::Constant(1234))
            }
            _ => panic!("expected unwind information"),
        };

        assert_eq!(format!("{:?}", fde), "FrameDescriptionEntry { address: Constant(1234), length: 40, lsda: None, instructions: [(12, CfaOffset(16)), (12, Offset(Register(8), -16)), (12, Offset(Register(1), -8)), (16, CfaRegister(Register(8)))] }");
    }

    fn create_function(call_conv: CallConv, stack_slot: Option<StackSlotData>) -> Function {
        let mut func =
            Function::with_name_signature(ExternalName::user(0, 0), Signature::new(call_conv));

        let block0 = func.dfg.make_block();
        let mut pos = FuncCursor::new(&mut func);
        pos.insert_block(block0);
        pos.ins().return_(&[]);

        if let Some(stack_slot) = stack_slot {
            func.sized_stack_slots.push(stack_slot);
        }

        func
    }
}
