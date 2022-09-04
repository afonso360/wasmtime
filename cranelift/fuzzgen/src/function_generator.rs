use crate::codegen::ir::{ArgumentExtension, ArgumentPurpose};
use crate::config::Config;
use anyhow::Result;
use arbitrary::{Arbitrary, Unstructured};
use cranelift::codegen::ir::instructions::InstructionFormat;
use cranelift::codegen::ir::{types::*, FuncRef, LibCall, UserExternalName, UserFuncName};
use cranelift::codegen::ir::{
    AbiParam, Block, ExternalName, Function, JumpTable, Opcode, Signature, StackSlot, Type, Value,
};
use cranelift::codegen::isa::CallConv;
use cranelift::frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift::prelude::{
    EntityRef, ExtFuncData, FloatCC, InstBuilder, IntCC, JumpTableData, StackSlotData,
    StackSlotKind,
};
use std::collections::HashMap;
use std::ops::RangeInclusive;

type BlockSignature = Vec<Type>;

/// Generates an instruction(`iconst`/`fconst`/etc...) to introduce a constant value
fn generate_const(u: &mut Unstructured, builder: &mut FunctionBuilder, ty: Type) -> Result<Value> {
    Ok(match ty {
        I128 => {
            // See: https://github.com/bytecodealliance/wasmtime/issues/2906
            let hi = builder.ins().iconst(I64, u.arbitrary::<i64>()?);
            let lo = builder.ins().iconst(I64, u.arbitrary::<i64>()?);
            builder.ins().iconcat(lo, hi)
        }
        ty if ty.is_int() => {
            let imm64 = match ty {
                I8 => u.arbitrary::<i8>()? as i64,
                I16 => u.arbitrary::<i16>()? as i64,
                I32 => u.arbitrary::<i32>()? as i64,
                I64 => u.arbitrary::<i64>()?,
                _ => unreachable!(),
            };
            builder.ins().iconst(ty, imm64)
        }
        ty if ty.is_bool() => builder.ins().bconst(ty, bool::arbitrary(u)?),
        // f{32,64}::arbitrary does not generate a bunch of important values
        // such as Signaling NaN's / NaN's with payload, so generate floats from integers.
        F32 => builder.ins().f32const(f32::from_bits(u32::arbitrary(u)?)),
        F64 => builder.ins().f64const(f64::from_bits(u64::arbitrary(u)?)),
        _ => unimplemented!(),
    })
}

fn insert_opcode(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    opcode: Opcode,
    args: &'static [Type],
    rets: &'static [Type],
) -> Result<()> {
    let mut vals = Vec::with_capacity(args.len());
    for &arg in args.into_iter() {
        let var = fgen.get_variable_of_type(arg)?;
        let val = builder.use_var(var);
        vals.push(val);
    }

    // For pretty much every instruction the control type is the return type
    // except for Iconcat and Isplit which are *special* and the control type
    // is the input type.
    let ctrl_type = if opcode == Opcode::Iconcat || opcode == Opcode::Isplit {
        args.first()
    } else {
        rets.first()
    }
    .copied()
    .unwrap_or(INVALID);

    // Choose the appropriate instruction format for this opcode
    let (inst, dfg) = match opcode.format() {
        InstructionFormat::NullAry => builder.ins().NullAry(opcode, ctrl_type),
        InstructionFormat::Unary => builder.ins().Unary(opcode, ctrl_type, vals[0]),
        InstructionFormat::Binary => builder.ins().Binary(opcode, ctrl_type, vals[0], vals[1]),
        InstructionFormat::Ternary => builder
            .ins()
            .Ternary(opcode, ctrl_type, vals[0], vals[1], vals[2]),
        _ => unimplemented!(),
    };
    let results = dfg.inst_results(inst).to_vec();

    for (val, &ty) in results.into_iter().zip(rets) {
        let var = fgen.get_variable_of_type(ty)?;
        builder.def_var(var, val);
    }
    Ok(())
}

fn insert_call(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    opcode: Opcode,
    _args: &'static [Type],
    _rets: &'static [Type],
) -> Result<()> {
    assert_eq!(opcode, Opcode::Call, "only call handled at the moment");
    let (sig, func_ref) = fgen.u.choose(&fgen.resources.func_refs)?.clone();

    let actuals = fgen.generate_values_for_signature(
        builder,
        sig.params.iter().map(|abi_param| abi_param.value_type),
    )?;

    builder.ins().call(func_ref, &actuals);
    Ok(())
}

fn insert_stack_load(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    _opcode: Opcode,
    _args: &'static [Type],
    rets: &'static [Type],
) -> Result<()> {
    let typevar = rets[0];
    let slot = fgen.stack_slot_with_size(typevar.bytes())?;
    let slot_size = builder.func.sized_stack_slots[slot].size;
    let type_size = typevar.bytes();
    let offset = fgen.u.int_in_range(0..=(slot_size - type_size))? as i32;

    let val = builder.ins().stack_load(typevar, slot, offset);
    let var = fgen.get_variable_of_type(typevar)?;
    builder.def_var(var, val);

    Ok(())
}

fn insert_stack_store(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    _opcode: Opcode,
    args: &'static [Type],
    _rets: &'static [Type],
) -> Result<()> {
    let typevar = args[0];
    let slot = fgen.stack_slot_with_size(typevar.bytes())?;
    let slot_size = builder.func.sized_stack_slots[slot].size;
    let type_size = typevar.bytes();
    let offset = fgen.u.int_in_range(0..=(slot_size - type_size))? as i32;

    let arg0 = fgen.get_variable_of_type(typevar)?;
    let arg0 = builder.use_var(arg0);

    builder.ins().stack_store(arg0, slot, offset);
    Ok(())
}

fn insert_cmp(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    opcode: Opcode,
    args: &'static [Type],
    rets: &'static [Type],
) -> Result<()> {
    let lhs = fgen.get_variable_of_type(args[0])?;
    let lhs = builder.use_var(lhs);

    let rhs = fgen.get_variable_of_type(args[1])?;
    let rhs = builder.use_var(rhs);

    let res = if opcode == Opcode::Fcmp {
        // Some FloatCC's are not implemented on AArch64, see:
        // https://github.com/bytecodealliance/wasmtime/issues/4850
        let float_cc = if cfg!(target_arch = "aarch64") {
            &[
                FloatCC::Ordered,
                FloatCC::Unordered,
                FloatCC::Equal,
                FloatCC::NotEqual,
                FloatCC::LessThan,
                FloatCC::LessThanOrEqual,
                FloatCC::GreaterThan,
                FloatCC::GreaterThanOrEqual,
            ]
        } else {
            FloatCC::all()
        };

        let cc = *fgen.u.choose(float_cc)?;
        builder.ins().fcmp(cc, lhs, rhs)
    } else {
        let cc = *fgen.u.choose(IntCC::all())?;
        builder.ins().icmp(cc, lhs, rhs)
    };

    let var = fgen.get_variable_of_type(rets[0])?;
    builder.def_var(var, res);
    Ok(())
}

fn insert_const(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    _opcode: Opcode,
    _args: &'static [Type],
    rets: &'static [Type],
) -> Result<()> {
    let typevar = rets[0];
    let var = fgen.get_variable_of_type(typevar)?;
    let val = generate_const(&mut fgen.u, builder, typevar)?;
    builder.def_var(var, val);
    Ok(())
}

type InstTemplate<'a> = Box<dyn Fn(&mut Unstructured, &mut FunctionBuilder) -> Result<()> + 'a>;

fn generate_instruction_templates(resources: &Resources) -> Vec<InstTemplate> {
    let mut result: Vec<InstTemplate> = Vec::new();
    result.push(Box::new(|_u, b| {
        b.ins().nop();
        Ok(())
    }));

    use cranelift::frontend::FuncInstBuilder;

    for ty in [I8, I16, I32, I64] {
        if let Some(vars) = resources.vars.get(&ty) {
            let ops = [
                |i: FuncInstBuilder, x, y| i.iadd(x, y),
                |i: FuncInstBuilder, x, y| i.isub(x, y),
                |i: FuncInstBuilder, x, y| i.imul(x, y),
                |i: FuncInstBuilder, x, y| i.udiv(x, y),
                |i: FuncInstBuilder, x, y| i.sdiv(x, y),
            ];
            for op in ops {
                result.push(Box::new(move |u, b| {
                    let x = b.use_var(*u.choose(vars)?);
                    let y = b.use_var(*u.choose(vars)?);
                    let val = op(b.ins(), x, y);
                    b.def_var(*u.choose(vars)?, val);
                    Ok(())
                }));
            }

            for ty2 in [I8, I16, I32, I64, I128] {
                if let Some(vars2) = resources.vars.get(&ty2) {
                    let ops = [
                        |i: FuncInstBuilder, x, y| i.rotl(x, y),
                        |i: FuncInstBuilder, x, y| i.rotr(x, y),
                    ];
                    for op in ops {
                        result.push(Box::new(move |u, b| {
                            let x = b.use_var(*u.choose(vars)?);
                            let y = b.use_var(*u.choose(vars2)?);
                            let val = op(b.ins(), x, y);
                            b.def_var(*u.choose(vars)?, val);
                            Ok(())
                        }));
                    }

                    // Some test cases disabled due to:
                    // https://github.com/bytecodealliance/wasmtime/issues/4699
                    if ty.bits() >= 32 || ty2 != I128 {
                        let ops = [
                            |i: FuncInstBuilder, x, y| i.ishl(x, y),
                            |i: FuncInstBuilder, x, y| i.sshr(x, y),
                            |i: FuncInstBuilder, x, y| i.ushr(x, y),
                        ];
                        for op in ops {
                            result.push(Box::new(move |u, b| {
                                let x = b.use_var(*u.choose(vars)?);
                                let y = b.use_var(*u.choose(vars2)?);
                                let val = op(b.ins(), x, y);
                                b.def_var(*u.choose(vars)?, val);
                                Ok(())
                            }));
                        }
                    }

                    if ty.bits() < ty2.bits() {
                        result.push(Box::new(move |u, b| {
                            let x = b.use_var(*u.choose(vars)?);
                            let val = b.ins().uextend(ty2, x);
                            b.def_var(*u.choose(vars2)?, val);
                            Ok(())
                        }));
                        result.push(Box::new(move |u, b| {
                            let x = b.use_var(*u.choose(vars)?);
                            let val = b.ins().sextend(ty2, x);
                            b.def_var(*u.choose(vars2)?, val);
                            Ok(())
                        }));
                        result.push(Box::new(move |u, b| {
                            let x = b.use_var(*u.choose(vars2)?);
                            let val = b.ins().ireduce(ty, x);
                            b.def_var(*u.choose(vars)?, val);
                            Ok(())
                        }));
                    }
                }
            }
        }
    }

    if let (Some(vars128), Some(vars64)) = (resources.vars.get(&I128), resources.vars.get(&I64)) {
        result.push(Box::new(|u, b| {
            let x = b.use_var(*u.choose(vars128)?);
            let (lo, hi) = b.ins().isplit(x);
            b.def_var(*u.choose(vars64)?, lo);
            b.def_var(*u.choose(vars64)?, hi);
            Ok(())
        }));
        result.push(Box::new(|u, b| {
            let lo = b.use_var(*u.choose(vars64)?);
            let hi = b.use_var(*u.choose(vars64)?);
            let val = b.ins().iconcat(lo, hi);
            b.def_var(*u.choose(vars128)?, val);
            Ok(())
        }));
    }

    for ty in [F32, F64] {
        if let Some(vars) = resources.vars.get(&ty) {
            let ops = [
                |i: FuncInstBuilder, x, y| i.fadd(x, y),
                |i: FuncInstBuilder, x, y| i.fmul(x, y),
                |i: FuncInstBuilder, x, y| i.fsub(x, y),
                |i: FuncInstBuilder, x, y| i.fdiv(x, y),
                |i: FuncInstBuilder, x, y| i.fmin(x, y),
                |i: FuncInstBuilder, x, y| i.fmax(x, y),
                |i: FuncInstBuilder, x, y| i.fmin_pseudo(x, y),
                |i: FuncInstBuilder, x, y| i.fmax_pseudo(x, y),
                |i: FuncInstBuilder, x, y| i.fcopysign(x, y),
            ];
            for op in ops {
                result.push(Box::new(move |u, b| {
                    let x = b.use_var(*u.choose(vars)?);
                    let y = b.use_var(*u.choose(vars)?);
                    let val = op(b.ins(), x, y);
                    b.def_var(*u.choose(vars)?, val);
                    Ok(())
                }));
            }

            result.push(Box::new(move |u, b| {
                let x = b.use_var(*u.choose(vars)?);
                let y = b.use_var(*u.choose(vars)?);
                let z = b.use_var(*u.choose(vars)?);
                let val = b.ins().fma(x, y, z);
                b.def_var(*u.choose(vars)?, val);
                Ok(())
            }));

            let ops = [
                |i: FuncInstBuilder, x| i.fabs(x),
                |i: FuncInstBuilder, x| i.fneg(x),
                |i: FuncInstBuilder, x| i.sqrt(x),
                |i: FuncInstBuilder, x| i.ceil(x),
                |i: FuncInstBuilder, x| i.floor(x),
                |i: FuncInstBuilder, x| i.trunc(x),
                |i: FuncInstBuilder, x| i.nearest(x),
            ];
            for op in ops {
                result.push(Box::new(move |u, b| {
                    let x = b.use_var(*u.choose(vars)?);
                    let val = op(b.ins(), x);
                    b.def_var(*u.choose(vars)?, val);
                    Ok(())
                }));
            }
        }
    }

    if let Some(bools) = resources.vars.get(&B1) {
        for ty in [F32, F64] {
            if let Some(vars) = resources.vars.get(&ty) {
                result.push(Box::new(move |u, b| {
                    let x = b.use_var(*u.choose(vars)?);
                    let y = b.use_var(*u.choose(vars)?);
                    let val = b.ins().fcmp(*u.choose(FloatCC::all())?, x, y);
                    b.def_var(*u.choose(bools)?, val);
                    Ok(())
                }));
            }
        }

        // TODO: icmp of/nof broken for i128 on x86_64
        // See: https://github.com/bytecodealliance/wasmtime/issues/4406
        for ty in [I8, I16, I32, I64] {
            if let Some(vars) = resources.vars.get(&ty) {
                result.push(Box::new(|u, b| {
                    let x = b.use_var(*u.choose(vars)?);
                    let y = b.use_var(*u.choose(vars)?);
                    let val = b.ins().icmp(*u.choose(IntCC::all())?, x, y);
                    b.def_var(*u.choose(bools)?, val);
                    Ok(())
                }));
            }
        }
    }

    for ty in [I8, I16, I32, I64, I128] {
        if let Some(vars) = resources.vars.get(&ty) {
            let type_size = ty.bytes();
            let slots = resources.stack_slots_with_size(type_size);
            if !slots.is_empty() {
                result.push(Box::new(move |u, b| {
                    let (slot_size, slot) = *u.choose(slots)?;
                    let offset = u.int_in_range(0..=(slot_size - type_size))? as i32;
                    let x = b.use_var(*u.choose(vars)?);
                    b.ins().stack_store(x, slot, offset);
                    Ok(())
                }));
                result.push(Box::new(move |u, b| {
                    let (slot_size, slot) = *u.choose(slots)?;
                    let offset = u.int_in_range(0..=(slot_size - type_size))? as i32;
                    let val = b.ins().stack_load(ty, slot, offset);
                    b.def_var(*u.choose(vars)?, val);
                    Ok(())
                }));
            }
        }
    }

    for ty in [I8, I16, I32, I64, I128, F32, F64, B1] {
        if let Some(vars) = resources.vars.get(&ty) {
            result.push(Box::new(move |u, b| {
                let val = generate_const(u, b, ty)?;
                b.def_var(*u.choose(vars)?, val);
                Ok(())
            }));
        }
    }

    result
}

/*
fn generate_terminator_templates(resources: &Resources) -> Vec<InstTemplate> {
    let mut result: Vec<InstTemplate> = Vec::new();

    let ret_vars: Vec<_> =
    result.push(Box::new(move |u, b| {
    }));

    result
}
*/

type OpcodeInserter = fn(
    fgen: &mut FunctionGenerator,
    builder: &mut FunctionBuilder,
    Opcode,
    &'static [Type],
    &'static [Type],
) -> Result<()>;

// TODO: Derive this from the `cranelift-meta` generator.
const OPCODE_SIGNATURES: &'static [(
    Opcode,
    &'static [Type], // Args
    &'static [Type], // Rets
    OpcodeInserter,
)] = &[
    (Opcode::Nop, &[], &[], insert_opcode),
    // Iadd
    (Opcode::Iadd, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Iadd, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Iadd, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Iadd, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Iadd, &[I128, I128], &[I128], insert_opcode),
    // Isub
    (Opcode::Isub, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Isub, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Isub, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Isub, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Isub, &[I128, I128], &[I128], insert_opcode),
    // Imul
    (Opcode::Imul, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Imul, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Imul, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Imul, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Imul, &[I128, I128], &[I128], insert_opcode),
    // Udiv
    (Opcode::Udiv, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Udiv, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Udiv, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Udiv, &[I64, I64], &[I64], insert_opcode),
    // udiv.i128 not implemented in some backends:
    //   x64: https://github.com/bytecodealliance/wasmtime/issues/4756
    //   aarch64: https://github.com/bytecodealliance/wasmtime/issues/4864
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    (Opcode::Udiv, &[I128, I128], &[I128], insert_opcode),
    // Sdiv
    (Opcode::Sdiv, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Sdiv, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Sdiv, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Sdiv, &[I64, I64], &[I64], insert_opcode),
    // sdiv.i128 not implemented in some backends:
    //   x64: https://github.com/bytecodealliance/wasmtime/issues/4770
    //   aarch64: https://github.com/bytecodealliance/wasmtime/issues/4864
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    (Opcode::Sdiv, &[I128, I128], &[I128], insert_opcode),
    // Rotr
    (Opcode::Rotr, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Rotr, &[I8, I16], &[I8], insert_opcode),
    (Opcode::Rotr, &[I8, I32], &[I8], insert_opcode),
    (Opcode::Rotr, &[I8, I64], &[I8], insert_opcode),
    (Opcode::Rotr, &[I8, I128], &[I8], insert_opcode),
    (Opcode::Rotr, &[I16, I8], &[I16], insert_opcode),
    (Opcode::Rotr, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Rotr, &[I16, I32], &[I16], insert_opcode),
    (Opcode::Rotr, &[I16, I64], &[I16], insert_opcode),
    (Opcode::Rotr, &[I16, I128], &[I16], insert_opcode),
    (Opcode::Rotr, &[I32, I8], &[I32], insert_opcode),
    (Opcode::Rotr, &[I32, I16], &[I32], insert_opcode),
    (Opcode::Rotr, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Rotr, &[I32, I64], &[I32], insert_opcode),
    (Opcode::Rotr, &[I32, I128], &[I32], insert_opcode),
    (Opcode::Rotr, &[I64, I8], &[I64], insert_opcode),
    (Opcode::Rotr, &[I64, I16], &[I64], insert_opcode),
    (Opcode::Rotr, &[I64, I32], &[I64], insert_opcode),
    (Opcode::Rotr, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Rotr, &[I64, I128], &[I64], insert_opcode),
    (Opcode::Rotr, &[I128, I8], &[I128], insert_opcode),
    (Opcode::Rotr, &[I128, I16], &[I128], insert_opcode),
    (Opcode::Rotr, &[I128, I32], &[I128], insert_opcode),
    (Opcode::Rotr, &[I128, I64], &[I128], insert_opcode),
    (Opcode::Rotr, &[I128, I128], &[I128], insert_opcode),
    // Rotl
    (Opcode::Rotl, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Rotl, &[I8, I16], &[I8], insert_opcode),
    (Opcode::Rotl, &[I8, I32], &[I8], insert_opcode),
    (Opcode::Rotl, &[I8, I64], &[I8], insert_opcode),
    (Opcode::Rotl, &[I8, I128], &[I8], insert_opcode),
    (Opcode::Rotl, &[I16, I8], &[I16], insert_opcode),
    (Opcode::Rotl, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Rotl, &[I16, I32], &[I16], insert_opcode),
    (Opcode::Rotl, &[I16, I64], &[I16], insert_opcode),
    (Opcode::Rotl, &[I16, I128], &[I16], insert_opcode),
    (Opcode::Rotl, &[I32, I8], &[I32], insert_opcode),
    (Opcode::Rotl, &[I32, I16], &[I32], insert_opcode),
    (Opcode::Rotl, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Rotl, &[I32, I64], &[I32], insert_opcode),
    (Opcode::Rotl, &[I32, I128], &[I32], insert_opcode),
    (Opcode::Rotl, &[I64, I8], &[I64], insert_opcode),
    (Opcode::Rotl, &[I64, I16], &[I64], insert_opcode),
    (Opcode::Rotl, &[I64, I32], &[I64], insert_opcode),
    (Opcode::Rotl, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Rotl, &[I64, I128], &[I64], insert_opcode),
    (Opcode::Rotl, &[I128, I8], &[I128], insert_opcode),
    (Opcode::Rotl, &[I128, I16], &[I128], insert_opcode),
    (Opcode::Rotl, &[I128, I32], &[I128], insert_opcode),
    (Opcode::Rotl, &[I128, I64], &[I128], insert_opcode),
    (Opcode::Rotl, &[I128, I128], &[I128], insert_opcode),
    // Ishl
    (Opcode::Ishl, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Ishl, &[I8, I16], &[I8], insert_opcode),
    (Opcode::Ishl, &[I8, I32], &[I8], insert_opcode),
    (Opcode::Ishl, &[I8, I64], &[I8], insert_opcode),
    (Opcode::Ishl, &[I8, I128], &[I8], insert_opcode),
    (Opcode::Ishl, &[I16, I8], &[I16], insert_opcode),
    (Opcode::Ishl, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Ishl, &[I16, I32], &[I16], insert_opcode),
    (Opcode::Ishl, &[I16, I64], &[I16], insert_opcode),
    (Opcode::Ishl, &[I16, I128], &[I16], insert_opcode),
    (Opcode::Ishl, &[I32, I8], &[I32], insert_opcode),
    (Opcode::Ishl, &[I32, I16], &[I32], insert_opcode),
    (Opcode::Ishl, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Ishl, &[I32, I64], &[I32], insert_opcode),
    (Opcode::Ishl, &[I32, I128], &[I32], insert_opcode),
    (Opcode::Ishl, &[I64, I8], &[I64], insert_opcode),
    (Opcode::Ishl, &[I64, I16], &[I64], insert_opcode),
    (Opcode::Ishl, &[I64, I32], &[I64], insert_opcode),
    (Opcode::Ishl, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Ishl, &[I64, I128], &[I64], insert_opcode),
    (Opcode::Ishl, &[I128, I8], &[I128], insert_opcode),
    (Opcode::Ishl, &[I128, I16], &[I128], insert_opcode),
    (Opcode::Ishl, &[I128, I32], &[I128], insert_opcode),
    (Opcode::Ishl, &[I128, I64], &[I128], insert_opcode),
    (Opcode::Ishl, &[I128, I128], &[I128], insert_opcode),
    // Sshr
    (Opcode::Sshr, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Sshr, &[I8, I16], &[I8], insert_opcode),
    (Opcode::Sshr, &[I8, I32], &[I8], insert_opcode),
    (Opcode::Sshr, &[I8, I64], &[I8], insert_opcode),
    (Opcode::Sshr, &[I8, I128], &[I8], insert_opcode),
    (Opcode::Sshr, &[I16, I8], &[I16], insert_opcode),
    (Opcode::Sshr, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Sshr, &[I16, I32], &[I16], insert_opcode),
    (Opcode::Sshr, &[I16, I64], &[I16], insert_opcode),
    (Opcode::Sshr, &[I16, I128], &[I16], insert_opcode),
    (Opcode::Sshr, &[I32, I8], &[I32], insert_opcode),
    (Opcode::Sshr, &[I32, I16], &[I32], insert_opcode),
    (Opcode::Sshr, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Sshr, &[I32, I64], &[I32], insert_opcode),
    (Opcode::Sshr, &[I32, I128], &[I32], insert_opcode),
    (Opcode::Sshr, &[I64, I8], &[I64], insert_opcode),
    (Opcode::Sshr, &[I64, I16], &[I64], insert_opcode),
    (Opcode::Sshr, &[I64, I32], &[I64], insert_opcode),
    (Opcode::Sshr, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Sshr, &[I64, I128], &[I64], insert_opcode),
    (Opcode::Sshr, &[I128, I8], &[I128], insert_opcode),
    (Opcode::Sshr, &[I128, I16], &[I128], insert_opcode),
    (Opcode::Sshr, &[I128, I32], &[I128], insert_opcode),
    (Opcode::Sshr, &[I128, I64], &[I128], insert_opcode),
    (Opcode::Sshr, &[I128, I128], &[I128], insert_opcode),
    // Ushr
    (Opcode::Ushr, &[I8, I8], &[I8], insert_opcode),
    (Opcode::Ushr, &[I8, I16], &[I8], insert_opcode),
    (Opcode::Ushr, &[I8, I32], &[I8], insert_opcode),
    (Opcode::Ushr, &[I8, I64], &[I8], insert_opcode),
    (Opcode::Ushr, &[I8, I128], &[I8], insert_opcode),
    (Opcode::Ushr, &[I16, I8], &[I16], insert_opcode),
    (Opcode::Ushr, &[I16, I16], &[I16], insert_opcode),
    (Opcode::Ushr, &[I16, I32], &[I16], insert_opcode),
    (Opcode::Ushr, &[I16, I64], &[I16], insert_opcode),
    (Opcode::Ushr, &[I16, I128], &[I16], insert_opcode),
    (Opcode::Ushr, &[I32, I8], &[I32], insert_opcode),
    (Opcode::Ushr, &[I32, I16], &[I32], insert_opcode),
    (Opcode::Ushr, &[I32, I32], &[I32], insert_opcode),
    (Opcode::Ushr, &[I32, I64], &[I32], insert_opcode),
    (Opcode::Ushr, &[I32, I128], &[I32], insert_opcode),
    (Opcode::Ushr, &[I64, I8], &[I64], insert_opcode),
    (Opcode::Ushr, &[I64, I16], &[I64], insert_opcode),
    (Opcode::Ushr, &[I64, I32], &[I64], insert_opcode),
    (Opcode::Ushr, &[I64, I64], &[I64], insert_opcode),
    (Opcode::Ushr, &[I64, I128], &[I64], insert_opcode),
    (Opcode::Ushr, &[I128, I8], &[I128], insert_opcode),
    (Opcode::Ushr, &[I128, I16], &[I128], insert_opcode),
    (Opcode::Ushr, &[I128, I32], &[I128], insert_opcode),
    (Opcode::Ushr, &[I128, I64], &[I128], insert_opcode),
    (Opcode::Ushr, &[I128, I128], &[I128], insert_opcode),
    // Uextend
    (Opcode::Uextend, &[I8], &[I16], insert_opcode),
    (Opcode::Uextend, &[I8], &[I32], insert_opcode),
    (Opcode::Uextend, &[I8], &[I64], insert_opcode),
    (Opcode::Uextend, &[I8], &[I128], insert_opcode),
    (Opcode::Uextend, &[I16], &[I32], insert_opcode),
    (Opcode::Uextend, &[I16], &[I64], insert_opcode),
    (Opcode::Uextend, &[I16], &[I128], insert_opcode),
    (Opcode::Uextend, &[I32], &[I64], insert_opcode),
    (Opcode::Uextend, &[I32], &[I128], insert_opcode),
    (Opcode::Uextend, &[I64], &[I128], insert_opcode),
    // Sextend
    (Opcode::Sextend, &[I8], &[I16], insert_opcode),
    (Opcode::Sextend, &[I8], &[I32], insert_opcode),
    (Opcode::Sextend, &[I8], &[I64], insert_opcode),
    (Opcode::Sextend, &[I8], &[I128], insert_opcode),
    (Opcode::Sextend, &[I16], &[I32], insert_opcode),
    (Opcode::Sextend, &[I16], &[I64], insert_opcode),
    (Opcode::Sextend, &[I16], &[I128], insert_opcode),
    (Opcode::Sextend, &[I32], &[I64], insert_opcode),
    (Opcode::Sextend, &[I32], &[I128], insert_opcode),
    (Opcode::Sextend, &[I64], &[I128], insert_opcode),
    // Ireduce
    (Opcode::Ireduce, &[I16], &[I8], insert_opcode),
    (Opcode::Ireduce, &[I32], &[I8], insert_opcode),
    (Opcode::Ireduce, &[I32], &[I16], insert_opcode),
    (Opcode::Ireduce, &[I64], &[I8], insert_opcode),
    (Opcode::Ireduce, &[I64], &[I16], insert_opcode),
    (Opcode::Ireduce, &[I64], &[I32], insert_opcode),
    (Opcode::Ireduce, &[I128], &[I8], insert_opcode),
    (Opcode::Ireduce, &[I128], &[I16], insert_opcode),
    (Opcode::Ireduce, &[I128], &[I32], insert_opcode),
    (Opcode::Ireduce, &[I128], &[I64], insert_opcode),
    // Isplit
    (Opcode::Isplit, &[I128], &[I64, I64], insert_opcode),
    // Iconcat
    (Opcode::Iconcat, &[I64, I64], &[I128], insert_opcode),
    // Fadd
    (Opcode::Fadd, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fadd, &[F64, F64], &[F64], insert_opcode),
    // Fmul
    (Opcode::Fmul, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fmul, &[F64, F64], &[F64], insert_opcode),
    // Fsub
    (Opcode::Fsub, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fsub, &[F64, F64], &[F64], insert_opcode),
    // Fdiv
    (Opcode::Fdiv, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fdiv, &[F64, F64], &[F64], insert_opcode),
    // Fmin
    (Opcode::Fmin, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fmin, &[F64, F64], &[F64], insert_opcode),
    // Fmax
    (Opcode::Fmax, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fmax, &[F64, F64], &[F64], insert_opcode),
    // FminPseudo
    (Opcode::FminPseudo, &[F32, F32], &[F32], insert_opcode),
    (Opcode::FminPseudo, &[F64, F64], &[F64], insert_opcode),
    // FmaxPseudo
    (Opcode::FmaxPseudo, &[F32, F32], &[F32], insert_opcode),
    (Opcode::FmaxPseudo, &[F64, F64], &[F64], insert_opcode),
    // Fcopysign
    (Opcode::Fcopysign, &[F32, F32], &[F32], insert_opcode),
    (Opcode::Fcopysign, &[F64, F64], &[F64], insert_opcode),
    // Fma
    (Opcode::Fma, &[F32, F32, F32], &[F32], insert_opcode),
    (Opcode::Fma, &[F64, F64, F64], &[F64], insert_opcode),
    // Fabs
    (Opcode::Fabs, &[F32], &[F32], insert_opcode),
    (Opcode::Fabs, &[F64], &[F64], insert_opcode),
    // Fneg
    (Opcode::Fneg, &[F32], &[F32], insert_opcode),
    (Opcode::Fneg, &[F64], &[F64], insert_opcode),
    // Sqrt
    (Opcode::Sqrt, &[F32], &[F32], insert_opcode),
    (Opcode::Sqrt, &[F64], &[F64], insert_opcode),
    // Ceil
    (Opcode::Ceil, &[F32], &[F32], insert_opcode),
    (Opcode::Ceil, &[F64], &[F64], insert_opcode),
    // Floor
    (Opcode::Floor, &[F32], &[F32], insert_opcode),
    (Opcode::Floor, &[F64], &[F64], insert_opcode),
    // Trunc
    (Opcode::Trunc, &[F32], &[F32], insert_opcode),
    (Opcode::Trunc, &[F64], &[F64], insert_opcode),
    // Nearest
    (Opcode::Nearest, &[F32], &[F32], insert_opcode),
    (Opcode::Nearest, &[F64], &[F64], insert_opcode),
    // Fcmp
    (Opcode::Fcmp, &[F32, F32], &[B1], insert_cmp),
    (Opcode::Fcmp, &[F64, F64], &[B1], insert_cmp),
    // Icmp
    (Opcode::Icmp, &[I8, I8], &[B1], insert_cmp),
    (Opcode::Icmp, &[I16, I16], &[B1], insert_cmp),
    (Opcode::Icmp, &[I32, I32], &[B1], insert_cmp),
    (Opcode::Icmp, &[I64, I64], &[B1], insert_cmp),
    // TODO: icmp of/nof broken for i128 on x86_64
    // See: https://github.com/bytecodealliance/wasmtime/issues/4406
    #[cfg(not(target_arch = "x86_64"))]
    (Opcode::Icmp, &[I128, I128], &[B1], insert_cmp),
    // Stack Access
    (Opcode::StackStore, &[I8], &[], insert_stack_store),
    (Opcode::StackStore, &[I16], &[], insert_stack_store),
    (Opcode::StackStore, &[I32], &[], insert_stack_store),
    (Opcode::StackStore, &[I64], &[], insert_stack_store),
    (Opcode::StackStore, &[I128], &[], insert_stack_store),
    (Opcode::StackLoad, &[], &[I8], insert_stack_load),
    (Opcode::StackLoad, &[], &[I16], insert_stack_load),
    (Opcode::StackLoad, &[], &[I32], insert_stack_load),
    (Opcode::StackLoad, &[], &[I64], insert_stack_load),
    (Opcode::StackLoad, &[], &[I128], insert_stack_load),
    // Integer Consts
    (Opcode::Iconst, &[], &[I8], insert_const),
    (Opcode::Iconst, &[], &[I16], insert_const),
    (Opcode::Iconst, &[], &[I32], insert_const),
    (Opcode::Iconst, &[], &[I64], insert_const),
    (Opcode::Iconst, &[], &[I128], insert_const),
    // Float Consts
    (Opcode::F32const, &[], &[F32], insert_const),
    (Opcode::F64const, &[], &[F64], insert_const),
    // Bool Consts
    (Opcode::Bconst, &[], &[B1], insert_const),
    // Call
    (Opcode::Call, &[], &[], insert_call),
];

/// These libcalls need a interpreter implementation in `cranelift-fuzzgen.rs`
const ALLOWED_LIBCALLS: &'static [LibCall] = &[
    LibCall::CeilF32,
    LibCall::CeilF64,
    LibCall::FloorF32,
    LibCall::FloorF64,
    LibCall::TruncF32,
    LibCall::TruncF64,
];

pub struct FunctionGenerator<'r, 'data>
where
    'data: 'r,
{
    u: &'r mut Unstructured<'data>,
    config: &'r Config,
    resources: Resources,
}

#[derive(Default)]
struct Resources {
    vars: HashMap<Type, Vec<Variable>>,
    blocks: Vec<(Block, BlockSignature)>,
    blocks_without_params: Vec<Block>,
    jump_tables: Vec<JumpTable>,
    func_refs: Vec<(Signature, FuncRef)>,
    stack_slots: Vec<(u32, StackSlot)>,
}

impl Resources {
    fn stack_slots_with_size(&self, n: u32) -> &[(u32, StackSlot)] {
        let first = self
            .stack_slots
            .partition_point(|&(bytes, _slot)| bytes < n);
        &self.stack_slots[first..]
    }

    fn get_variables_of_type(&self, ty: Type) -> &[Variable] {
        self.vars.get(&ty).map_or(&[][..], Vec::as_slice)
    }

    fn is_first_block(&self, block: Block) -> bool {
        self.blocks.first().map_or(false, |(b, _)| *b == block)
    }

    fn is_last_block(&self, block: Block) -> bool {
        self.blocks.last().map_or(false, |(b, _)| *b == block)
    }

    /// Generates a slice of blocks ahead of `block`
    fn forward_blocks(&self, block: Block) -> &[(Block, BlockSignature)] {
        let partition_point = self
            .blocks
            .iter()
            .enumerate()
            .find(|(i, (b, _))| *b > block)
            .map_or(self.blocks.len(), |(i, _)| i);

        let (_, forward_blocks) = self.blocks.split_at(partition_point);
        forward_blocks
    }

    /// Generates a slice of `blocks_without_params` ahead of `block`
    fn forward_blocks_without_params(&self, block: Block) -> &[Block] {
        let partition_point = self
            .blocks_without_params
            .iter()
            .enumerate()
            .find(|(i, b)| **b > block)
            .map_or(self.blocks_without_params.len(), |(i, _)| i);

        let (_, forward_blocks) = self.blocks_without_params.split_at(partition_point);
        forward_blocks
    }
}

impl<'r, 'data> FunctionGenerator<'r, 'data>
where
    'data: 'r,
{
    pub fn new(u: &'r mut Unstructured<'data>, config: &'r Config) -> Self {
        Self {
            u,
            config,
            resources: Resources::default(),
        }
    }

    /// Generates a random value for config `param`
    fn param(&mut self, param: &RangeInclusive<usize>) -> Result<usize> {
        Ok(self.u.int_in_range(param.clone())?)
    }

    fn generate_callconv(&mut self) -> Result<CallConv> {
        // TODO: Generate random CallConvs per target
        Ok(CallConv::SystemV)
    }

    fn system_callconv(&mut self) -> CallConv {
        // TODO: This currently only runs on linux, so this is the only choice
        // We should improve this once we generate flags and targets
        CallConv::SystemV
    }

    fn generate_type(&mut self) -> Result<Type> {
        // TODO: It would be nice if we could get these directly from cranelift
        let scalars = [
            // IFLAGS, FFLAGS,
            B1, // B8, B16, B32, B64, B128,
            I8, I16, I32, I64, I128, F32, F64,
            // R32, R64,
        ];
        // TODO: vector types

        let ty = self.u.choose(&scalars[..])?;
        Ok(*ty)
    }

    fn generate_abi_param(&mut self) -> Result<AbiParam> {
        let value_type = self.generate_type()?;
        // TODO: There are more argument purposes to be explored...
        let purpose = ArgumentPurpose::Normal;
        let extension = match self.u.int_in_range(0..=2)? {
            2 => ArgumentExtension::Sext,
            1 => ArgumentExtension::Uext,
            _ => ArgumentExtension::None,
        };

        Ok(AbiParam {
            value_type,
            purpose,
            extension,
        })
    }

    fn generate_signature(&mut self) -> Result<Signature> {
        let callconv = self.generate_callconv()?;
        let mut sig = Signature::new(callconv);

        for _ in 0..self.param(&self.config.signature_params)? {
            sig.params.push(self.generate_abi_param()?);
        }

        for _ in 0..self.param(&self.config.signature_rets)? {
            sig.returns.push(self.generate_abi_param()?);
        }

        Ok(sig)
    }

    /// Finds a stack slot with size of at least n bytes
    fn stack_slot_with_size(&mut self, n: u32) -> Result<StackSlot> {
        Ok(self.u.choose(self.resources.stack_slots_with_size(n))?.1)
    }

    /// Get a variable of type `ty` from the current function
    fn get_variable_of_type(&mut self, ty: Type) -> Result<Variable> {
        Ok(*self.u.choose(self.resources.get_variables_of_type(ty))?)
    }

    /// Chooses a random block which can be targeted by a jump / branch.
    /// This means any block that is not the first block.
    ///
    /// For convenience we also generate values that match the block's signature
    fn generate_target_block(
        &mut self,
        builder: &mut FunctionBuilder,
        source_block: Block,
    ) -> Result<(Block, Vec<Value>)> {
        // We try to mostly generate forward branches to avoid generating an excessive amount of
        // infinite loops. But they are still important, so give them a 1% chance of existing.
        let block_targets = if self.u.ratio(1, 100)? {
            &self.resources.blocks[..]
        } else {
            self.resources.forward_blocks(source_block)
        };
        let (block, signature) = self.u.choose(block_targets)?.clone();
        let args = self.generate_values_for_signature(builder, signature.into_iter())?;
        Ok((block, args))
    }

    fn generate_values_for_signature<I: Iterator<Item = Type>>(
        &mut self,
        builder: &mut FunctionBuilder,
        signature: I,
    ) -> Result<Vec<Value>> {
        signature
            .map(|ty| {
                let var = self.get_variable_of_type(ty)?;
                let val = builder.use_var(var);
                Ok(val)
            })
            .collect()
    }

    fn generate_return(
        &mut self,
        builder: &mut FunctionBuilder,
        _source_block: Block,
    ) -> Result<()> {
        let types: Vec<Type> = {
            let rets = &builder.func.signature.returns;
            rets.iter().map(|p| p.value_type).collect()
        };
        let vals = self.generate_values_for_signature(builder, types.into_iter())?;

        builder.ins().return_(&vals[..]);
        Ok(())
    }

    fn generate_jump(&mut self, builder: &mut FunctionBuilder, source_block: Block) -> Result<()> {
        let (block, args) = self.generate_target_block(builder, source_block)?;
        builder.ins().jump(block, &args[..]);
        Ok(())
    }

    /// Generates a br_table into a random block
    fn generate_br_table(
        &mut self,
        builder: &mut FunctionBuilder,
        source_block: Block,
    ) -> Result<()> {
        let var = self.get_variable_of_type(I32)?; // br_table only supports I32
        let val = builder.use_var(var);

        let target_blocks = self.resources.forward_blocks_without_params(source_block);
        let default_block = *self.u.choose(target_blocks)?;

        // We can still select a backwards branching jump table here!
        let jt = *self.u.choose(&self.resources.jump_tables)?;
        builder.ins().br_table(val, default_block, jt);
        Ok(())
    }

    /// Generates a brz/brnz into a random block
    fn generate_br(&mut self, builder: &mut FunctionBuilder, source_block: Block) -> Result<()> {
        let (block, args) = self.generate_target_block(builder, source_block)?;

        let condbr_types = [I8, I16, I32, I64, I128, B1];
        let _type = *self.u.choose(&condbr_types[..])?;
        let var = self.get_variable_of_type(_type)?;
        let val = builder.use_var(var);

        if bool::arbitrary(self.u)? {
            builder.ins().brz(val, block, &args[..]);
        } else {
            builder.ins().brnz(val, block, &args[..]);
        }

        // After brz/brnz we must generate a jump
        self.generate_jump(builder, source_block)?;
        Ok(())
    }

    fn generate_bricmp(
        &mut self,
        builder: &mut FunctionBuilder,
        source_block: Block,
    ) -> Result<()> {
        let (block, args) = self.generate_target_block(builder, source_block)?;
        let cond = *self.u.choose(IntCC::all())?;

        let bricmp_types = [
            I8, I16, I32,
            I64,
            // I128 - TODO: https://github.com/bytecodealliance/wasmtime/issues/4406
        ];
        let _type = *self.u.choose(&bricmp_types[..])?;

        let lhs_var = self.get_variable_of_type(_type)?;
        let lhs_val = builder.use_var(lhs_var);

        let rhs_var = self.get_variable_of_type(_type)?;
        let rhs_val = builder.use_var(rhs_var);

        builder
            .ins()
            .br_icmp(cond, lhs_val, rhs_val, block, &args[..]);

        // After bricmp's we must generate a jump
        self.generate_jump(builder, source_block)?;
        Ok(())
    }

    fn generate_switch(
        &mut self,
        builder: &mut FunctionBuilder,
        source_block: Block,
    ) -> Result<()> {
        let _type = *self.u.choose(&[I8, I16, I32, I64, I128][..])?;
        let switch_var = self.get_variable_of_type(_type)?;
        let switch_val = builder.use_var(switch_var);

        let default_block = {
            let target_blocks = self.resources.forward_blocks_without_params(source_block);
            *self.u.choose(target_blocks)?
        };

        // Build this into a HashMap since we cannot have duplicate entries.
        let mut entries = HashMap::new();
        for _ in 0..self.param(&self.config.switch_cases)? {
            // The Switch API only allows for entries that are addressable by the index type
            // so we need to limit the range of values that we generate.
            let (ty_min, ty_max) = _type.bounds(false);
            let range_start = self.u.int_in_range(ty_min..=ty_max)?;

            // We can either insert a contiguous range of blocks or a individual block
            // This is done because the Switch API specializes contiguous ranges.
            let range_size = if bool::arbitrary(self.u)? {
                1
            } else {
                self.param(&self.config.switch_max_range_size)?
            } as u128;

            // Build the switch entries
            for i in 0..range_size {
                let index = range_start.wrapping_add(i) % ty_max;
                let block = {
                    let target_blocks = self.resources.forward_blocks_without_params(source_block);
                    *self.u.choose(target_blocks)?
                };

                entries.insert(index, block);
            }
        }

        let mut switch = Switch::new();
        for (entry, block) in entries.into_iter() {
            switch.set_entry(entry, block);
        }
        switch.emit(builder, switch_val, default_block);

        Ok(())
    }

    /// We always need to exit safely out of a block.
    /// This either means a jump into another block or a return.
    fn finalize_block(&mut self, builder: &mut FunctionBuilder, source_block: Block) -> Result<()> {
        // We do mostly forward branching, the last block has no further blocks to jump to
        // so generate a return.
        if self.resources.is_last_block(source_block) {
            self.generate_return(builder, source_block)?;
        } else {
            let gen = self.u.choose(
                &[
                    Self::generate_bricmp,
                    Self::generate_br,
                    Self::generate_br_table,
                    Self::generate_jump,
                    Self::generate_return,
                    Self::generate_switch,
                ][..],
            )?;

            gen(self, builder, source_block)?;
        }

        Ok(())
    }

    /// Fills the current block with random instructions
    fn generate_instructions(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        for _ in 0..self.param(&self.config.instructions_per_block)? {
            // I know this is terrible, don't judge me
            let inst_templates = generate_instruction_templates(&self.resources);

            let template = self.u.choose(&inst_templates[..])?;
            template(self.u, builder)?;
            // let (op, args, rets, inserter) = *self.u.choose(OPCODE_SIGNATURES)?;
            // inserter(self, builder, op, args, rets)?;
        }

        Ok(())
    }

    fn generate_jumptables(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        for _ in 0..self.param(&self.config.jump_tables_per_function)? {
            let mut jt_data = JumpTableData::new();

            for _ in 0..self.param(&self.config.jump_table_entries)? {
                let block = *self.u.choose(&self.resources.blocks_without_params)?;
                jt_data.push_entry(block);
            }

            self.resources
                .jump_tables
                .push(builder.create_jump_table(jt_data));
        }
        Ok(())
    }

    fn generate_funcrefs(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        let count = self.param(&self.config.funcrefs_per_function)?;
        for func_index in 0..count.try_into().unwrap() {
            let (ext_name, sig) = if self.u.arbitrary::<bool>()? {
                let user_func_ref = builder
                    .func
                    .declare_imported_user_function(UserExternalName {
                        namespace: 0,
                        index: func_index,
                    });
                let name = ExternalName::User(user_func_ref);
                let signature = self.generate_signature()?;
                (name, signature)
            } else {
                let libcall = *self.u.choose(ALLOWED_LIBCALLS)?;
                // TODO: Use [CallConv::for_libcall] once we generate flags.
                let callconv = self.system_callconv();
                let signature = libcall.signature(callconv);
                (ExternalName::LibCall(libcall), signature)
            };

            let sig_ref = builder.import_signature(sig.clone());
            let func_ref = builder.import_function(ExtFuncData {
                name: ext_name,
                signature: sig_ref,
                colocated: self.u.arbitrary()?,
            });

            self.resources.func_refs.push((sig, func_ref));
        }

        Ok(())
    }

    fn generate_stack_slots(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        for _ in 0..self.param(&self.config.static_stack_slots_per_function)? {
            let bytes = self.param(&self.config.static_stack_slot_size)? as u32;
            let ss_data = StackSlotData::new(StackSlotKind::ExplicitSlot, bytes);
            let slot = builder.create_sized_stack_slot(ss_data);

            self.resources.stack_slots.push((bytes, slot));
        }

        self.resources
            .stack_slots
            .sort_unstable_by_key(|&(bytes, _slot)| bytes);

        Ok(())
    }

    /// Zero initializes the stack slot by inserting `stack_store`'s.
    fn initialize_stack_slots(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        let i128_zero = builder.ins().iconst(I128, 0);
        let i64_zero = builder.ins().iconst(I64, 0);
        let i32_zero = builder.ins().iconst(I32, 0);
        let i16_zero = builder.ins().iconst(I16, 0);
        let i8_zero = builder.ins().iconst(I8, 0);

        for &(init_size, slot) in self.resources.stack_slots.iter() {
            let mut size = init_size;

            // Insert the largest available store for the remaining size.
            while size != 0 {
                let offset = (init_size - size) as i32;
                let (val, filled) = match size {
                    sz if sz / 16 > 0 => (i128_zero, 16),
                    sz if sz / 8 > 0 => (i64_zero, 8),
                    sz if sz / 4 > 0 => (i32_zero, 4),
                    sz if sz / 2 > 0 => (i16_zero, 2),
                    _ => (i8_zero, 1),
                };
                builder.ins().stack_store(val, slot, offset);
                size -= filled;
            }
        }
        Ok(())
    }

    /// Creates a random amount of blocks in this function
    fn generate_blocks(
        &mut self,
        builder: &mut FunctionBuilder,
        sig: &Signature,
    ) -> Result<Vec<(Block, BlockSignature)>> {
        let extra_block_count = self.param(&self.config.blocks_per_function)?;

        // We must always have at least one block, so we generate the "extra" blocks and add 1 for
        // the entry block.
        let block_count = 1 + extra_block_count;

        // Blocks need to be sorted in ascending order
        (0..block_count)
            .map(|i| {
                let is_entry = i == 0;
                let block = builder.create_block();

                // Optionally mark blocks that are not the entry block as cold
                if !is_entry {
                    if bool::arbitrary(self.u)? {
                        builder.set_cold_block(block);
                    }
                }

                // The first block has to have the function signature, but for the rest of them we generate
                // a random signature;
                if is_entry {
                    builder.append_block_params_for_function_params(block);
                    Ok((block, sig.params.iter().map(|a| a.value_type).collect()))
                } else {
                    let sig = self.generate_block_signature()?;
                    sig.iter().for_each(|ty| {
                        builder.append_block_param(block, *ty);
                    });
                    Ok((block, sig))
                }
            })
            .collect()
    }

    fn generate_block_signature(&mut self) -> Result<BlockSignature> {
        let param_count = self.param(&self.config.block_signature_params)?;

        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(self.generate_type()?);
        }
        Ok(params)
    }

    fn build_variable_pool(&mut self, builder: &mut FunctionBuilder) -> Result<()> {
        let block = builder.current_block().unwrap();

        // Define variables for the function signature
        let mut vars: Vec<_> = builder
            .func
            .signature
            .params
            .iter()
            .map(|param| param.value_type)
            .zip(builder.block_params(block).iter().copied())
            .collect();

        // Create a pool of vars that are going to be used in this function
        for _ in 0..self.param(&self.config.vars_per_function)? {
            let ty = self.generate_type()?;
            let value = generate_const(&mut self.u, builder, ty)?;
            vars.push((ty, value));
        }

        for (id, (ty, value)) in vars.into_iter().enumerate() {
            let var = Variable::new(id);
            builder.declare_var(var, ty);
            builder.def_var(var, value);
            self.resources
                .vars
                .entry(ty)
                .or_insert_with(Vec::new)
                .push(var);
        }

        Ok(())
    }

    /// We generate a function in multiple stages:
    ///
    /// * First we generate a random number of empty blocks
    /// * Then we generate a random pool of variables to be used throughout the function
    /// * We then visit each block and generate random instructions
    ///
    /// Because we generate all blocks and variables up front we already know everything that
    /// we need when generating instructions (i.e. jump targets / variables)
    pub fn generate(mut self) -> Result<Function> {
        let sig = self.generate_signature()?;

        let mut fn_builder_ctx = FunctionBuilderContext::new();
        // function name must be in a different namespace than TESTFILE_NAMESPACE (0)
        let mut func = Function::with_name_signature(UserFuncName::user(1, 0), sig.clone());

        let mut builder = FunctionBuilder::new(&mut func, &mut fn_builder_ctx);

        self.resources.blocks = self.generate_blocks(&mut builder, &sig)?;

        // Valid blocks for jump tables have to have no parameters in the signature, and must also
        // not be the first block.
        self.resources.blocks_without_params = self.resources.blocks[1..]
            .iter()
            .filter(|(_, sig)| sig.len() == 0)
            .map(|(b, _)| *b)
            .collect();

        // Function preamble
        self.generate_jumptables(&mut builder)?;
        self.generate_funcrefs(&mut builder)?;
        self.generate_stack_slots(&mut builder)?;

        // Main instruction generation loop
        for (i, (block, block_sig)) in self.resources.blocks.clone().iter().enumerate() {
            let is_block0 = i == 0;
            builder.switch_to_block(*block);

            if is_block0 {
                // The first block is special because we must create variables both for the
                // block signature and for the variable pool. Additionally, we must also define
                // initial values for all variables that are not the function signature.
                self.build_variable_pool(&mut builder)?;

                // Stack slots have random bytes at the beginning of the function
                // initialize them to a constant value so that execution stays predictable.
                self.initialize_stack_slots(&mut builder)?;
            } else {
                // Define variables for the block params
                for (i, ty) in block_sig.iter().enumerate() {
                    let var = self.get_variable_of_type(*ty)?;
                    let block_param = builder.block_params(*block)[i];
                    builder.def_var(var, block_param);
                }
            }

            // Generate block instructions
            self.generate_instructions(&mut builder)?;

            self.finalize_block(&mut builder, *block)?;
        }

        builder.seal_all_blocks();
        builder.finalize();

        Ok(func)
    }
}
