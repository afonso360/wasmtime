//! Generate Wasm modules that contain a single instruction.

use super::ModuleConfig;
use arbitrary::Unstructured;
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

/// The name of the function generated by this module.
const FUNCTION_NAME: &'static str = "test";

/// Configure a single instruction module.
///
/// By explicitly defining the parameter and result types (versus generating the
/// module directly), we can more easily generate values of the right type.
#[derive(Clone)]
pub struct SingleInstModule<'a> {
    instruction: Instruction<'a>,
    parameters: &'a [ValType],
    results: &'a [ValType],
    feature: fn(&ModuleConfig) -> bool,
}

impl<'a> SingleInstModule<'a> {
    /// Choose a single-instruction module that matches `config`.
    pub fn new(u: &mut Unstructured<'a>, config: &mut ModuleConfig) -> arbitrary::Result<&'a Self> {
        // To avoid skipping modules unnecessarily during fuzzing, fix up the
        // `ModuleConfig` to match the inherent limits of a single-instruction
        // module.
        config.config.min_funcs = 1;
        config.config.max_funcs = 1;
        config.config.min_tables = 0;
        config.config.max_tables = 0;
        config.config.min_memories = 0;
        config.config.max_memories = 0;

        // Only select instructions that match the `ModuleConfig`.
        let instructions = &INSTRUCTIONS
            .iter()
            .filter(|i| (i.feature)(config))
            .collect::<Vec<_>>();
        u.choose(&instructions[..]).copied()
    }

    /// Encode a binary Wasm module with a single exported function, `test`,
    /// that executes the single instruction.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut module = Module::new();

        // Encode the type section.
        let mut types = TypeSection::new();
        types.function(
            self.parameters.iter().cloned(),
            self.results.iter().cloned(),
        );
        module.section(&types);

        // Encode the function section.
        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        // Encode the export section.
        let mut exports = ExportSection::new();
        exports.export(FUNCTION_NAME, ExportKind::Func, 0);
        module.section(&exports);

        // Encode the code section.
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        for (index, _) in self.parameters.iter().enumerate() {
            f.instruction(&Instruction::LocalGet(index as u32));
        }
        f.instruction(&self.instruction);
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);

        // Extract the encoded Wasm bytes for this module.
        module.finish()
    }
}

// MACROS
//
// These macros make it a bit easier to define the instructions available for
// generation. The idea is that, with these macros, we can define the list of
// instructions compactly and allow for easier changes to the Rust code (e.g.,
// `SingleInstModule`).

macro_rules! valtype {
    (i32) => {
        ValType::I32
    };
    (i64) => {
        ValType::I64
    };
    (f32) => {
        ValType::F32
    };
    (f64) => {
        ValType::F64
    };
}

macro_rules! binary {
    ($inst:ident, $rust_ty:tt) => {
        binary! { $inst, $rust_ty, $rust_ty }
    };
    ($inst:ident, $arguments_ty:tt,  $result_ty:tt) => {
        SingleInstModule {
            instruction: Instruction::$inst,
            parameters: &[valtype!($arguments_ty), valtype!($arguments_ty)],
            results: &[valtype!($result_ty)],
            feature: |_| true,
        }
    };
}

macro_rules! compare {
    ($inst:ident, $rust_ty:tt) => {
        binary! { $inst, $rust_ty, i32 }
    };
}

macro_rules! unary {
    ($inst:ident, $rust_ty:tt) => {
        unary! { $inst, $rust_ty, $rust_ty }
    };
    ($inst:ident, $argument_ty:tt, $result_ty:tt) => {
        SingleInstModule {
            instruction: Instruction::$inst,
            parameters: &[valtype!($argument_ty)],
            results: &[valtype!($result_ty)],
            feature: |_| true,
        }
    };
    ($inst:ident, $argument_ty:tt, $result_ty:tt, $feature:expr) => {
        SingleInstModule {
            instruction: Instruction::$inst,
            parameters: &[valtype!($argument_ty)],
            results: &[valtype!($result_ty)],
            feature: $feature,
        }
    };
}

macro_rules! convert {
    ($inst:ident, $from_ty:tt -> $to_ty:tt) => {
        unary! { $inst, $from_ty, $to_ty }
    };
    ($inst:ident, $from_ty:tt -> $to_ty:tt, $feature:expr) => {
        unary! { $inst, $from_ty, $to_ty, $feature }
    };
}

static INSTRUCTIONS: &[SingleInstModule] = &[
    // Integer arithmetic.
    // I32Const
    // I64Const
    // F32Const
    // F64Const
    unary!(I32Clz, i32),
    unary!(I64Clz, i64),
    unary!(I32Ctz, i32),
    unary!(I64Ctz, i64),
    unary!(I32Popcnt, i32),
    unary!(I64Popcnt, i64),
    binary!(I32Add, i32),
    binary!(I64Add, i64),
    binary!(I32Sub, i32),
    binary!(I64Sub, i64),
    binary!(I32Mul, i32),
    binary!(I64Mul, i64),
    binary!(I32DivS, i32),
    binary!(I64DivS, i64),
    binary!(I32DivU, i32),
    binary!(I64DivU, i64),
    binary!(I32RemS, i32),
    binary!(I64RemS, i64),
    binary!(I32RemU, i32),
    binary!(I64RemU, i64),
    // Integer bitwise.
    binary!(I32And, i32),
    binary!(I64And, i64),
    binary!(I32Or, i32),
    binary!(I64Or, i64),
    binary!(I32Xor, i32),
    binary!(I64Xor, i64),
    binary!(I32Shl, i32),
    binary!(I64Shl, i64),
    binary!(I32ShrS, i32),
    binary!(I64ShrS, i64),
    binary!(I32ShrU, i32),
    binary!(I64ShrU, i64),
    binary!(I32Rotl, i32),
    binary!(I64Rotl, i64),
    binary!(I32Rotr, i32),
    binary!(I64Rotr, i64),
    // Integer comparison.
    unary!(I32Eqz, i32),
    unary!(I64Eqz, i64, i32),
    compare!(I32Eq, i32),
    compare!(I64Eq, i64),
    compare!(I32Ne, i32),
    compare!(I64Ne, i64),
    compare!(I32LtS, i32),
    compare!(I64LtS, i64),
    compare!(I32LtU, i32),
    compare!(I64LtU, i64),
    compare!(I32GtS, i32),
    compare!(I64GtS, i64),
    compare!(I32GtU, i32),
    compare!(I64GtU, i64),
    compare!(I32LeS, i32),
    compare!(I64LeS, i64),
    compare!(I32LeU, i32),
    compare!(I64LeU, i64),
    compare!(I32GeS, i32),
    compare!(I64GeS, i64),
    compare!(I32GeU, i32),
    compare!(I64GeU, i64),
    // Floating-point arithmetic.
    unary!(F32Abs, f32),
    unary!(F64Abs, f64),
    unary!(F32Sqrt, f32),
    unary!(F64Sqrt, f64),
    unary!(F32Ceil, f32),
    unary!(F64Ceil, f64),
    unary!(F32Floor, f32),
    unary!(F64Floor, f64),
    unary!(F32Trunc, f32),
    unary!(F64Trunc, f64),
    unary!(F32Nearest, f32),
    unary!(F64Nearest, f64),
    unary!(F32Neg, f32),
    unary!(F64Neg, f64),
    binary!(F32Add, f32),
    binary!(F64Add, f64),
    binary!(F32Sub, f32),
    binary!(F64Sub, f64),
    binary!(F32Mul, f32),
    binary!(F64Mul, f64),
    binary!(F32Div, f32),
    binary!(F64Div, f64),
    binary!(F32Min, f32),
    binary!(F64Min, f64),
    binary!(F32Max, f32),
    binary!(F64Max, f64),
    binary!(F32Copysign, f32),
    binary!(F64Copysign, f64),
    // Floating-point comparison.
    compare!(F32Eq, f32),
    compare!(F64Eq, f64),
    compare!(F32Ne, f32),
    compare!(F64Ne, f64),
    compare!(F32Lt, f32),
    compare!(F64Lt, f64),
    compare!(F32Gt, f32),
    compare!(F64Gt, f64),
    compare!(F32Le, f32),
    compare!(F64Le, f64),
    compare!(F32Ge, f32),
    compare!(F64Ge, f64),
    // Integer conversions ("to integer").
    unary!(I32Extend8S, i32, i32, |c| c.config.sign_extension_enabled),
    unary!(I32Extend16S, i32, i32, |c| c.config.sign_extension_enabled),
    unary!(I64Extend8S, i64, i64, |c| c.config.sign_extension_enabled),
    unary!(I64Extend16S, i64, i64, |c| c.config.sign_extension_enabled),
    convert!(I64Extend32S, i64 -> i64, |c| c.config.sign_extension_enabled),
    convert!(I32WrapI64, i64 -> i32),
    convert!(I64ExtendI32S, i32 -> i64),
    convert!(I64ExtendI32U, i32 -> i64),
    convert!(I32TruncF32S, f32 -> i32),
    convert!(I32TruncF32U, f32 -> i32),
    convert!(I32TruncF64S, f64 -> i32),
    convert!(I32TruncF64U, f64 -> i32),
    convert!(I64TruncF32S, f32 -> i64),
    convert!(I64TruncF32U, f32 -> i64),
    convert!(I64TruncF64S, f64 -> i64),
    convert!(I64TruncF64U, f64 -> i64),
    convert!(I32TruncSatF32S, f32 -> i32, |c| c.config.saturating_float_to_int_enabled),
    convert!(I32TruncSatF32U, f32 -> i32, |c| c.config.saturating_float_to_int_enabled),
    convert!(I32TruncSatF64S, f64 -> i32, |c| c.config.saturating_float_to_int_enabled),
    convert!(I32TruncSatF64U, f64 -> i32, |c| c.config.saturating_float_to_int_enabled),
    convert!(I64TruncSatF32S, f32 -> i64, |c| c.config.saturating_float_to_int_enabled),
    convert!(I64TruncSatF32U, f32 -> i64, |c| c.config.saturating_float_to_int_enabled),
    convert!(I64TruncSatF64S, f64 -> i64, |c| c.config.saturating_float_to_int_enabled),
    convert!(I64TruncSatF64U, f64 -> i64, |c| c.config.saturating_float_to_int_enabled),
    convert!(I32ReinterpretF32, f32 -> i32),
    convert!(I64ReinterpretF64, f64 -> i64),
    // Floating-point conversions ("to float").
    convert!(F32DemoteF64, f64 -> f32),
    convert!(F64PromoteF32, f32 -> f64),
    convert!(F32ConvertI32S, i32 -> f32),
    convert!(F32ConvertI32U, i32 -> f32),
    convert!(F32ConvertI64S, i64 -> f32),
    convert!(F32ConvertI64U, i64 -> f32),
    convert!(F64ConvertI32S, i32 -> f64),
    convert!(F64ConvertI32U, i32 -> f64),
    convert!(F64ConvertI64S, i64 -> f64),
    convert!(F64ConvertI64U, i64 -> f64),
    convert!(F32ReinterpretI32, i32 -> f32),
    convert!(F64ReinterpretI64, i64 -> f64),
];

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity() {
        let sut = SingleInstModule {
            instruction: Instruction::I32Add,
            parameters: &[ValType::I32, ValType::I32],
            results: &[ValType::I32],
            feature: |_| true,
        };
        let wasm = sut.to_bytes();
        let wat = wasmprinter::print_bytes(wasm).unwrap();
        assert_eq!(
            wat,
            r#"(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (func (;0;) (type 0) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
  (export "test" (func 0))
)"#
        )
    }

    #[test]
    fn instructions_encode_to_valid_modules() {
        for inst in INSTRUCTIONS {
            assert!(wat::parse_bytes(&inst.to_bytes()).is_ok());
        }
    }
}
