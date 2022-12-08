use anyhow::Result;
use arbitrary::{Arbitrary, Unstructured};
use cranelift::codegen::data_value::DataValue;
use cranelift::prelude::types::*;
use cranelift::prelude::Ieee32;
use cranelift::prelude::Ieee64;
use cranelift::prelude::Type;

/// Provides an extension to [Arbitrary] that allows to generate
/// CLIF types and other clif structure.
pub trait ArbitraryClifExt {
    fn clif_type(&mut self) -> Result<Type>;
    fn datavalue(&mut self, ty: Type) -> Result<DataValue>;
}

impl<'a> ArbitraryClifExt for Unstructured<'a> {
    fn clif_type(&mut self) -> Result<Type> {
        // TODO: It would be nice if we could get these directly from cranelift
        let scalars = [
            // IFLAGS, FFLAGS,
            I8, I16, I32, I64, I128, F32, F64,
            // R32, R64,
        ];
        // TODO: vector types

        let ty = self.choose(&scalars[..])?;
        Ok(*ty)
    }

    fn datavalue(&mut self, ty: Type) -> Result<DataValue> {
        Ok(match ty {
            ty if ty.is_int() => {
                let imm = match ty {
                    I8 => self.arbitrary::<i8>()? as i128,
                    I16 => self.arbitrary::<i16>()? as i128,
                    I32 => self.arbitrary::<i32>()? as i128,
                    I64 => self.arbitrary::<i64>()? as i128,
                    I128 => self.arbitrary::<i128>()?,
                    _ => unreachable!(),
                };
                DataValue::from_integer(imm, ty)?
            }
            // f{32,64}::arbitrary does not generate a bunch of important values
            // such as Signaling NaN's / NaN's with payload, so generate floats from integers.
            F32 => DataValue::F32(Ieee32::with_bits(u32::arbitrary(self)?)),
            F64 => DataValue::F64(Ieee64::with_bits(u64::arbitrary(self)?)),
            _ => unimplemented!(),
        })
    }
}
