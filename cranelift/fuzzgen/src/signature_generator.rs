use crate::arbitrary_ext::ArbitraryClifExt;
use crate::codegen::ir::{ArgumentExtension, ArgumentPurpose};
use crate::config::Config;
use anyhow::Result;
use arbitrary::Unstructured;
use cranelift::codegen::ir::{AbiParam, Signature};
pub struct SignatureGenerator<'r, 'data>
where
    'data: 'r,
{
    u: &'r mut Unstructured<'data>,
    config: &'r Config,
}

impl<'r, 'data> SignatureGenerator<'r, 'data>
where
    'data: 'r,
{
    pub fn new(u: &'r mut Unstructured<'data>, config: &'r Config) -> Self {
        Self { u, config }
    }

    fn generate_abi_param(&mut self) -> Result<AbiParam> {
        let value_type = self.u.clif_type()?;
        // TODO: There are more argument purposes to be explored...
        let purpose = ArgumentPurpose::Normal;
        let extension = *self.u.choose(&[
            ArgumentExtension::Sext,
            ArgumentExtension::Uext,
            ArgumentExtension::None,
        ])?;

        Ok(AbiParam {
            value_type,
            purpose,
            extension,
        })
    }

    fn generate_signature(mut self) -> Result<Signature> {
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
}
