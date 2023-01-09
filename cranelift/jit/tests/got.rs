//! Tests Global Offset Table entries.

use codegen::{
    ir::{Function, UserFuncName},
    Context,
};
use cranelift::prelude::{types::I64, *};
use cranelift_codegen::{
    isa::{self, CallConv},
    settings,
};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, DataContext, Linkage, Module};
use target_lexicon::Triple;

#[test]
// Currently we only support GOT values in x86
#[cfg_attr(not(target_arch = "x86_64"), ignore)]
fn read_got_value() -> Result<(), anyhow::Error> {
    let host = Triple::host();
    let isa_builder = isa::lookup(host.clone())?;
    let mut settings_builder = settings::builder();
    settings_builder.set("is_pic", "true")?;

    let isa_flags = settings::Flags::new(settings_builder);
    let isa = isa_builder.finish(isa_flags)?;

    let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));

    // Define the variable
    let var_global_id = module.declare_data("variable-name-x", Linkage::Local, false, false)?;
    let mut data = DataContext::new();
    data.define(Box::new(32i64.to_le_bytes()));
    module.define_data(var_global_id, &data)?;

    // Define the hello function
    // It reads from our imported variable, adds 1 and returns the value.
    let signature = Signature {
        params: vec![],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::triple_default(&host),
    };
    let hello_id = module.declare_function("hello", Linkage::Local, &signature)?;
    let mut ctx = Context::new();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, hello_id.as_u32()), signature);

    // local versions of the object globals
    let var_local_id = module.declare_data_in_func(var_global_id, &mut ctx.func);

    // set up the block
    let mut fctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let main_block = builder.create_block();
    builder.switch_to_block(main_block);

    // Read our value from the GOT
    let var_ptr = builder.ins().symbol_value(I64, var_local_id);
    let var_val = builder.ins().load(I64, MemFlags::trusted(), var_ptr, 0);

    // Add 10 to it.
    let sum = builder.ins().iadd_imm(var_val, 10);

    // close it all up
    builder.ins().return_(&[sum]);
    builder.seal_all_blocks();
    builder.finalize();
    module.define_function(hello_id, &mut ctx)?;
    module.finalize_definitions()?;

    // Get the function pointer and call it
    let function_ptr = module.get_finalized_function(hello_id);
    let callable: fn() -> i64 = unsafe { std::mem::transmute(function_ptr) };
    let res = callable();

    assert_eq!(res, 42);

    Ok(())
}
