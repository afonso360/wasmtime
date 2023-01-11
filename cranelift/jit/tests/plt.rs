//! Tests calling a function via the PLT.

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
use cranelift_module::{default_libcall_names, Linkage, Module};
use target_lexicon::Triple;

#[test]
// Currently we only support the PLT in x86 and aarch64
#[cfg_attr(not(any(target_arch = "x86_64", target_arch = "aarch64")), ignore)]
fn plt_call() -> Result<(), anyhow::Error> {
    let host = Triple::host();
    let isa_builder = isa::lookup(host.clone())?;
    let mut settings_builder = settings::builder();
    settings_builder.set("is_pic", "true")?;

    let isa_flags = settings::Flags::new(settings_builder);
    let isa = isa_builder.finish(isa_flags)?;

    // To force all function calls to go through the PLT use hotswap mode.
    let mut jit_builder = JITBuilder::with_isa(isa, default_libcall_names());
    jit_builder.hotswap(true);
    let mut module = JITModule::new(jit_builder);

    // Define the callee function
    // It adds 10 to whatever number is passed into it.
    let callee_sig = Signature {
        params: vec![AbiParam::new(I64)],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::triple_default(&host),
    };
    let callee = module.declare_function("callee", Linkage::Local, &callee_sig)?;

    // Define the caller function
    // It just calls callee
    let caller_sig = Signature {
        params: vec![],
        returns: vec![AbiParam::new(I64)],
        call_conv: CallConv::triple_default(&host),
    };
    let caller = module.declare_function("caller", Linkage::Local, &caller_sig)?;

    let mut fctx = FunctionBuilderContext::new();
    let mut ctx = Context::new();
    ctx.func = Function::with_name_signature(UserFuncName::user(0, callee.as_u32()), callee_sig);

    // Set up the function
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let main_block = builder.create_block();
        builder.append_block_params_for_function_params(main_block);
        builder.switch_to_block(main_block);

        let input = builder.block_params(main_block)[0];
        let sum = builder.ins().iadd_imm(input, 10);
        builder.ins().return_(&[sum]);

        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(callee, &mut ctx)?;
    module.clear_context(&mut ctx);

    // ----------
    // ----------
    // ----------
    // ----------
    // ----------

    // Set up the function
    ctx.func = Function::with_name_signature(UserFuncName::user(0, caller.as_u32()), caller_sig);
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let main_block = builder.create_block();
        builder.switch_to_block(main_block);

        // Declare the callee here.
        let callee_func = module.declare_func_in_func(callee, &mut builder.func);
        let base = builder.ins().iconst(I64, 32);
        let call = builder.ins().call(callee_func, &[base]);
        let value = builder.inst_results(call)[0];
        builder.ins().return_(&[value]);

        builder.seal_all_blocks();
        builder.finalize();
    }
    module.define_function(caller, &mut ctx)?;
    module.clear_context(&mut ctx);

    // Perform linking.
    module.finalize_definitions().unwrap();

    // Get the caller function pointer and call it
    let function_ptr = module.get_finalized_function(caller);
    let callable: fn() -> i64 = unsafe { std::mem::transmute(function_ptr) };
    let res = callable();

    assert_eq!(res, 42);

    Ok(())
}
