//! Test command for testing the jump threading pass.

use crate::subtest::{check_precise_output, run_filecheck, Context, SubTest};
use anyhow::Result;
use cranelift_codegen::ir;
use cranelift_reader::{TestCommand, TestOption};
use std::borrow::Cow;

struct TestJumpThreading {
    /// Flag indicating that the text expectation, comments after the function,
    /// must be a precise 100% match on the compiled output of the function.
    /// This test assertion is also automatically-update-able to allow tweaking
    /// the code generator and easily updating all affected tests.
    precise_output: bool,
}

pub fn subtest(parsed: &TestCommand) -> Result<Box<dyn SubTest>> {
    assert_eq!(parsed.command, "jump-threading");
    let mut test = TestJumpThreading {
        precise_output: false,
    };
    for option in parsed.options.iter() {
        match option {
            TestOption::Flag("precise-output") => test.precise_output = true,
            _ => anyhow::bail!("unknown option on {}", parsed),
        }
    }
    Ok(Box::new(test))
}

impl SubTest for TestJumpThreading {
    fn name(&self) -> &'static str {
        "jump-threading"
    }

    fn is_mutating(&self) -> bool {
        true
    }

    fn needs_isa(&self) -> bool {
        true
    }

    fn run(&self, func: Cow<ir::Function>, context: &Context) -> Result<()> {
        let isa = context.isa.expect("jump-threading needs an ISA");
        let mut comp_ctx = cranelift_codegen::Context::for_function(func.into_owned());

        comp_ctx.compute_cfg();
        comp_ctx.compute_domtree();
        comp_ctx.compute_loop_analysis();

        comp_ctx
            .jump_threading_pass(isa)
            .map_err(|e| crate::pretty_anyhow_error(&comp_ctx.func, e))?;

        let clif = format!("{:?}", comp_ctx.func);

        if self.precise_output {
            let actual: Vec<_> = clif.lines().collect();
            check_precise_output(&actual, context)
        } else {
            run_filecheck(&clif, context)
        }
    }
}
