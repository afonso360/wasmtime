//! CLI tool to read Cranelift IR files and compile them into native code.

use crate::disasm::{print_all, PrintRelocs, PrintStackMaps, PrintTraps};
use crate::utils::{parse_sets_and_triple, read_to_string, OwnedFlagsOrIsa};
use anyhow::{Context as _, Result};
use cranelift::codegen::binemit::{NullStackMapSink, NullTrapSink};
use cranelift::codegen::isa::TargetIsa;
use cranelift_codegen::print_errors::pretty_error;
use cranelift_codegen::settings::FlagsOrIsa;
use cranelift_codegen::timing;
use cranelift_codegen::Context;
use cranelift_module::{DataContext, Linkage, Module};
use cranelift_native::builder as host_isa_builder;
use cranelift_object::{ObjectBuilder, ObjectModule};
use cranelift_reader::{parse_test, ParseOptions, TestFile};
use std::path::Path;
use std::path::PathBuf;
use structopt::StructOpt;

/// Compiles Cranelift IR into target language
#[derive(StructOpt)]
pub struct Options {
    /// Print the resulting Cranelift IR
    #[structopt(short("p"))]
    print: bool,

    /// Print pass timing report
    #[structopt(short("T"))]
    report_times: bool,

    /// Print machine code disassembly
    #[structopt(short("D"), long("disasm"))]
    disasm: bool,

    /// Configure Cranelift settings
    #[structopt(long("set"))]
    settings: Vec<String>,

    /// Specify the Cranelift target
    #[structopt(long("target"))]
    target: String,

    /// Specify an input file to be used. Use '-' for stdin.
    #[structopt(parse(from_os_str))]
    files: Vec<PathBuf>,

    /// Enable debug output on stderr/stdout
    #[structopt(short = "d")]
    debug: bool,

    /// Output the object file to
    #[structopt(short = "o")]
    output: Option<PathBuf>,
}

pub fn run(options: &Options) -> Result<()> {
    crate::handle_debug_flag(options.debug);
    let parsed = parse_sets_and_triple(&options.settings, &options.target)?;
    for path in &options.files {
        let name = String::from(path.as_os_str().to_string_lossy());
        handle_module(options, path, &name, parsed.as_fisa())?;
    }
    Ok(())
}

fn handle_module(options: &Options, path: &Path, name: &str, fisa: FlagsOrIsa) -> Result<()> {
    let buffer = read_to_string(&path)?;
    let test_file = parse_test(&buffer, ParseOptions::default())
        .with_context(|| format!("failed to parse {}", name))?;

    // If we have an isa from the command-line, use that. Otherwise if the
    // file contains a unique isa, use that.
    let isa = fisa.isa.or(test_file.isa_spec.unique_isa());

    if isa.is_none() {
        anyhow::bail!("compilation requires a target isa");
    };

    for (func, _) in &test_file.functions {
        let mut relocs = PrintRelocs::new(options.print);
        let mut traps = PrintTraps::new(options.print);
        let mut stack_maps = PrintStackMaps::new(options.print);

        if let Some(isa) = isa {
            let mut context = Context::new();
            context.func = func.clone();
            let mut mem = vec![];

            // Compile and encode the result to machine code.
            let code_info = context
                .compile_and_emit(isa, &mut mem, &mut relocs, &mut traps, &mut stack_maps)
                .map_err(|err| {
                    anyhow::anyhow!("{}", pretty_error(&context.func, Some(isa), err))
                })?;

            if options.print {
                println!("{}", context.func.display(isa));
            }

            if options.disasm {
                print_all(
                    isa,
                    &mem,
                    code_info.code_size,
                    code_info.jumptables_size + code_info.rodata_size,
                    &relocs,
                    &traps,
                    &stack_maps,
                )?;
            }
        }
    }

    if let Some(out_path) = options.output.clone() {
        write_output(test_file, out_path, options)?;
    }

    if options.report_times {
        print!("{}", timing::take_current());
    }

    Ok(())
}

fn write_output(test_file: TestFile, path: PathBuf, options: &Options) -> Result<()> {
    let filename = format!(
        "{}.o",
        path.file_name()
            .expect("Expected file name")
            .to_str()
            .unwrap()
    );

    let isa = {
        let parsed = parse_sets_and_triple(&options.settings, &options.target)?;
        match parsed {
            OwnedFlagsOrIsa::Isa(isa) => isa,
            OwnedFlagsOrIsa::Flags(flags) => host_isa_builder()
                .map_err(|s| anyhow::anyhow!("{}", s))?
                .finish(flags),
        }
    };

    let builder =
        ObjectBuilder::new(isa, filename, cranelift_module::default_libcall_names()).unwrap();
    let mut module = ObjectModule::new(builder);

    for (func, _) in test_file.functions {
        let mut context = Context::new();
        context.func = func.clone();

        let mut dctx = DataContext::new();
        let data_id = module.declare_anonymous_data(true, true)?;
        let _gv = module.declare_data_in_data(data_id, &mut dctx);

        let name = format!("{}", func.name);
        let func_id = module.declare_function(&name, Linkage::Local, &func.signature)?;

        let mut trap_sink = NullTrapSink {};
        let mut stack_map_sink = NullStackMapSink {};
        module
            .define_function(func_id, &mut context, &mut trap_sink, &mut stack_map_sink)
            .unwrap();
    }

    let product = module.finish();
    let obj = product.object.write().unwrap();
    if let Err(err) = std::fs::write(&path, obj) {
        anyhow::bail!("error writing object file: {}", err);
    }

    Ok(())
}
