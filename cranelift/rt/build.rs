use std::env;

fn main() {
    let mut build = cc::Build::new();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if arch == "x86_64" && os != "windows" {
        println!("cargo:rerun-if-changed=src/x86_64.s");
        build.file("src/x86_64.s");
        build.compile("cranelift-rt");
    }
}
