use std::env;

fn main() {
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if !["windows", "linux", "android", "macos"].contains(&os.as_str()) {
        cc::Build::new().file("src/c/icache.c").compile("icache");
    }
}
