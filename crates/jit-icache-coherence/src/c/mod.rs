//! The C implementation is to just compile a C file and use `__builtin___clear_cache`
use std::ffi::{c_char, c_void};
use std::io::Result;

#[link(name = "icache")]
extern "C" {
    fn icache__clear_cache(begin: *mut c_char, end: *const c_char);
}

pub fn pipeline_flush_mt() -> Result<()> {
    Ok(())
}

pub unsafe fn clear_cache(ptr: *const c_void, len: usize) -> Result<()> {
    let begin = ptr as *mut c_char;
    let end = begin.add(len);
    unsafe { icache__clear_cache(begin, end) }
    Ok(())
}
