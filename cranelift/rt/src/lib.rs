extern "C" {
    /// The probestack function for cranelift
    ///
    /// Do *not* try to call this function, it will go wrong.
    ///
    /// This is here so that it's easier to get its address and pass it onto cranelift.
    #[cfg(all(not(target_os = "windows"), target_arch = "x86_64"))]
    pub fn __cranelift_probestack(size: u64);
}
