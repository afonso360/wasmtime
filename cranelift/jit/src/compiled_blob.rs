use cranelift_codegen::binemit::Reloc;
use cranelift_module::ModuleExtName;
use cranelift_module::ModuleReloc;
use std::convert::TryFrom;

/// Reads a 32bit instruction at `iptr`, and writes it again after
/// being altered by `modifier`
fn modify_inst32(iptr: *mut u32, modifier: impl FnOnce(u32) -> u32) {
    let inst = unsafe { iptr.read_unaligned() };
    let new_inst = modifier(inst);
    unsafe {
        iptr.write_unaligned(new_inst);
    }
}

#[derive(Clone)]
pub(crate) struct CompiledBlob {
    pub(crate) ptr: *mut u8,
    pub(crate) size: usize,
    pub(crate) relocs: Vec<ModuleReloc>,
}

impl CompiledBlob {
    pub(crate) fn perform_relocations(
        &self,
        get_address: impl Fn(&ModuleExtName) -> *const u8,
        get_got_entry: impl Fn(&ModuleExtName) -> *const u8,
        get_plt_entry: impl Fn(&ModuleExtName) -> *const u8,
    ) {
        use std::ptr::write_unaligned;

        for &ModuleReloc {
            kind,
            offset,
            ref name,
            addend,
        } in &self.relocs
        {
            debug_assert!((offset as usize) < self.size);
            let at = unsafe { self.ptr.offset(isize::try_from(offset).unwrap()) };
            match kind {
                Reloc::Abs4 => {
                    let base = get_address(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut u32, u32::try_from(what as usize).unwrap())
                    };
                }
                Reloc::Abs8 => {
                    let base = get_address(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut u64, u64::try_from(what as usize).unwrap())
                    };
                }
                Reloc::X86PCRel4 | Reloc::X86CallPCRel4 => {
                    let base = get_address(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    let pcrel = i32::try_from((what as isize) - (at as isize)).unwrap();
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut i32, pcrel)
                    };
                }
                Reloc::X86GOTPCRel4 => {
                    let base = get_got_entry(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    let pcrel = i32::try_from((what as isize) - (at as isize)).unwrap();
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut i32, pcrel)
                    };
                }
                Reloc::X86CallPLTRel4 => {
                    let base = get_plt_entry(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    let pcrel = i32::try_from((what as isize) - (at as isize)).unwrap();
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut i32, pcrel)
                    };
                }
                Reloc::S390xPCRel32Dbl | Reloc::S390xPLTRel32Dbl => {
                    let base = get_address(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };
                    let pcrel = i32::try_from(((what as isize) - (at as isize)) >> 1).unwrap();
                    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_ptr_alignment))]
                    unsafe {
                        write_unaligned(at as *mut i32, pcrel)
                    };
                }
                Reloc::Arm64Call => {
                    let base = get_address(name);
                    // The instruction is 32 bits long.
                    let iptr = at as *mut u32;
                    // The offset encoded in the `bl` instruction is the
                    // number of bytes divided by 4.
                    let diff = ((base as isize) - (at as isize)) >> 2;
                    // Sign propagating right shift disposes of the
                    // included bits, so the result is expected to be
                    // either all sign bits or 0, depending on if the original
                    // value was negative or positive.
                    assert!((diff >> 26 == -1) || (diff >> 26 == 0));
                    // The lower 26 bits of the `bl` instruction form the
                    // immediate offset argument.
                    let chop = 32 - 26;
                    let imm26 = (diff as u32) << chop >> chop;
                    modify_inst32(iptr, |inst| inst | imm26);
                }
                Reloc::Aarch64AdrGotPage21 => {
                    // Updates the immediate bits of an ADR instruction
                    fn aarch64_update_adr(inst: u32, imm: isize) -> u32 {
                        let imm_lo: u32 = ((imm & 0x3) as u32) << 29;
                        let imm_hi: u32 = ((imm & 0x1FFFFC) as u32) << 3;
                        let mask: u32 = !((0x3 << 29) | (0x1FFFFC << 3));
                        (inst & mask) | imm_lo | imm_hi
                    }

                    let iptr = at as *mut u32;
                    let base = get_got_entry(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };

                    // Calculate the number of pages from the GOT entry to the current instruction page
                    let what_page = (what as isize) & !0xfff;
                    let at_page = (at as isize) & !0xfff;
                    let rel_page = (what_page - at_page) >> 12;

                    modify_inst32(iptr, |inst| aarch64_update_adr(inst, rel_page));
                }
                Reloc::Aarch64Ld64GotLo12Nc => {
                    // Set the LD/ST immediate field to bits [11:3] of X. No overflow check; check that X&7 = 0
                    let iptr = at as *mut u32;
                    let base = get_got_entry(name);
                    let what = unsafe { base.offset(isize::try_from(addend).unwrap()) };

                    // The entry must be 8 bytes aligned
                    assert_eq!(what.align_offset(8), 0);

                    modify_inst32(iptr, |inst| {
                        let lo12 = ((what as usize) & 0xFF8) as u32;
                        // Update the immediate bits of an LDR instruction with `lo12`
                        inst | (lo12 << 7)
                    });
                }
                r => unimplemented!("Unimplemented relocation {:?}", r),
            }
        }
    }
}
