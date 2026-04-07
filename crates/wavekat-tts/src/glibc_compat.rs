// glibc 2.38 introduced C23 variants of the strto* family (ISO C23 §7.22.1).
// ORT prebuilt binaries compiled on newer toolchains emit references to these
// symbols, but Ubuntu 22.04 (glibc 2.35) — used by Google Colab and many CI
// hosts — does not provide them.  Define thin wrappers so the linker is happy.
use std::ffi::c_char;
use std::os::raw::{c_int, c_long, c_longlong, c_ulonglong};

extern "C" {
    fn strtol(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoll(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_longlong;
    fn strtoull(nptr: *const c_char, endptr: *mut *mut c_char, base: c_int) -> c_ulonglong;
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtol(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_long {
    strtol(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoll(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_longlong {
    strtoll(nptr, endptr, base)
}

#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoull(
    nptr: *const c_char,
    endptr: *mut *mut c_char,
    base: c_int,
) -> c_ulonglong {
    strtoull(nptr, endptr, base)
}
