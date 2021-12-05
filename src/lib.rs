pub mod hyksort;

/// The raw C language MPI API
///
/// Documented in the [Message Passing Interface specification][spec]
///
/// [spec]: http://www.mpi-forum.org/docs/docs.html
#[allow(missing_docs, dead_code, non_snake_case, non_camel_case_types)]
#[macro_use]
pub mod ffi {
    pub use mpi_sys::*;
}
use crate::ffi::MPI_Aint;
