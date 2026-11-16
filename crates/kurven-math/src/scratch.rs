//! Scratch space for the evaluations made by the million: one buffer per
//! thread, reused, so an evaluation neither allocates nor clears memory.
//!
//! Swift takes this space on the stack (`withUnsafeTemporaryAllocation`), of
//! the size each call needs and uninitialized. Rust has no stack array of a
//! size known only at run time, and zeroing a fixed one costs more than a
//! short tape's work, so the buffer is a thread's own vector, grown when a
//! call needs more and never cleared. Every caller writes each element before
//! it reads it.

use std::cell::RefCell;

use crate::complex::Complex;

thread_local! {
    static SCRATCH: RefCell<Vec<Complex>> = const { RefCell::new(Vec::new()) };
}

/// `f` on `n` complex values of scratch whose contents are unspecified. A
/// call made from inside `f` on the same thread, which nothing does today,
/// gets a buffer of its own.
pub(crate) fn with_complex<R>(n: usize, f: impl FnOnce(&mut [Complex]) -> R) -> R {
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buffer) => {
            if buffer.len() < n {
                buffer.resize(n, Complex::ZERO);
            }
            f(&mut buffer[..n])
        }
        Err(_) => f(&mut vec![Complex::ZERO; n]),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_call_gets_its_own_buffer() {
        let outer = with_complex(3, |a| {
            a.fill(Complex::ONE);
            let inner = with_complex(5, |b| {
                b.fill(Complex::I);
                b.len()
            });
            (a.iter().all(|&x| x == Complex::ONE), inner)
        });
        assert_eq!(outer, (true, 5));
    }
}
