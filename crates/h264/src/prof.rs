//! Stage timing for decoder work (the `profile` feature; a no-op without it).
//!
//! Each worker thread accumulates the wall time of its macroblock stages; [`flush_thread`] adds
//! them to the process total after a picture, and [`take`] reads and clears the total. `h264dec`
//! prints the breakdown. Without the feature every call compiles to the closure itself.

/// Stage names, in the order of [`take`]'s array.
pub const STAGES: [&str; 7] = ["macroblocks", "intra recon", "inter pred", "inter residual", "deblock", "publish", "residual parse"];
pub const MB: usize = 0;
pub const INTRA: usize = 1;
pub const INTER_PRED: usize = 2;
pub const INTER_RESID: usize = 3;
pub const DEBLOCK: usize = 4;
pub const PUBLISH: usize = 5;
/// Parsing the residual blocks of a macroblock (inside [`MB`]).
pub const RESID: usize = 6;

#[cfg(feature = "profile")]
mod on {
    use std::cell::Cell;
    use std::sync::{Mutex, PoisonError};

    thread_local! {
        static ACC: Cell<[u64; 7]> = const { Cell::new([0; 7]) };
    }
    static TOTAL: Mutex<[u64; 7]> = Mutex::new([0; 7]);

    #[inline]
    pub fn time<T>(stage: usize, f: impl FnOnce() -> T) -> T {
        let t0 = std::time::Instant::now();
        let r = f();
        let ns = t0.elapsed().as_nanos() as u64;
        ACC.with(|a| {
            let mut v = a.get();
            if let Some(s) = v.get_mut(stage) {
                *s = s.saturating_add(ns);
            }
            a.set(v);
        });
        r
    }

    pub fn flush_thread() {
        let v = ACC.with(|a| a.replace([0; 7]));
        let mut t = TOTAL.lock().unwrap_or_else(PoisonError::into_inner);
        for (t, v) in t.iter_mut().zip(v) {
            *t = t.saturating_add(v);
        }
    }

    pub fn take() -> [u64; 7] {
        std::mem::take(&mut *TOTAL.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[cfg(feature = "profile")]
pub use on::{flush_thread, take, time};

/// Run `f` (timing is off without the `profile` feature).
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn time<T>(_stage: usize, f: impl FnOnce() -> T) -> T {
    f()
}

/// Nothing to add without the `profile` feature.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn flush_thread() {}

/// Always zero without the `profile` feature.
#[cfg(not(feature = "profile"))]
pub fn take() -> [u64; 7] {
    [0; 7]
}
