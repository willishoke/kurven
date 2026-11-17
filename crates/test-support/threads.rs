//! What the suites of both kurven-landscape and kurven-surfaces use to
//! spread their work over threads, and to keep it off what they time.
//! Each crate's `tests/common/mod.rs` includes this file by `#[path]`, so
//! both run the same code; it is no crate of its own.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// The threads the suites spread their oracles over: the machine's, or
/// `KURVEN_TEST_THREADS` where it is set, to time the suites as a smaller
/// runner would run them (with `RUST_TEST_THREADS` for libtest's own).
pub fn threads() -> usize {
    std::env::var("KURVEN_TEST_THREADS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
}

/// `items.map(body)` over the machine's threads, in order. Each thread
/// takes the next few items as it comes free, so a run of costly items
/// (the cameras near edge-on, a dense stretch of samples) is shared out
/// rather than left to one thread; each item is computed as it would be in
/// a plain loop, and the results come back in the items' order.
pub fn par_map<T: Sync, R: Send>(items: &[T], body: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads().min(items.len()).max(1);
    let grain = (items.len() / (threads * 64)).max(1);
    let next = AtomicUsize::new(0);
    let mut pieces: Vec<(usize, Vec<R>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let start = next.fetch_add(grain, Ordering::Relaxed);
                        if start >= items.len() {
                            break out;
                        }
                        let end = (start + grain).min(items.len());
                        out.push((start, items[start..end].iter().map(&body).collect::<Vec<R>>()));
                    }
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    pieces.sort_by_key(|p| p.0);
    pieces.into_iter().flat_map(|p| p.1).collect()
}

/// One test binary's tests, as libtest runs them side by side: a test that
/// asserts a wall-clock time runs its timed stretch `alone`, and the file's
/// other tests, which saturate the cores with `par_map`, hold `sharing`
/// while they run. A test takes one guard at a time, and never `sharing`
/// twice: a reader waiting behind a writer would wait for itself.
static QUIET: RwLock<()> = RwLock::new(());

/// The guard a test holds while it runs beside the others.
pub fn sharing() -> RwLockReadGuard<'static, ()> {
    QUIET.read().unwrap_or_else(|e| e.into_inner())
}

/// The guard a timed stretch holds: no other test of the file runs meanwhile.
pub fn alone() -> RwLockWriteGuard<'static, ()> {
    QUIET.write().unwrap_or_else(|e| e.into_inner())
}
