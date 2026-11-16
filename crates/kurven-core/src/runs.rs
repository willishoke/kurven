//! The runs of a polyline that are seen, one segment's spans at a time: the
//! loop Solid.swift's `visibleCurves` (both forms) and Composite.swift's
//! `cut` each spell out, here once.
//!
//! A path is kept as one run wherever the spans of consecutive segments
//! meet at the vertex between them, so a curve breaks only where something
//! hides it; a span shorter than 1e-12 of its segment is dropped, and a run
//! of fewer than two vertices is not kept.

/// The runs found so far, and the one being drawn.
pub struct Runs<P> {
    out: Vec<Vec<P>>,
    run: Vec<P>,
    /// Whether the run reaches the end of the last segment.
    open_at_end: bool,
}

impl<P: Copy> Default for Runs<P> {
    fn default() -> Self {
        Runs { out: Vec::new(), run: Vec::new(), open_at_end: false }
    }
}

impl<P: Copy> Runs<P> {
    pub fn new() -> Runs<P> {
        Runs::default()
    }

    /// The next segment of the path, as the fractions of it seen, in order,
    /// `at` placing a fraction. A segment of which nothing is seen (no
    /// spans) ends the run.
    pub fn segment(&mut self, spans: &[(f64, f64)], at: impl Fn(f64) -> P) {
        // Whether the run so far reaches this segment's start.
        let continuing = self.open_at_end;
        self.open_at_end = false;
        for &(lo, hi) in spans {
            if !(hi - lo > 1e-12) {
                continue;
            }
            let (start, end) = (at(lo), at(hi));
            if continuing && lo == 0.0 && !self.run.is_empty() {
                self.run.push(end);
            } else {
                self.flush();
                self.run = vec![start, end];
            }
            self.open_at_end = hi == 1.0;
        }
        if !self.open_at_end {
            self.flush();
        }
    }

    /// The end of a path: its last run kept, and the next path starts afresh.
    pub fn end_path(&mut self) {
        self.flush();
        self.open_at_end = false;
    }

    /// Every run kept.
    pub fn finish(mut self) -> Vec<Vec<P>> {
        self.flush();
        self.out
    }

    fn flush(&mut self) {
        let run = std::mem::take(&mut self.run);
        if run.len() >= 2 {
            self.out.push(run);
        }
    }
}
