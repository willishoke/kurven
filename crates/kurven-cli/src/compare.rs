//! Two drawings of one scene, compared per layer: the third layer of the
//! gate against Swift (HANDOFF-web-port.md, "The gate").
//!
//! The port does not match Swift bit for bit, since Swift calls Darwin's
//! libm, so a drawing is held to Swift's by tolerances in page units:
//!
//! - the same layers in the same order, each of the same colour and stroke
//!   width (in points, which is what a layer's width is);
//! - per layer, the symmetric Hausdorff distance between the two sets of
//!   strokes, as curves and not only their vertices, under a distance
//!   tolerance;
//! - per layer, the total stroke length equal to a relative tolerance;
//! - the stroke counts, reported: a difference is a finding to explain (a
//!   bisection that fell the other way is acceptable; a missing run is not)
//!   and does not by itself fail the comparison.
//!
//! The Hausdorff distance is computed to within `eps`, and the verdict is
//! taken on the upper end of the bound, so a pass is never a rounding of a
//! fail. Directed from A to B it is the largest distance from a point of A to
//! the nearest point of B. Along a segment of A that distance f is
//! 1-Lipschitz in arc length, so between ends p and q it is at most
//! (f(p) + f(q) + |pq|) / 2; and the distance to any one segment of B is
//! convex along it, so f is at most the larger of that segment's distances
//! at p and q. Segments whose bound cannot beat the largest distance found by
//! more than `eps` are dropped; the rest are halved. Each stroke of B is
//! found through a uniform grid.

use crate::svg::{Drawing, Point};

/// The tolerances a comparison is held to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Hausdorff distance, in page units.
    pub distance: f64,
    /// Total stroke length, relative to the longer of the two.
    pub length: f64,
}

impl Default for Tolerance {
    /// The defaults until a phase sets its own per drawing (crates/FINDINGS.md):
    /// a thousandth of a page unit, some fifteen times the SVG's own rounding
    /// to four decimals, and a part in a thousand of the length.
    fn default() -> Self {
        Tolerance { distance: 1e-3, length: 1e-3 }
    }
}

/// A distance known to lie in [lo, hi].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    pub lo: f64,
    pub hi: f64,
}

impl Bound {
    pub const ZERO: Bound = Bound { lo: 0.0, hi: 0.0 };
    pub const INFINITE: Bound = Bound { lo: f64::INFINITY, hi: f64::INFINITY };

    fn max(self, other: Bound) -> Bound {
        Bound { lo: self.lo.max(other.lo), hi: self.hi.max(other.hi) }
    }
}

fn distance(a: Point, b: Point) -> f64 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

/// The distance from p to the segment ab: exact at the ends, so a point that
/// is a vertex of the other drawing is at distance 0, not a rounding of it.
pub fn segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    if l2 == 0.0 {
        return distance(p, a);
    }
    let t = (ap[0] * ab[0] + ap[1] * ab[1]) / l2;
    if t <= 0.0 {
        distance(p, a)
    } else if t >= 1.0 {
        distance(p, b)
    } else {
        let cross = ab[0] * ap[1] - ab[1] * ap[0];
        (cross.abs() / l2.sqrt()).min(distance(p, a)).min(distance(p, b))
    }
}

/// The total length of a set of strokes.
pub fn length(paths: &[Vec<Point>]) -> f64 {
    paths.iter().flat_map(|p| p.windows(2)).map(|w| distance(w[0], w[1])).sum()
}

/// A set of strokes as segments, with a uniform grid to find the nearest.
pub struct Segments {
    segments: Vec<(Point, Point)>,
    origin: Point,
    cell: f64,
    nx: i64,
    ny: i64,
    cells: Vec<Vec<u32>>,
}

impl Segments {
    /// A stroke of one point is a segment of length zero.
    pub fn new(paths: &[Vec<Point>]) -> Segments {
        let mut segments = Vec::new();
        for path in paths {
            if path.len() == 1 {
                segments.push((path[0], path[0]));
            }
            segments.extend(path.windows(2).map(|w| (w[0], w[1])));
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for &(a, b) in &segments {
            for p in [a, b] {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        if segments.is_empty() {
            return Segments { segments, origin: [0.0; 2], cell: 1.0, nx: 0, ny: 0, cells: Vec::new() };
        }
        // About one segment a cell, no more than 2048 cells a side.
        let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
        let n = segments.len() as f64;
        let mut cell = if w * h > 0.0 { (w * h / n).sqrt() } else { w.max(h) / n };
        cell = cell.max(w.max(h) / 2048.0);
        if cell.is_nan() || cell <= 0.0 {
            cell = 1.0;
        }
        let nx = ((w / cell).floor() as i64 + 1).max(1);
        let ny = ((h / cell).floor() as i64 + 1).max(1);
        let mut index = Segments { segments, origin: lo, cell, nx, ny, cells: vec![Vec::new(); (nx * ny) as usize] };
        for k in 0..index.segments.len() {
            index.insert(k);
        }
        index
    }

    fn cell_of(&self, p: Point) -> (i64, i64) {
        (((p[0] - self.origin[0]) / self.cell).floor() as i64, ((p[1] - self.origin[1]) / self.cell).floor() as i64)
    }

    /// Into every cell the segment passes through: cut into pieces no longer
    /// than a cell, each piece's box meets at most four. A segment is
    /// inserted whole before the next, so a cell already holding it has it
    /// last.
    fn insert(&mut self, k: usize) {
        let (a, b) = self.segments[k];
        let pieces = (distance(a, b) / self.cell).ceil().max(1.0) as usize;
        for i in 0..pieces {
            let (s, t) = (i as f64 / pieces as f64, (i + 1) as f64 / pieces as f64);
            let p = [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])];
            let q = if i + 1 == pieces { b } else { [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])] };
            let (c0, c1) = (self.cell_of(p), self.cell_of(q));
            for x in c0.0.min(c1.0)..=c0.0.max(c1.0) {
                for y in c0.1.min(c1.1)..=c0.1.max(c1.1) {
                    if (0..self.nx).contains(&x) && (0..self.ny).contains(&y) {
                        let cell = &mut self.cells[(y * self.nx + x) as usize];
                        if cell.last() != Some(&(k as u32)) {
                            cell.push(k as u32);
                        }
                    }
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The distance from p to the nearest segment, and that segment; none
    /// when there are no segments.
    ///
    /// The rings of cells round p's are walked outward, each clamped to the
    /// grid, from the first that meets the grid, so a point far off it costs
    /// no more than one beside it. The walk starts from segment 0, so the
    /// answer always names a segment, even for a point no distance can be
    /// measured from.
    pub fn nearest(&self, p: Point) -> Option<(f64, usize)> {
        let (a, b) = *self.segments.first()?;
        let mut best = (segment_distance(p, a, b), 0);
        // Past this many cells a point is beyond any grid; clamping it nearer
        // only lowers the bound below, which stays a bound.
        const FAR: i64 = 1 << 40;
        let (cx, cy) = self.cell_of(p);
        let (cx, cy) = (cx.clamp(-FAR, FAR), cy.clamp(-FAR, FAR));
        let gap = |c: i64, n: i64| {
            if c < 0 {
                -c
            } else if c >= n {
                c - (n - 1)
            } else {
                0
            }
        };
        let first = gap(cx, self.nx).max(gap(cy, self.ny));
        let last = [cx, self.nx - 1 - cx, cy, self.ny - 1 - cy].into_iter().map(i64::abs).max().unwrap_or(0);
        let visit = |x: i64, y: i64, best: &mut (f64, usize)| {
            for &k in &self.cells[(y * self.nx + x) as usize] {
                let (a, b) = self.segments[k as usize];
                let d = segment_distance(p, a, b);
                if d < best.0 {
                    *best = (d, k as usize);
                }
            }
        };
        for r in first..=last {
            if r == 0 {
                visit(cx, cy, &mut best);
            } else {
                let (x0, x1) = ((cx - r).max(0), (cx + r).min(self.nx - 1));
                for y in [cy - r, cy + r] {
                    if (0..self.ny).contains(&y) {
                        for x in x0..=x1 {
                            visit(x, y, &mut best);
                        }
                    }
                }
                let (y0, y1) = ((cy - r + 1).max(0), (cy + r - 1).min(self.ny - 1));
                for x in [cx - r, cx + r] {
                    if (0..self.nx).contains(&x) {
                        for y in y0..=y1 {
                            visit(x, y, &mut best);
                        }
                    }
                }
            }
            // Every cell not yet visited is at least r cells from p's.
            if best.0 <= r as f64 * self.cell {
                break;
            }
        }
        Some(best)
    }

    fn distance_to(&self, k: usize, p: Point) -> f64 {
        let (a, b) = self.segments[k];
        segment_distance(p, a, b)
    }
}

/// The directed Hausdorff distance from the strokes `a` to the segments `b`,
/// to within `eps`.
pub fn directed(a: &[Vec<Point>], b: &Segments, eps: f64) -> Bound {
    if a.iter().all(Vec::is_empty) {
        return Bound::ZERO;
    }
    if b.is_empty() {
        return Bound::INFINITE;
    }
    let mut lo = 0.0f64;
    let mut hi = 0.0f64;
    let mut pending: Vec<(Point, f64, usize, Point, f64, usize)> = Vec::new();
    for path in a {
        let near: Vec<(f64, usize)> = path.iter().map(|&p| b.nearest(p).expect("b holds a segment")).collect();
        for &(f, _) in &near {
            lo = lo.max(f);
        }
        for i in 1..path.len() {
            pending.push((path[i - 1], near[i - 1].0, near[i - 1].1, path[i], near[i].0, near[i].1));
        }
    }
    while let Some((p, fp, ip, q, fq, iq)) = pending.pop() {
        let bound =
            ((fp + fq + distance(p, q)) / 2.0).min(fp.max(b.distance_to(ip, q))).min(b.distance_to(iq, p).max(fq));
        if bound <= lo + eps {
            hi = hi.max(bound);
            continue;
        }
        let m = [(p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0];
        let (fm, im) = b.nearest(m).expect("b holds a segment");
        lo = lo.max(fm);
        pending.push((p, fp, ip, m, fm, im));
        pending.push((m, fm, im, q, fq, iq));
    }
    Bound { lo, hi: hi.max(lo) }
}

/// The symmetric Hausdorff distance between two sets of strokes, to within
/// `eps`. Two identical sets are at distance 0 exactly.
pub fn hausdorff(a: &[Vec<Point>], b: &[Vec<Point>], eps: f64) -> Bound {
    if a == b {
        return Bound::ZERO;
    }
    directed(a, &Segments::new(b), eps).max(directed(b, &Segments::new(a), eps))
}

/// One layer of each drawing, side by side.
#[derive(Debug, Clone)]
pub struct LayerReport {
    pub color: (String, String),
    pub width: (f64, f64),
    pub paths: (usize, usize),
    pub length: (f64, f64),
    pub hausdorff: Bound,
}

impl LayerReport {
    pub fn relative_length(&self) -> f64 {
        let (a, b) = self.length;
        if a == b { 0.0 } else { (a - b).abs() / a.max(b) }
    }
}

#[derive(Debug, Clone)]
pub struct Report {
    pub tolerance: Tolerance,
    pub layers: Vec<LayerReport>,
    /// Why the drawings differ; empty when they are the same within the
    /// tolerance.
    pub failures: Vec<String>,
    /// What differs without failing: stroke counts.
    pub notes: Vec<String>,
}

impl Report {
    pub fn same(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Stroke widths are written to six decimals of a page unit; in points that
/// is this much of a point per page unit of the viewBox's width.
fn width_rounding(d: &Drawing) -> f64 {
    5e-7 * crate::svg::PAGE_POINTS / d.view_box[2]
}

pub fn compare(a: &Drawing, b: &Drawing, tolerance: Tolerance) -> Report {
    let eps = tolerance.distance / 64.0;
    let mut failures = Vec::new();
    let mut notes = Vec::new();
    if a.layers.len() != b.layers.len() {
        let list = |d: &Drawing| {
            d.layers.iter().map(|l| format!("{} {:.4}pt", l.color, d.width_points(l))).collect::<Vec<_>>().join(", ")
        };
        failures.push(format!("{} layers against {}: [{}] and [{}]", a.layers.len(), b.layers.len(), list(a), list(b)));
    }
    let mut layers = Vec::new();
    for (i, (la, lb)) in a.layers.iter().zip(&b.layers).enumerate() {
        let report = LayerReport {
            color: (la.color.clone(), lb.color.clone()),
            width: (a.width_points(la), b.width_points(lb)),
            paths: (la.paths.len(), lb.paths.len()),
            length: (length(&la.paths), length(&lb.paths)),
            hausdorff: hausdorff(&la.paths, &lb.paths, eps),
        };
        if report.color.0 != report.color.1 {
            failures.push(format!("layer {i}: stroke {} against {}", report.color.0, report.color.1));
        }
        let (wa, wb) = report.width;
        if (wa - wb).abs() > 1e-3 * wa.max(wb) + width_rounding(a) + width_rounding(b) {
            failures.push(format!("layer {i}: width {wa:.6}pt against {wb:.6}pt"));
        }
        if report.hausdorff.hi > tolerance.distance {
            failures.push(format!(
                "layer {i}: Hausdorff distance {} is over {}",
                if report.hausdorff.lo > tolerance.distance {
                    format!("{:.3e}", report.hausdorff.lo)
                } else {
                    format!("up to {:.3e}", report.hausdorff.hi)
                },
                tolerance.distance
            ));
        }
        if report.relative_length() > tolerance.length {
            failures.push(format!(
                "layer {i}: length {:.6} against {:.6}, {:.3e} apart, over {}",
                report.length.0,
                report.length.1,
                report.relative_length(),
                tolerance.length
            ));
        }
        if report.paths.0 != report.paths.1 {
            notes.push(format!(
                "layer {i}: {} strokes against {}, a finding to explain",
                report.paths.0, report.paths.1
            ));
        }
        layers.push(report);
    }
    Report { tolerance, layers, failures, notes }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "{:>3}  {:<9} {:>9}  {:>7} {:>7}  {:>12} {:>12} {:>10}  {:>12}",
            "#", "stroke", "width pt", "paths a", "paths b", "length a", "length b", "rel. len", "hausdorff"
        )?;
        for (i, l) in self.layers.iter().enumerate() {
            let h = l.hausdorff;
            let hausdorff = if h.hi == 0.0 {
                "0".to_string()
            } else if h.lo.is_infinite() {
                "inf".to_string()
            } else {
                // The search's upper bound, which the verdict is taken on.
                format!("<= {:.3e}", h.hi)
            };
            let relative = match l.relative_length() {
                0.0 => "0".to_string(),
                r => format!("{r:.3e}"),
            };
            writeln!(
                f,
                "{i:>3}  {:<9} {:>9.4}  {:>7} {:>7}  {:>12.6} {:>12.6} {:>10}  {:>12}",
                l.color.0, l.width.0, l.paths.0, l.paths.1, l.length.0, l.length.1, relative, hausdorff
            )?;
        }
        for note in &self.notes {
            writeln!(f, "note: {note}")?;
        }
        if self.same() {
            write!(
                f,
                "same within tolerance: distance {} page units, length {} relative",
                self.tolerance.distance, self.tolerance.length
            )
        } else {
            write!(f, "DIFFERENT:")?;
            for failure in &self.failures {
                write!(f, "\n  {failure}")?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn within(b: Bound, want: f64) {
        assert!(b.lo <= want + 1e-12 && want <= b.hi + 1e-12, "{b:?} does not hold {want}");
        assert!(b.hi - b.lo <= EPS + 1e-12, "{b:?} is wider than eps");
    }

    #[test]
    fn segment_distances() {
        let (a, b) = ([0.0, 0.0], [2.0, 0.0]);
        assert_eq!(segment_distance([1.0, 3.0], a, b), 3.0);
        assert_eq!(segment_distance([-3.0, 4.0], a, b), 5.0);
        assert_eq!(segment_distance([5.0, -4.0], a, b), 5.0);
        assert_eq!(segment_distance([2.0, 0.0], a, b), 0.0);
        assert_eq!(segment_distance([1.0, 1.0], a, a), 2f64.sqrt());
    }

    #[test]
    fn identical_sets_are_at_zero() {
        let s = vec![vec![[0.0, 0.0], [1.0, 0.5], [2.0, 0.25]], vec![[0.3, 0.3]]];
        assert_eq!(hausdorff(&s, &s, EPS), Bound::ZERO);
        // And by the search, not only by the shortcut: within rounding.
        let b = directed(&s, &Segments::new(&s), EPS);
        assert!(b.hi < 1e-15, "{b:?}");
    }

    /// Two parallel segments a distance d apart: d each way.
    #[test]
    fn parallel_segments() {
        let a = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        let b = vec![vec![[0.0, 0.25], [1.0, 0.25]]];
        within(hausdorff(&a, &b, EPS), 0.25);
    }

    /// A segment and a longer one along the same line: the longer one's far
    /// end is 2 from the shorter; the other way the distance is 0.
    #[test]
    fn asymmetric() {
        let a = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        let b = vec![vec![[-2.0, 0.0], [1.0, 0.0]]];
        within(directed(&a, &Segments::new(&b), EPS), 0.0);
        within(directed(&b, &Segments::new(&a), EPS), 2.0);
        within(hausdorff(&a, &b, EPS), 2.0);
    }

    /// The farthest point of a curve from the other set can lie inside a
    /// segment, at no vertex of either drawing.
    #[test]
    fn the_far_point_is_inside_a_segment() {
        // From (x, 0) the nearer of the points (±1, 1) is √((1 − |x|)² + 1)
        // away, largest at x = 0: √2, where the ends are at 1.
        let a = vec![vec![[-1.0, 0.0], [1.0, 0.0]]];
        let b = vec![vec![[-1.0, 1.0]], vec![[1.0, 1.0]]];
        within(directed(&a, &Segments::new(&b), EPS), 2f64.sqrt());
        // The same, off the midpoint: along (x, 1), x from 0 to 3, the nearer
        // of (0, 0) and (4, 0) changes at x = 2, two thirds of the way,
        // where the distance is √5.
        let a = vec![vec![[0.0, 1.0], [3.0, 1.0]]];
        let b = vec![vec![[0.0, 0.0]], vec![[4.0, 0.0]]];
        within(directed(&a, &Segments::new(&b), EPS), 5f64.sqrt());
        // A curve whose vertices all lie on b and which bows away between
        // them, through (0.5, 0.3).
        let a = vec![vec![[0.0, 0.0], [0.5, 0.3], [1.0, 0.0]]];
        let b = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        within(hausdorff(&a, &b, EPS), 0.3);
    }

    /// A unit square's outline against the same outline with one side
    /// missing: the missing side's midpoint is 0.5 from the rest.
    #[test]
    fn a_missing_stroke() {
        let full = vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]];
        let open = vec![vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]];
        within(directed(&open, &Segments::new(&full), EPS), 0.0);
        within(hausdorff(&full, &open, EPS), 0.5);
    }

    #[test]
    fn empty_sets() {
        let s = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        assert_eq!(hausdorff(&[], &[], EPS), Bound::ZERO);
        assert_eq!(hausdorff(&s, &[], EPS), Bound::INFINITE);
        assert_eq!(hausdorff(&[], &s, EPS), Bound::INFINITE);
    }

    /// Against a brute-force reference on random polylines: the search's
    /// bound holds the distance from a dense sampling, which is a lower
    /// bound, and the sampling plus half its step, an upper one.
    #[test]
    fn agrees_with_dense_sampling() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..20 {
            let mut set = || -> Vec<Vec<Point>> {
                (0..3).map(|_| (0..6).map(|_| [next() * 4.0, next() * 3.0]).collect()).collect()
            };
            let (a, b) = (set(), set());
            let bound = directed(&a, &Segments::new(&b), EPS);
            let step = 1e-4;
            let mut sampled = 0.0f64;
            for w in a.iter().flat_map(|p| p.windows(2)) {
                let n = (distance(w[0], w[1]) / step).ceil() as usize;
                for k in 0..=n {
                    let t = k as f64 / n as f64;
                    let p = [w[0][0] + t * (w[1][0] - w[0][0]), w[0][1] + t * (w[1][1] - w[0][1])];
                    let d = b
                        .iter()
                        .flat_map(|q| q.windows(2))
                        .map(|v| segment_distance(p, v[0], v[1]))
                        .fold(f64::INFINITY, f64::min);
                    sampled = sampled.max(d);
                }
            }
            assert!(bound.hi >= sampled - 1e-12, "{bound:?} below the sampled {sampled}");
            assert!(bound.lo <= sampled + step / 2.0 + 1e-12, "{bound:?} above the sampled {sampled} + step/2");
        }
    }

    /// A point far off a small, dense set costs no more than one beside it:
    /// the rings start where the grid begins. Walking every ring out to it,
    /// as the search once did, would visit some 10¹⁶ cells here.
    #[test]
    fn a_far_point_is_found_quickly() {
        let dense: Vec<Vec<Point>> =
            (0..100).map(|i| (0..101).map(|j| [f64::from(j) * 0.01, f64::from(i) * 0.01]).collect()).collect();
        let index = Segments::new(&dense);
        let started = std::time::Instant::now();
        let (d, k) = index.nearest([1e6, 0.5]).unwrap();
        assert_eq!(d, 1e6 - 1.0);
        assert!(k < 100 * 100);
        let (d, _) = index.nearest([-3e5, -4e5]).unwrap();
        assert!((d - 5e5).abs() < 1e-6, "{d}");
        within(directed(&[vec![[1e6, 0.5], [1e6, 0.6]]], &index, EPS), 1e6 - 1.0);
        assert!(started.elapsed() < std::time::Duration::from_secs(1), "took {:?}", started.elapsed());
    }

    /// No point, however unmeasurable, makes the search name no segment.
    #[test]
    fn nearest_always_names_a_segment() {
        let index = Segments::new(&[vec![[0.0, 0.0], [1.0, 0.0]], vec![[0.0, 1.0], [1.0, 1.0]]]);
        for p in [[f64::NAN, 0.0], [0.5, f64::INFINITY], [f64::NEG_INFINITY, f64::NAN]] {
            let (_, k) = index.nearest(p).unwrap();
            assert!(k < 2);
        }
        assert_eq!(Segments::new(&[]).nearest([0.0, 0.0]), None);
    }

    #[test]
    fn lengths() {
        assert_eq!(length(&[vec![[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]], vec![[9.0, 9.0]]]), 6.0);
    }
}
