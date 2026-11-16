//! Marching squares on a sample grid: the port of
//! `Sources/KurvenCore/Contour.swift`.
//!
//! Porting it is what turned a level set from something baked into a bundle
//! into something you can drag a slider on. contourpy remains the oracle --
//! `tests/oracles.py` dumps its answers and the suite compares against
//! them -- but it is not in the loop.
//!
//! No stitching by construction: this contours the whole grid as one region, so
//! there are no chunk boundaries to stitch and none of the thread-completion
//! nondeterminism that makes the oracle pass `chunk_count = 1` to get a
//! reproducible answer.

use std::collections::{HashMap, HashSet};

use crate::grid::Grid2D;
use crate::spaces::{DomainSpace, P2};

/// A crossing sits on a grid edge, and a grid edge is named by an integer.
///
/// Linking segments into paths by comparing *coordinates* means deciding
/// how close is close enough. Naming the edge instead makes the join exact:
/// two cells that share an edge compute the same crossing from the same two
/// samples, and they agree because they are the same arithmetic, not because
/// they are within a tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EdgeID {
    horizontal: bool,
    x: usize,
    y: usize,
}

/// A coarse min/max summary of the grid, so a level can skip the blocks it
/// cannot cross.
///
/// Contouring a level means asking every cell whether the level passes
/// through it, and for a smooth field the answer is almost always no. On a
/// 1200-square grid with ninety levels that is a hundred and thirty million
/// questions with a known answer. Blocks of 32 make it a few hundred
/// thousand.
///
/// Built once per grid and reused across levels, which is why it is a value
/// rather than something `lines` computes for itself.
#[derive(Debug, Clone)]
pub struct Index {
    blocks_across: usize,
    blocks_down: usize,
    lo: Vec<f32>,
    hi: Vec<f32>,
}

impl Index {
    const BLOCK: usize = 32;

    pub fn new(grid: &Grid2D<f32>) -> Index {
        let cells_x = grid.width.saturating_sub(1);
        let cells_y = grid.height.saturating_sub(1);
        let blocks_across = cells_x.div_ceil(Index::BLOCK);
        let blocks_down = cells_y.div_ceil(Index::BLOCK);
        let mut lo = vec![f32::INFINITY; blocks_across * blocks_down];
        let mut hi = vec![f32::NEG_INFINITY; lo.len()];
        for by in 0..blocks_down {
            let y0 = by * Index::BLOCK;
            let y1 = (y0 + Index::BLOCK + 1).min(grid.height);
            for bx in 0..blocks_across {
                let x0 = bx * Index::BLOCK;
                let x1 = (x0 + Index::BLOCK + 1).min(grid.width);
                let (mut l, mut h) = (f32::INFINITY, f32::NEG_INFINITY);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let v = grid.at(x, y);
                        if v < l {
                            l = v;
                        }
                        if v > h {
                            h = v;
                        }
                    }
                }
                lo[by * blocks_across + bx] = l;
                hi[by * blocks_across + bx] = h;
            }
        }
        Index { blocks_across, blocks_down, lo, hi }
    }

    /// A block can only contribute when some corner is above the level and
    /// some corner is not -- the same test the cell code performs, lifted to
    /// thirty-two cells at a time.
    fn crosses(&self, b: usize, level: f64) -> bool {
        f64::from(self.hi[b]) > level && f64::from(self.lo[b]) <= level
    }
}

/// Iso-lines of `grid` at `level`, in domain coordinates.
///
/// Closed loops come back with their first vertex repeated at the end, which
/// is what contourpy does and what a consumer drawing a polyline needs.
pub fn lines(grid: &Grid2D<f32>, level: f64, index: Option<&Index>) -> Vec<Vec<P2<DomainSpace>>> {
    lines_placed(grid, level, index, (false, false), |a, b| Some(interpolate(grid, level, a, b)))
}

/// `lines` with each crossing placed by `place`, given the sample indices
/// `[x, y]` of its edge's two ends, lower first. A `None` crossing is no
/// crossing: the segments that meet its edge are not drawn.
///
/// On an axis that `wraps`, the last row or column of samples is the
/// first again, a periodic axis sampled one cell past its end, and an edge
/// there is the edge at the start: a contour that crosses the seam is one
/// path, not two ending at it. The crossing is asked of the edge at the
/// start, so the two cells beside the seam agree on it.
pub fn lines_placed<S>(
    grid: &Grid2D<f32>,
    level: f64,
    index: Option<&Index>,
    wraps: (bool, bool),
    mut place: impl FnMut([usize; 2], [usize; 2]) -> Option<P2<S>>,
) -> Vec<Vec<P2<S>>> {
    if grid.width < 2 || grid.height < 2 {
        return Vec::new();
    }
    let owned;
    let index = match index {
        Some(i) => i,
        None => {
            owned = Index::new(grid);
            &owned
        }
    };

    // Every cell contributes zero, one or two segments, each joining two of
    // its four edges. Collected first, linked second.
    let mut from: HashMap<EdgeID, EdgeID> = HashMap::new();
    let mut points: HashMap<EdgeID, Option<P2<S>>> = HashMap::new();
    let mut has_predecessor: HashSet<EdgeID> = HashSet::new();
    // Segment starts in the order the scan meets them. Walking the map's
    // keys instead starts each path wherever the hasher puts it, and the
    // same grid then comes back as the same lines in a different order on
    // every run.
    let mut order: Vec<EdgeID> = Vec::new();

    let (w, h) = (grid.width, grid.height);
    for by in 0..index.blocks_down {
        for bx in 0..index.blocks_across {
            if !index.crosses(by * index.blocks_across + bx, level) {
                continue;
            }
            let ys = (by * Index::BLOCK)..((by + 1) * Index::BLOCK).min(h - 1);
            let xs = (bx * Index::BLOCK)..((bx + 1) * Index::BLOCK).min(w - 1);
            for y in ys {
                for x in xs.clone() {
                    let z0 = f64::from(grid.at(x, y)); // bottom-left
                    let z1 = f64::from(grid.at(x + 1, y)); // bottom-right
                    let z2 = f64::from(grid.at(x + 1, y + 1)); // top-right
                    let z3 = f64::from(grid.at(x, y + 1)); // top-left

                    let mut code = 0;
                    if z0 > level {
                        code |= 1;
                    }
                    if z1 > level {
                        code |= 2;
                    }
                    if z2 > level {
                        code |= 4;
                    }
                    if z3 > level {
                        code |= 8;
                    }
                    if code == 0 || code == 15 {
                        continue;
                    }

                    // Edge ids for this cell: bottom, right, top, left.
                    let right = if wraps.0 && x + 1 == w - 1 { 0 } else { x + 1 };
                    let top = if wraps.1 && y + 1 == h - 1 { 0 } else { y + 1 };
                    let e0 = EdgeID { horizontal: true, x, y };
                    let e1 = EdgeID { horizontal: false, x: right, y };
                    let e2 = EdgeID { horizontal: true, x, y: top };
                    let e3 = EdgeID { horizontal: false, x, y };

                    let mut link = |a: EdgeID, b: EdgeID| {
                        let mut crossing = |id: EdgeID| -> bool {
                            if let Some(p) = points.get(&id) {
                                return p.is_some();
                            }
                            let lo = [id.x, id.y];
                            let hi = if id.horizontal { [id.x + 1, id.y] } else { [id.x, id.y + 1] };
                            let p = place(lo, hi);
                            let found = p.is_some();
                            points.insert(id, p);
                            found
                        };
                        // Both asked, in this order, as Swift's guard asks
                        // the second only when the first is there.
                        if !crossing(a) || !crossing(b) {
                            return;
                        }
                        from.insert(a, b);
                        order.push(a);
                        has_predecessor.insert(b);
                    };

                    // Directed so the region above `level` is on the left. The
                    // direction is arbitrary for drawing but not for linking: with a
                    // consistent rule, a segment's end edge is the next cell's start
                    // edge, and the walk is a lookup rather than a search.
                    match code {
                        1 => link(e0, e3),
                        2 => link(e1, e0),
                        3 => link(e1, e3),
                        4 => link(e2, e1),
                        6 => link(e2, e0),
                        7 => link(e2, e3),
                        8 => link(e3, e2),
                        9 => link(e0, e2),
                        11 => link(e1, e2),
                        12 => link(e3, e1),
                        13 => link(e0, e1),
                        14 => link(e3, e0),
                        // The two saddles, resolved by the cell's mean -- the same rule
                        // contourpy uses. Whether the high corners are joined through
                        // the middle or separated is genuinely ambiguous from four
                        // samples; the mean is the linear interpolant's own answer.
                        5 => {
                            // c0 and c2 above
                            if (z0 + z1 + z2 + z3) / 4.0 > level {
                                link(e0, e1); // above connects; c1, c3 isolated
                                link(e2, e3);
                            } else {
                                link(e0, e3); // c0 and c2 isolated
                                link(e2, e1);
                            }
                        }
                        10 => {
                            // c1 and c3 above
                            if (z0 + z1 + z2 + z3) / 4.0 > level {
                                link(e3, e0); // above connects; c0, c2 isolated
                                link(e1, e2);
                            } else {
                                link(e1, e0); // c1 and c3 isolated
                                link(e3, e2);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    walk(&from, &order, &points, &has_predecessor)
}

/// Whether a phase level is the cut: ±π, where arg f jumps.
pub fn is_cut(level: f64) -> bool {
    (level.abs() - core::f64::consts::PI).abs() < 1e-9
}

/// Swift's `Float.pi`: π rounded toward zero, so that it is under π. (For
/// f64 the two roundings agree, and `Double.pi` is `f64::consts::PI`.)
pub const SWIFT_FLOAT_PI: f32 = f32::from_bits(0x4049_0FDA);

/// The branch cut of a phase grid, as iso-lines: where arg f passes from
/// π to -π, which no level of the grid itself reaches, since a landscape's
/// sample is never over π (`NativeLandscape::quantize` clamps into the
/// samples' range, so ±π is stored as ±`SWIFT_FLOAT_PI`). It is the zero
/// level of arg(-f) -- the same grid turned a half turn -- whose own cut
/// lies where arg f is zero.
/// Marching squares emits a bundle of spurious crossings along that wrap,
/// as along every wrap; a refiner drops them, which is why this is
/// contoured only when one is in play.
///
/// The half turn is Swift's `Float.pi`, which is π rounded toward zero
/// (0x40490FDA, 3.1415925), not `f32::consts::PI`, which is π rounded to
/// nearest (0x40490FDB, 3.1415927): the turned grid, and so every vertex of
/// the cut the grid places, is Swift's to the bit (crates/FINDINGS.md).
///
/// A sample on the cut, f real and negative, is ±π, its sign the sign of
/// Im f's zero; turned by the same float it is exactly zero, on the level
/// whichever the sign. That holds for a grid clamped as a landscape's is.
/// Swift assumes the clamp; here a sample over `SWIFT_FLOAT_PI` either way
/// (π rounded up to 0x40490FDB, say, from a grid not quantized) is first
/// clamped to it, so the cut never hangs on the sign of a rounding. On a
/// quantized grid the clamp moves nothing, and the bits are Swift's. A NaN
/// stays NaN.
pub fn cut(phase: &Grid2D<f32>) -> Vec<Vec<P2<DomainSpace>>> {
    let pi = SWIFT_FLOAT_PI;
    let turn = |v: f32| {
        let v = if v > pi {
            pi
        } else if v < -pi {
            -pi
        } else {
            v
        };
        if v > 0.0 { v - pi } else { v + pi }
    };
    let turned = Grid2D::new(phase.width, phase.height, phase.domain, phase.values.iter().map(|&v| turn(v)).collect());
    lines(&turned, 0.0, Some(&Index::new(&turned)))
}

/// Iso-lines at several levels, in level order -- the shape
/// `contours.contour_levels` returns.
///
/// Levels are independent of each other: that they *are* is a property of
/// contouring the whole grid at once. Swift runs them concurrently; here
/// they run in turn, in level order, and the answer is the same either way
/// (the handoff's rule for `concurrentPerform`).
pub fn levels(grid: &Grid2D<f32>, levels: &[f64]) -> Vec<(f64, Vec<Vec<P2<DomainSpace>>>)> {
    if levels.is_empty() {
        return Vec::new();
    }
    let index = Index::new(grid);
    levels.iter().map(|&level| (level, lines(grid, level, Some(&index)))).collect()
}

fn interpolate(grid: &Grid2D<f32>, level: f64, a: [usize; 2], b: [usize; 2]) -> P2<DomainSpace> {
    let za = f64::from(grid.at(a[0], a[1]));
    let zb = f64::from(grid.at(b[0], b[1]));
    // The denominator cannot be zero: this edge is only asked about when one
    // end is above the level and the other is not.
    let t = ((level - za) / (zb - za)).clamp(0.0, 1.0);
    let pa = grid.position(a[0], a[1]);
    let pb = grid.position(b[0], b[1]);
    P2::new(pa.x() + (pb.x() - pa.x()) * t, pa.y() + (pb.y() - pa.y()) * t)
}

/// Chain the segments into polylines: open runs first, from their starts,
/// then whatever is left, which is closed loops -- each in scan order, so
/// the answer is the same on every run.
fn walk<S>(
    from: &HashMap<EdgeID, EdgeID>,
    order: &[EdgeID],
    points: &HashMap<EdgeID, Option<P2<S>>>,
    has_predecessor: &HashSet<EdgeID>,
) -> Vec<Vec<P2<S>>> {
    let mut out = Vec::new();
    let mut visited: HashSet<EdgeID> = HashSet::new();

    let mut trace = |start: EdgeID, closing: bool, visited: &mut HashSet<EdgeID>| {
        let mut path: Vec<P2<S>> = Vec::new();
        let mut cursor = Some(start);
        while let Some(id) = cursor {
            if !visited.insert(id) {
                break;
            }
            if let Some(Some(p)) = points.get(&id) {
                path.push(*p);
            }
            cursor = from.get(&id).copied();
        }
        // A closed loop ends where it began; repeating the first vertex is
        // what contourpy does and what a polyline needs to close.
        if closing && path.len() >= 2 {
            path.push(path[0]);
        }
        if path.len() >= 2 {
            out.push(path);
        }
    };

    // Open chains: a start is an edge nothing leads into.
    for &id in order.iter().filter(|id| !has_predecessor.contains(id)) {
        trace(id, false, &mut visited);
    }
    // What remains is cycles.
    for &id in order {
        if !visited.contains(&id) {
            trace(id, true, &mut visited);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use crate::json::JsonValue;
    use crate::manifest::{Domain, Interval};
    use crate::simd::Vec2;

    // Comparing two sets of polylines directly means comparing decisions the
    // shape does not depend on: where a closed loop starts, which way round it
    // goes, whether an open run was traced from one end or the other. The
    // canonical form is the multiset of *segment midpoints*, which is invariant
    // to all three and still distinguishes any two different curve sets.
    fn midpoints(paths: &[Vec<P2<DomainSpace>>]) -> Vec<Vec2> {
        let mut out = Vec::new();
        for path in paths {
            for w in path.windows(2) {
                out.push(Vec2::new((w[0].x() + w[1].x()) / 2.0, (w[0].y() + w[1].y()) / 2.0));
            }
        }
        out
    }

    /// How many midpoints of one set have no counterpart in the other, and how
    /// far the worst stray is. Sorting and zipping would be cheaper and wrong:
    /// crossings on vertical grid edges share an x exactly, so a difference in
    /// the last bit permutes the order and then compares unrelated points.
    fn separation(a: &[Vec2], b: &[Vec2]) -> (usize, f64) {
        fn directed(from: &[Vec2], to: &[Vec2]) -> (usize, f64) {
            let (mut count, mut worst) = (0, 0.0f64);
            for &p in from {
                let mut best = f64::INFINITY;
                for &q in to {
                    best = best.min((p - q).mul_elements(p - q).sum());
                }
                let d = best.sqrt();
                if d > 1e-9 {
                    count += 1;
                    worst = worst.max(d);
                }
            }
            (count, worst)
        }
        let (ca, wa) = directed(a, b);
        let (cb, wb) = directed(b, a);
        (ca + cb, wa.max(wb))
    }

    /// "contour: the native marching squares reproduces contourpy", with
    /// Swift's tolerance: every midpoint within 1e-9 of one in the other set.
    #[test]
    fn the_native_marching_squares_reproduces_contourpy() {
        let index = fixtures::shared_json("contour/index.json");
        let grid = index.get("grid").unwrap();
        let nx = grid.get("nx").and_then(JsonValue::as_i64).unwrap() as usize;
        let ny = grid.get("ny").and_then(JsonValue::as_i64).unwrap() as usize;
        let bounds = |key: &str| -> Interval {
            let a = grid.get(key).and_then(JsonValue::as_array).unwrap();
            Interval::new(a[0].as_f64().unwrap(), a[1].as_f64().unwrap())
        };
        let domain = Domain::new(bounds("real"), bounds("imag"));
        let fields = index.get("fields").and_then(JsonValue::as_object).unwrap();
        let mut checked = 0;
        for (name, entries) in fields {
            let values = fixtures::shared_npy(&format!("contour/{name}.npy")).floats().unwrap();
            let g = Grid2D::new(nx, ny, domain, values);
            for entry in entries.as_array().expect("the index lists levels") {
                let level = entry.get("level").and_then(JsonValue::as_f64).unwrap();
                let tag = entry.get("tag").and_then(JsonValue::as_str).unwrap();
                let want_paths = entry.get("paths").and_then(JsonValue::as_i64).unwrap() as usize;

                // The fixture is in contourpy's (imag, real) column order; the
                // grid, and everything downstream of the bundle, is world order.
                let flat = fixtures::shared_npy(&format!("contour/{tag}.npy")).doubles().unwrap();
                let offsets = fixtures::shared_npy(&format!("contour/{tag}.idx.npy")).ints().unwrap();
                let want: Vec<Vec<P2<DomainSpace>>> = offsets
                    .windows(2)
                    .map(|w| (w[0] as usize..w[1] as usize).map(|k| P2::new(flat[2 * k + 1], flat[2 * k])).collect())
                    .collect();

                let got = lines(&g, level, None);
                let (a, b) = (midpoints(&got), midpoints(&want));
                let (unmatched, worst) = separation(&a, &b);
                assert!(
                    got.len() == want_paths && a.len() == b.len() && unmatched == 0,
                    "{tag}: {} paths, {} segments vs {want_paths}/{}; {unmatched} of {} midpoints unmatched, worst {worst}",
                    got.len(),
                    a.len(),
                    b.len(),
                    a.len() + b.len()
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 13, "every level the index lists");
    }

    /// "contour: closed loops close and open runs reach the edge"
    #[test]
    fn closed_loops_close_and_open_runs_reach_the_edge() {
        // A cone: one nested closed loop per level, none of them touching the
        // grid boundary.
        let n = 81usize;
        let mut values = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                let dx = (x as f64 - (n / 2) as f64) / (n / 2) as f64;
                let dy = (y as f64 - (n / 2) as f64) / (n / 2) as f64;
                values[y * n + x] = (1.0 - (dx * dx + dy * dy).sqrt()) as f32;
            }
        }
        let domain = Domain::new(Interval::new(-1.0, 1.0), Interval::new(-1.0, 1.0));
        let g = Grid2D::new(n, n, domain, values);
        let loops = lines(&g, 0.5, None);
        assert_eq!(loops.len(), 1, "one level set, one loop");
        let lp = &loops[0];
        assert_eq!(lp.first(), lp.last(), "and it closes");
        // Radius 0.5 of a unit cone: circumference pi.
        let length: f64 = lp.windows(2).map(|w| (w[1].v - w[0].v).length()).sum();
        assert!(
            (length - core::f64::consts::PI / 2.0 * 2.0).abs() < 0.02,
            "with the circumference of a circle of radius 1/2: {length} vs {}",
            core::f64::consts::PI
        );

        // A ramp: one open run per level, ending on the grid boundary.
        let mut ramp = vec![0.0f32; n * n];
        for y in 0..n {
            for x in 0..n {
                ramp[y * n + x] = x as f32 / (n - 1) as f32;
            }
        }
        let r = Grid2D::new(n, n, domain, ramp);
        let runs = lines(&r, 0.5, None);
        assert!(runs.len() == 1 && runs[0].first() != runs[0].last(), "a ramp gives one open run, not a loop");
        assert!(runs[0].iter().all(|p| (p.x() - 0.0).abs() < 1e-9), "at exactly the level's coordinate");
    }

    #[test]
    fn levels_come_back_in_level_order() {
        let n = 33usize;
        let values: Vec<f32> = (0..n * n).map(|k| (k % n) as f32 + (k / n) as f32).collect();
        let g = Grid2D::new(n, n, Domain::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0)), values);
        let got = levels(&g, &[40.0, 3.5, 10.0]);
        assert_eq!(got.iter().map(|l| l.0).collect::<Vec<_>>(), [40.0, 3.5, 10.0]);
        for (level, paths) in got {
            assert_eq!(paths, lines(&g, level, None));
        }
        assert!(is_cut(core::f64::consts::PI) && is_cut(-core::f64::consts::PI) && !is_cut(3.0));
    }

    /// The branch cut is placed on the grid turned by Swift's `Float.pi`, π
    /// rounded toward zero, so its vertices are Swift's to the bit. Turned
    /// by π rounded to nearest, every vertex of the cut moved by about a
    /// ten-millionth of a cell, and the refiner, handed those vertices,
    /// split and followed the six drawings' phase lines otherwise than
    /// Swift (crates/FINDINGS.md, 2026-11-16).
    #[test]
    fn the_cut_turns_the_grid_by_swifts_float_pi() {
        assert_eq!(SWIFT_FLOAT_PI.to_bits(), 0x4049_0FDA);
        assert!(f64::from(SWIFT_FLOAT_PI) < core::f64::consts::PI);
        assert_eq!(SWIFT_FLOAT_PI.to_bits() + 1, core::f32::consts::PI.to_bits(), "an ulp under nearest");
        // arg f is 3 on the left column and -2 on the right: the cut, the
        // zero of arg(-f), crosses each row where the turned samples
        // 3 - π and -2 + π, in float, interpolate to zero.
        let domain = Domain::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0));
        let cut = cut(&Grid2D::new(2, 2, domain, vec![3.0, -2.0, 3.0, -2.0]));
        let at = |pi: f32| {
            let (za, zb) = (f64::from(3.0 - pi), f64::from(-2.0 + pi));
            -za / (zb - za)
        };
        let (swift, nearest) = (at(SWIFT_FLOAT_PI), at(core::f32::consts::PI));
        assert!((swift - nearest).abs() > 1e-8, "the two π place the crossing apart: {swift} {nearest}");
        assert!(cut.len() == 1 && cut[0].len() == 2, "one run across the two rows: {cut:?}");
        assert!(cut[0].iter().all(|p| p.x() == swift), "at Swift's crossing, {swift}: {cut:?}");

        // A grid not clamped as a landscape's is: the cut runs down the
        // middle column, its samples π rounded up, one sign or the other as
        // rounding fell. Clamped to Float.pi they turn to zero, and the cut
        // is the column, straight, whichever the signs.
        let up = core::f32::consts::PI;
        let domain = Domain::new(Interval::new(0.0, 2.0), Interval::new(0.0, 3.0));
        for signs in [[1.0f32, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]] {
            let mut values = Vec::new();
            for s in signs {
                values.extend([3.0, s * up, -3.0]);
            }
            let cut = super::cut(&Grid2D::new(3, 4, domain, values));
            assert!(cut.len() == 1 && cut[0].len() == 4, "{signs:?}: one run up the column: {cut:?}");
            assert!(cut[0].iter().all(|p| p.x() == 1.0), "{signs:?}: straight, at x = 1: {cut:?}");
        }
    }
}
