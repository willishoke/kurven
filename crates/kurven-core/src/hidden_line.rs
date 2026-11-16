//! Hidden-line removal against a depth image: the depth-buffer half of
//! `Sources/KurvenCore/HiddenLine.swift`. `HeightfieldVisibility`, the
//! other half, needs the grid traversal and comes with it.
//!
//! A line-for-line port of `outline.clip_hidden_lines`. A vertex is visible when
//! `z + depthTolerance > buffer[pixel]`; hidden vertices split a path into runs; runs
//! shorter than two vertices draw nothing and are dropped. Because it takes a
//! `DepthImage` *value*, it is tested against a Python-dumped buffer and
//! Python-dumped polylines with no GPU anywhere in the loop -- which is what
//! makes "the bake reproduces the drawing" a checkable claim rather than a
//! rendering that looks about right.
//!
//! The realtime preview does the same test per *fragment* in a shader, which can
//! disagree with this on sub-pixel runs. That disagreement is deliberate and
//! bounded: the preview is for navigating, the bake is the artifact, and the
//! bake is this.

use crate::contour;
use crate::depth::{DepthImage, RasterOrder};
use crate::grid::Grid2D;
use crate::manifest::{Domain, Interval};
use crate::polyline::PolylineSet;
use crate::spaces::{P3, PageSpace, ViewSpace};

/// Where along a segment a face turns edge-on, as a fraction of the way,
/// given that the facing at its ends, `fa` and `fb`, differs in sign:
/// bisection on the exact facing at a fraction, to the last bit. The facing
/// is continuous along the segment and its sign is all that is asked of it,
/// so nothing faster is also as certain.
pub fn fold_crossing(fa: f64, _fb: f64, mut facing: impl FnMut(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1.0);
    let front_at_lo = fa > 0.0;
    while hi - lo > 1e-13 {
        let mid = 0.5 * (lo + hi);
        if (facing(mid) > 0.0) == front_at_lo {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Hidden-line removal: a pure function of a depth image and some polylines.
pub enum HiddenLine {}

impl HiddenLine {
    pub fn clip(paths: &PolylineSet<ViewSpace>, depth: &DepthImage, depth_tolerance: f64) -> PolylineSet<PageSpace> {
        let mut out: Vec<Vec<P3<PageSpace>>> = Vec::new();
        for i in 0..paths.count() {
            let mut run: Vec<P3<PageSpace>> = Vec::new();
            for v in paths.path(i) {
                if v.z() + depth_tolerance > depth.depth(v.xy()) {
                    run.push(P3::new(v.x(), v.y(), v.z()));
                } else {
                    if run.len() >= 2 {
                        out.push(std::mem::take(&mut run));
                    }
                    run.clear();
                }
            }
            if run.len() >= 2 {
                out.push(run);
            }
        }
        PolylineSet::from_paths(&out)
    }

    /// Curves that are not depth-tested: every layer the bake draws, which
    /// arrives already tested by its face and its scene's occlusion test
    /// (`Scene.visibleLayers`). Passing it through this rather than casting
    /// keeps "which space am I in" a decision the type system witnesses.
    pub fn pass(paths: &PolylineSet<ViewSpace>) -> PolylineSet<PageSpace> {
        PolylineSet::new(paths.vertices.iter().map(|v| P3::from_vec(v.v)).collect(), paths.offsets.clone(), None)
    }
}

impl DepthImage {
    /// The outline of the drawn region -- `outline.extract_outline`.
    ///
    /// Binarize coverage into a signed field and take its zero level set. The
    /// Python version does this by handing the binarized buffer to matplotlib's
    /// contouring and mapping the result back out of pixel space; with traversing
    /// squares in Core it is four lines, and it needs no plotting library to
    /// find the edge of a picture.
    ///
    /// None of the three plates draws one -- they are bounded by their own wall
    /// curves -- so the bake offers it rather than assuming it.
    pub fn outline(&self) -> PolylineSet<PageSpace> {
        let frame = self.frame;
        if frame.rows < 2 || frame.cols < 2 {
            return PolylineSet::empty();
        }
        let coverage: Vec<f32> = self.values.iter().map(|&v| if v > self.empty { 1.0 } else { -1.0 }).collect();
        // A grid whose sample positions are the frame's own grid, so a
        // contour vertex comes out in view units without a second mapping.
        let grid = Grid2D::new(
            frame.cols as usize,
            frame.rows as usize,
            Domain::new(Interval::new(frame.axis1.lo, frame.axis1.hi), Interval::new(frame.axis0.lo, frame.axis0.hi)),
            coverage,
        );
        let paths: Vec<Vec<P3<PageSpace>>> = contour::lines(&grid, 0.0, None)
            .into_iter()
            .map(|line| {
                line.into_iter()
                    .map(|p| match frame.order {
                        // `p.x` is the column value and `p.y` the row value;
                        // which view component each is depends on the
                        // frame's raster order.
                        RasterOrder::Buffer => P3::new(p.y(), p.x(), 0.0),
                        RasterOrder::Screen => P3::new(p.x(), p.y(), 0.0),
                    })
                    .collect()
            })
            .collect();
        PolylineSet::from_paths(&paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::DepthFrame;
    use crate::fixtures;
    use crate::json::Members;
    use crate::spaces::P2;

    /// `main.swift`'s "clip: reproduces the committed visible runs exactly".
    /// `clip` is pure and takes a `DepthImage` value, so this runs the real
    /// hidden-line stage against a frozen CPU depth buffer with no GPU in
    /// the loop. Exact equality, not a tolerance: same integer pixel lookup,
    /// same run splitting, same dropped singletons.
    #[test]
    fn reproduces_the_committed_visible_runs_exactly() {
        let meta = fixtures::shared_json("clip/meta.json");
        let meta = meta.object("clip meta").unwrap();
        let axis0 = meta.doubles("axis0", "clip meta").unwrap();
        let axis1 = meta.doubles("axis1", "clip meta").unwrap();
        let shape = meta.ints("shape", "clip meta").unwrap();
        let depth_tolerance = meta.double("depthTolerance", "clip meta").unwrap();
        let frame =
            DepthFrame::new(Interval::new(axis0[0], axis0[1]), Interval::new(axis1[0], axis1[1]), shape[0], shape[1])
                .with_nudge(meta.double("nudge", "clip meta").unwrap());
        let depth = DepthImage::new(frame, fixtures::shared_npy("clip/depth.npy").floats().unwrap(), f32::NEG_INFINITY);
        let view: Vec<P3<ViewSpace>> = fixtures::shared_npy("clip/view.npy")
            .rows3()
            .unwrap()
            .into_iter()
            .map(|r| P3::new(r[0], r[1], r[2]))
            .collect();
        let offsets: Vec<usize> =
            fixtures::shared_npy("clip/view.idx.npy").ints().unwrap().into_iter().map(|i| i as usize).collect();
        let got = HiddenLine::clip(&PolylineSet::new(view, offsets, None), &depth, depth_tolerance);

        let want_vertices = fixtures::shared_npy("clip/expected.npy");
        let want_offsets: Vec<usize> =
            fixtures::shared_npy("clip/expected.idx.npy").ints().unwrap().into_iter().map(|i| i as usize).collect();
        assert_eq!(
            got.offsets,
            want_offsets,
            "the same runs, split at the same vertices: {} vs {}",
            got.count(),
            want_offsets.len() - 1
        );
        let flat = want_vertices.doubles().unwrap();
        assert_eq!(want_vertices.shape, vec![got.vertices.len(), 2]);
        for (i, v) in got.vertices.iter().enumerate() {
            assert!(v.x() == flat[2 * i] && v.y() == flat[2 * i + 1], "every surviving vertex is bit-identical: {i}");
        }
    }

    /// `main.swift`'s "clip: the GPU's empty sentinel and Python's -infinity
    /// decide alike". Carrying the sentinel instead of rewriting it saves a
    /// pass over every pixel of the buffer. It is only sound because the
    /// visibility test cannot tell the two apart, which is what this asserts.
    #[test]
    fn the_gpus_empty_sentinel_and_pythons_minus_infinity_decide_alike() {
        let frame = DepthFrame::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0), 2, 2);
        let sentinel: f32 = -1e30;
        let python = DepthImage::new(frame, vec![f32::NEG_INFINITY; 4], f32::NEG_INFINITY);
        let metal = DepthImage::new(frame, vec![sentinel; 4], sentinel);
        for z in [-1e6f32, -1.0, 0.0, 1e6].map(f64::from) {
            for depth_tolerance in [0.0, 0.02, 0.2] {
                let a = z + depth_tolerance > python.depth(P2::new(0.25, 0.25));
                let b = z + depth_tolerance > metal.depth(P2::new(0.25, 0.25));
                assert_eq!(a, b, "every visibility test agrees over both conventions");
            }
        }
        assert!(!python.is_covered(0, 0) && !metal.is_covered(0, 0), "and coverage still reads as empty under both");
    }

    /// `main.swift`'s "clip: out-of-frame vertices read as occluded, as they
    /// do in Python".
    #[test]
    fn out_of_frame_vertices_read_as_occluded() {
        let frame = DepthFrame::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0), 4, 4);
        let depth = DepthImage::new(frame, vec![f32::NEG_INFINITY; 16], f32::NEG_INFINITY);
        assert_eq!(depth.depth(P2::new(0.5, 0.5)), f64::NEG_INFINITY, "inside and empty is -inf");
        assert_eq!(depth.depth(P2::new(-5.0, 0.5)), f64::INFINITY, "left of the frame is +inf");
        assert_eq!(depth.depth(P2::new(0.5, 9.0)), f64::INFINITY, "past the frame is +inf");
    }

    #[test]
    fn a_hidden_vertex_splits_a_run_and_a_singleton_is_dropped() {
        let frame = DepthFrame::new(Interval::new(0.0, 1.0), Interval::new(0.0, 4.0), 2, 5);
        // Column 2 is covered at depth 10, everything else empty.
        let mut values = vec![f32::NEG_INFINITY; 10];
        values[2] = 10.0;
        values[7] = 10.0;
        let depth = DepthImage::new(frame, values, f32::NEG_INFINITY);
        let path: Vec<P3<ViewSpace>> = (0..5).map(|k| P3::new(0.0, k as f64, 0.0)).collect();
        let lone: Vec<P3<ViewSpace>> = vec![P3::new(0.0, 0.0, 0.0), P3::new(0.0, 2.0, 0.0)];
        let got = HiddenLine::clip(&PolylineSet::from_paths(&[path, lone]), &depth, 0.02);
        assert_eq!(got.offsets, vec![0, 2, 4], "two runs either side of the hidden vertex, the singleton gone");
        let passed = HiddenLine::pass(&PolylineSet::from_paths(&[vec![
            P3::<ViewSpace>::new(1.0, 2.0, 3.0),
            P3::new(4.0, 5.0, 6.0),
        ]]));
        assert_eq!(passed.vertices, vec![P3::new(1.0, 2.0, 3.0), P3::new(4.0, 5.0, 6.0)]);
    }

    #[test]
    fn the_outline_is_the_edge_of_the_covered_region() {
        // A 5x5 frame over [0, 4]², covered on the middle 3x3 block.
        let frame = DepthFrame::new(Interval::new(0.0, 4.0), Interval::new(0.0, 4.0), 5, 5);
        let mut values = vec![f32::NEG_INFINITY; 25];
        for r in 1..4 {
            for c in 1..4 {
                values[r * 5 + c] = 0.0;
            }
        }
        let outline = DepthImage::new(frame, values, f32::NEG_INFINITY).outline();
        assert_eq!(outline.count(), 1, "one closed curve");
        let path = outline.path(0);
        assert_eq!(path[0], path[path.len() - 1]);
        for v in path {
            let (x, y) = (v.x(), v.y());
            let ring = (x - 2.0).abs().max((y - 2.0).abs());
            assert_eq!(ring, 1.5, "halfway between covered and empty: {v:?}");
        }
        let tiny = DepthImage::new(
            DepthFrame::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0), 1, 3),
            vec![0.0; 3],
            f32::NEG_INFINITY,
        );
        assert!(tiny.outline().is_empty(), "a frame one row high has none");
    }

    #[test]
    fn fold_crossing_bisects_to_the_last_bit() {
        let t = fold_crossing(1.0, -1.0, |s| 0.3 - s);
        assert!((t - 0.3).abs() < 1e-13);
        let t = fold_crossing(-2.0, 1.0, |s| s * s - 0.5);
        assert!((t - 0.5f64.sqrt()).abs() < 1e-13);
    }
}
