//! The coordinate image: the depth pass with a parametric surface's
//! coordinate beside each pixel's depth. The type of
//! `Sources/KurvenCore/SurfaceImage.swift`, whole: the type the scene's
//! visibility takes, and the CPU rasterizer that draws it.

use crate::depth::{DepthFrame, DepthImage};
use kurven_math::swift;

use crate::parametric::ParametricSurface;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, ParamSpace, Transform, ViewSpace, WorldSpace};

/// What the coordinate pass draws: the depth image, and the surface
/// coordinates of the front-most point at each pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceImage {
    pub depth: DepthImage,
    /// The surface coordinates of the front-most point at each pixel, row
    /// major like `depth.values`; NaN where nothing was drawn. Unwrapped as
    /// rasterized, so compare them with `ParamAxis::difference`.
    pub coords: Vec<[f32; 2]>,
}

/// A coordinate image whose coordinates do not match its pixels: Swift's
/// precondition, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinateCount {
    pub coordinates: usize,
    pub pixels: usize,
}

impl SurfaceImage {
    /// One coordinate per pixel, or the counts that differ.
    pub fn new(depth: DepthImage, coords: Vec<[f32; 2]>) -> Result<SurfaceImage, CoordinateCount> {
        if coords.len() != depth.values.len() {
            return Err(CoordinateCount { coordinates: coords.len(), pixels: depth.values.len() });
        }
        Ok(SurfaceImage { depth, coords })
    }

    pub fn frame(&self) -> DepthFrame {
        self.depth.frame
    }

    /// The front-most surface coordinate at a pixel, or None where nothing was
    /// drawn.
    pub fn coordinate(&self, row: i64, col: i64) -> Option<P2<ParamSpace>> {
        let c = self.coords[(row * self.frame().cols + col) as usize];
        if c[0].is_nan() { None } else { Some(P2::new(f64::from(c[0]), f64::from(c[1]))) }
    }

    /// Rasterize a parametric surface's grid on the CPU.
    ///
    /// The reference the GPU coordinate pass is held to, as the Python
    /// Z-buffer is for the depth pass. Two triangles per grid cell, sampled
    /// at the frame's grid points -- the same points `metal_clip` puts pixel
    /// centres on -- keeping the greatest view z, and carrying the
    /// interpolated surface coordinate of whichever triangle won.
    ///
    /// Swift's bake takes the image from Metal (`MetalRenderer.renderSurface`),
    /// which Swift's suite holds to this function; the port has no GPU in
    /// its bake and takes this one, which is deterministic in f64
    /// (crates/FINDINGS.md, "The coordinate image is Swift's CPU rasterizer").
    /// Every choice is Swift's: a sample is covered where its three
    /// barycentric weights are each at least -1e-12 (inclusive, so a sample on
    /// a shared edge is claimed by both triangles); depth and coordinate are
    /// the weights' combination of the corners' in f64; the depth is kept as
    /// a float32 and replaced only by a strictly greater one, so of two equal
    /// depths the triangle drawn first, cell by cell in rows of `u` and the
    /// cell's (00, 10, 01) before its (10, 11, 01), keeps the pixel; a
    /// triangle whose doubled area in pixels is within 1e-12 of zero draws
    /// nothing. A corner that is not a number, where Swift's `Int` of the
    /// triangle's pixel bounds traps, leaves its triangle undrawn.
    pub fn rasterize(
        surface: &ParametricSurface,
        view: &Transform<WorldSpace, ViewSpace>,
        frame: DepthFrame,
    ) -> SurfaceImage {
        let (nu, nv) = (surface.u.cells(), surface.v.cells());
        // An axis of one sample, or of no length, has every sample at one
        // view coordinate, its `lo` (`DepthFrame::coordinate`): Swift's
        // scale, (n - 1) over the length, is then 0 or 0/0 and every
        // triangle is dropped. Such an axis is rasterized once, at that
        // coordinate, in view units (scale 1, the sample at 0), and the
        // result copied to its every sample. A regular axis is Swift's to
        // the bit.
        let axis = |n: i64, length: f64| -> (i64, f64) {
            if n >= 2 && length > 0.0 { (n, (n - 1) as f64 / length) } else { (1, 1.0) }
        };
        let (rows, row_scale) = axis(frame.rows, frame.axis0.length());
        let (cols, col_scale) = axis(frame.cols, frame.axis1.length());
        let flat_rows = !(frame.rows >= 2 && frame.axis0.length() > 0.0);
        let flat_cols = !(frame.cols >= 2 && frame.axis1.length() > 0.0);

        // Every grid vertex once, as (row, col, view z) in continuous pixel
        // units, with its unwrapped coordinate beside it.
        let mut pixel: Vec<Vec3> = Vec::with_capacity((nu + 1) * (nv + 1));
        let mut param: Vec<Vec2> = Vec::with_capacity((nu + 1) * (nv + 1));
        for j in 0..=nv as i64 {
            for i in 0..=nu as i64 {
                let q = view.apply(surface.position(i, j));
                let (row, col) = frame.components(q.xy());
                pixel.push(Vec3::new((row - frame.axis0.lo) * row_scale, (col - frame.axis1.lo) * col_scale, q.z()));
                param.push(Vec2::new(surface.u.coordinate(i), surface.v.coordinate(j)));
            }
        }

        let count = (rows * cols) as usize;
        let mut depth = vec![f32::NEG_INFINITY; count];
        let mut coords = vec![[f32::NAN, f32::NAN]; count];

        let edge = |a: Vec3, b: Vec3, r: f64, c: f64| -> f64 { (b.x - a.x) * (c - a.y) - (r - a.x) * (b.y - a.y) };
        let min3 = |a: f64, b: f64, c: f64| swift::min(swift::min(a, b), c);
        let max3 = |a: f64, b: f64, c: f64| swift::max(swift::max(a, b), c);
        let mut triangle = |k0: usize, k1: usize, k2: usize| {
            let (p0, p1, p2) = (pixel[k0], pixel[k1], pixel[k2]);
            let area = edge(p0, p1, p2.x, p2.y);
            if !(area.abs() > 1e-12) {
                // A triangle seen edge-on across a flat axis lies on that
                // axis's one sample line: at each sample of the other axis,
                // its front-most point is on one of its edges.
                let corners = [(p0, param[k0]), (p1, param[k1]), (p2, param[k2])];
                if flat_rows && !flat_cols && corners.iter().all(|(p, _)| p.x == 0.0) {
                    collapsed(&corners, |p| p.y, cols, &mut |c, z, t| {
                        keep(&mut depth, &mut coords, c as usize, z, t);
                    });
                } else if flat_cols && !flat_rows && corners.iter().all(|(p, _)| p.y == 0.0) {
                    collapsed(&corners, |p| p.x, rows, &mut |r, z, t| {
                        keep(&mut depth, &mut coords, (r * cols) as usize, z, t);
                    });
                }
                return;
            }
            let (Some(r0), Some(r1), Some(c0), Some(c1)) = (
                swift::int(min3(p0.x, p1.x, p2.x).ceil()),
                swift::int(max3(p0.x, p1.x, p2.x).floor()),
                swift::int(min3(p0.y, p1.y, p2.y).ceil()),
                swift::int(max3(p0.y, p1.y, p2.y).floor()),
            ) else {
                return;
            };
            let (r0, r1, c0, c1) = (r0.max(0), r1.min(rows - 1), c0.max(0), c1.min(cols - 1));
            if !(r0 <= r1 && c0 <= c1) {
                return;
            }
            let (t0, t1, t2) = (param[k0], param[k1], param[k2]);
            for r in r0..=r1 {
                let rr = r as f64;
                for c in c0..=c1 {
                    let cc = c as f64;
                    let w0 = edge(p1, p2, rr, cc) / area;
                    let w1 = edge(p2, p0, rr, cc) / area;
                    let w2 = 1.0 - w0 - w1;
                    // Inclusive: a sample on a shared edge is claimed by both
                    // triangles and the greater depth decides, as MAX does.
                    if !(w0 >= -1e-12 && w1 >= -1e-12 && w2 >= -1e-12) {
                        continue;
                    }
                    let z = w0 * p0.z + w1 * p1.z + w2 * p2.z;
                    let k = (r * cols + c) as usize;
                    if (z as f32) > depth[k] {
                        depth[k] = z as f32;
                        let t = w0 * t0 + w1 * t1 + w2 * t2;
                        coords[k] = [t.x as f32, t.y as f32];
                    }
                }
            }
        };

        let stride = nu + 1;
        for j in 0..nv {
            for i in 0..nu {
                let a = j * stride + i;
                let (b, c) = (a + 1, a + stride);
                let d = c + 1;
                triangle(a, b, c);
                triangle(b, d, c);
            }
        }
        if rows == frame.rows && cols == frame.cols {
            return SurfaceImage { depth: DepthImage::new(frame, depth, f32::NEG_INFINITY), coords };
        }
        // A flat axis's one line of samples, copied to its every sample.
        let full = (frame.rows * frame.cols) as usize;
        let (mut all_depth, mut all_coords) = (Vec::with_capacity(full), Vec::with_capacity(full));
        for r in 0..frame.rows {
            for c in 0..frame.cols {
                let k = (r.min(rows - 1) * cols + c.min(cols - 1)) as usize;
                all_depth.push(depth[k]);
                all_coords.push(coords[k]);
            }
        }
        SurfaceImage { depth: DepthImage::new(frame, all_depth, f32::NEG_INFINITY), coords: all_coords }
    }
}

/// Keeps a sample's depth and coordinate where it is strictly in front of
/// what the pixel holds, as `rasterize` does.
fn keep(depth: &mut [f32], coords: &mut [[f32; 2]], k: usize, z: f64, t: Vec2) {
    if (z as f32) > depth[k] {
        depth[k] = z as f32;
        coords[k] = [t.x as f32, t.y as f32];
    }
}

/// A triangle collapsed onto a flat axis's sample line, rasterized along
/// the other axis (`along` reads a corner's pixel coordinate on it, `n` its
/// samples): at each sample its cross-section is a segment in depth, whose
/// front-most point lies on an edge; that edge's point, its depth and its
/// coordinate interpolated along the edge, is handed to `emit`. An edge
/// across the sample at its end is its end.
fn collapsed(corners: &[(Vec3, Vec2); 3], along: impl Fn(Vec3) -> f64, n: i64, emit: &mut impl FnMut(i64, f64, Vec2)) {
    let values = corners.map(|(p, _)| along(p));
    let lo = swift::min(swift::min(values[0], values[1]), values[2]);
    let hi = swift::max(swift::max(values[0], values[1]), values[2]);
    let (Some(k0), Some(k1)) = (swift::int(lo.ceil()), swift::int(hi.floor())) else { return };
    for k in k0.max(0)..=k1.min(n - 1) {
        let s = k as f64;
        let mut best: Option<(f64, Vec2)> = None;
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let ((pa, ta), (pb, tb)) = (corners[a], corners[b]);
            let (va, vb) = (values[a], values[b]);
            let points: Vec<(f64, Vec2)> = if va == vb {
                if va == s { vec![(pa.z, ta), (pb.z, tb)] } else { Vec::new() }
            } else if (va - s) * (vb - s) <= 0.0 {
                let t = (s - va) / (vb - va);
                vec![(pa.z + (pb.z - pa.z) * t, ta + (tb - ta) * t)]
            } else {
                Vec::new()
            };
            for (z, t) in points {
                if best.is_none_or(|(front, _)| z > front) {
                    best = Some((z, t));
                }
            }
        }
        if let Some((z, t)) = best {
            emit(k, z, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::manifest::Interval;
    use crate::parametric::{JetMap, ParamAxis, SurfaceJet};

    /// The annulus 1/2 ≤ ρ ≤ 1 in the plane y = 0, its angle round the y
    /// axis, which the identity view sees edge-on: every point at view y 0.
    fn edge_on_annulus() -> ParametricSurface {
        let map: JetMap = Arc::new(|c| {
            let (s, t) = (kurven_math::real::sin(c.x()), kurven_math::real::cos(c.x()));
            SurfaceJet::new(
                Vec3::new(c.y() * t, 0.0, c.y() * s),
                Vec3::new(-c.y() * s, 0.0, c.y() * t),
                Vec3::new(t, 0.0, s),
            )
        });
        let u = ParamAxis::angle(64);
        let v = ParamAxis::new(Interval::new(0.5, 1.0), false, 9);
        ParametricSurface::new(u, v, false, None, None, map).expect("the annulus")
    }

    /// Seen edge-on, the annulus's view bounds have no extent across it, and
    /// every sample of that axis lies on its line: each is covered where the
    /// annulus is, at its front-most depth, the same down the axis.
    #[test]
    fn a_surface_seen_edge_on_is_drawn_on_its_line() {
        let surface = edge_on_annulus();
        let view = Transform::<WorldSpace, ViewSpace>::from_rows(
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vec3::ZERO,
        );
        // Rows along view x, columns along view y, which is 0 throughout.
        let frame = DepthFrame::new(Interval::new(-1.0, 1.0), Interval::new(0.0, 0.0), 41, 7);
        let image = SurfaceImage::rasterize(&surface, &view, frame);
        let mut worst: f64 = 0.0;
        for r in 0..frame.rows {
            let x = frame.coordinate(r, 0).x();
            // The front-most point over x is on the outer rim, as the grid
            // draws it: the polygon of its 64 samples.
            let rim = |i: i64| surface.position(i, 8).v;
            let truth = (0..64)
                .filter_map(|i| {
                    let (a, b) = (rim(i), rim(i + 1));
                    let t = (x - a.x) / (b.x - a.x);
                    (b.x != a.x && (0.0..=1.0).contains(&t)).then_some(a.z + (b.z - a.z) * t)
                })
                .fold(f64::NEG_INFINITY, f64::max);
            for c in 0..frame.cols {
                assert!(image.depth.is_covered(r, c), "row {r} col {c} covered");
                assert_eq!(image.depth.at(r, c), image.depth.at(r, 0), "the same down the flat axis");
                let q = image.coordinate(r, c).expect("a coordinate");
                assert!(q.y() > 0.99, "on the outer rim: radius {}", q.y());
            }
            worst = worst.max((image.depth.at(r, 0) - truth).abs());
        }
        // To the float32 the depth is kept in.
        assert!(worst < 6e-8, "its depth is the drawn rim's: {worst}");
    }

    /// A frame of one row samples one line of view x: the same as that row
    /// of a frame of three whose middle row lies on it.
    #[test]
    fn a_frame_of_one_row_is_that_row_of_a_frame_of_three() {
        let surface = ParametricSurface::torus(2.0, 1.0, (96, 48)).expect("the torus");
        let view = Transform::<WorldSpace, ViewSpace>::from_rows(
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.6, -0.8), Vec3::new(0.0, 0.8, 0.6)),
            Vec3::ZERO,
        );
        let one = DepthFrame::new(Interval::new(0.25, 1.25), Interval::new(-3.0, 3.0), 1, 300);
        let three = DepthFrame::new(Interval::new(-0.75, 1.25), Interval::new(-3.0, 3.0), 3, 300);
        let (a, b) = (SurfaceImage::rasterize(&surface, &view, one), SurfaceImage::rasterize(&surface, &view, three));
        assert_eq!(three.coordinate(1, 0).x(), 0.25, "the middle row lies on the one row's line");
        let (mut covered, mut differ, mut worst) = (0, 0, 0.0f64);
        for c in 0..300 {
            let (p, q) = (a.depth.is_covered(0, c), b.depth.is_covered(1, c));
            if p {
                covered += 1;
            }
            if p != q {
                differ += 1;
            } else if p {
                worst = worst.max((a.depth.at(0, c) - b.depth.at(1, c)).abs());
            }
        }
        assert!(covered > 100, "the row crosses the torus: {covered} covered");
        assert!(differ <= 2 && worst < 1e-9, "{differ} samples differ in coverage, depth by {worst}");
    }
}
