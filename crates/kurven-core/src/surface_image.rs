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
        let (rows, cols) = (frame.rows, frame.cols);
        let row_scale = (rows - 1) as f64 / frame.axis0.length();
        let col_scale = (cols - 1) as f64 / frame.axis1.length();

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
        SurfaceImage { depth: DepthImage::new(frame, depth, f32::NEG_INFINITY), coords }
    }
}
