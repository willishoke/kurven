//! The depth pass of a landscape or a solid on the CPU: the triangles
//! `MetalRenderer.renderDepth` draws (`kv_height_vertex`, `kv_mesh_vertex`),
//! rasterized into a `DepthImage` as `SurfaceImage::rasterize` rasterizes a
//! parametric surface's. The page draws the same triangles on its GPU for the
//! preview (web/app/src/gpu); a bake and a pick read them here, where no GPU
//! is, natively and in the Worker alike.
//!
//! What a bake reads off the buffer is its outline (`DepthImage::outline`)
//! and nothing else: every stroke is tested by its face and the scene's
//! occlusion test, so no stroke depends on this pass. Its choices are
//! `SurfaceImage::rasterize`'s, all in doubles and deterministic, where Metal
//! interpolates in float32 under its own rasterization rules: the same
//! triangles in the same order, sampled at the frame's grid points (where
//! `metal_clip` puts pixel centres); a sample covered where all three
//! barycentric weights are at least -1e-12, so an edge sample is claimed by
//! both triangles; the depth kept as a float32 and replaced only by a
//! strictly greater one, which is the MAX blend's; a triangle of doubled
//! pixel area within 1e-12 of zero, or with a corner that is not a number,
//! draws nothing.

use std::sync::Arc;

use crate::depth::{DepthFrame, DepthImage};
use crate::grid::Grid2D;
use crate::manifest::Truncation;
use crate::scene::{Heightfield, Scene, SceneGeometry};
use crate::simd::Vec3;
use crate::spaces::{P2, P3, Transform, ViewSpace, WorldSpace};
use kurven_math::swift;

/// The grid a landscape's depth pass draws its heights from, the one rule
/// every reader of it shares (the GPU's resources, the CPU's pass, a pick):
/// the bundle's |f| under no truncation or a uniform one, which the pass
/// then caps at `ceiling`, or the clamped grid under bands, whose height
/// depends on x and is folded in (`SceneResources.init`). Borrowed where it
/// is the bundle's own.
pub struct DrawnGrid {
    pub grid: Arc<Grid2D<f32>>,
    /// Whether the truncation is in the grid already (`truncationBaked`).
    pub truncation_baked: bool,
    /// The uniform truncation height the pass applies, `None` where it
    /// applies none.
    pub ceiling: Option<f64>,
    /// The subsampling step, at least one. Swift's Metal side passes
    /// `Heightfield.step` as it is, where every CPU reader of it in Swift
    /// takes `max(step, 1)` (Mesh.swift, Surface.swift, Scene.swift): a step
    /// of none would draw a flat sheet at one texel. One rule here, the CPU's.
    pub step: usize,
}

impl DrawnGrid {
    pub fn of(h: &Heightfield) -> DrawnGrid {
        let (grid, truncation_baked) = match h.surface.truncation {
            Truncation::None | Truncation::Uniform(_) => (h.surface.height.clone(), false),
            Truncation::RealBands { .. } => (Arc::new(h.surface.clamped()), true),
        };
        DrawnGrid { grid, truncation_baked, ceiling: DrawnGrid::ceiling_of(h), step: h.step.max(1) }
    }

    /// The uniform truncation height the pass applies to the grid it reads,
    /// without building the grid: `None` under none, and under bands, which
    /// the grid holds already (`truncationBaked`).
    pub fn ceiling_of(h: &Heightfield) -> Option<f64> {
        match h.surface.truncation {
            Truncation::Uniform(height) => Some(height),
            _ => None,
        }
    }
}

/// A mesh's triangles, each corner looked up, a triangle naming a vertex the
/// mesh does not have left out, where Swift's index traps: the one lookup
/// the CPU's pass and the GPU's flattened walls share.
pub fn for_each_triangle(
    vertices: &[P3<WorldSpace>],
    triangles: &[[i32; 3]],
    mut body: impl FnMut(P3<WorldSpace>, P3<WorldSpace>, P3<WorldSpace>),
) {
    let get = |k: i32| usize::try_from(k).ok().and_then(|k| vertices.get(k).copied());
    for t in triangles {
        if let (Some(a), Some(b), Some(c)) = (get(t[0]), get(t[1]), get(t[2])) {
            body(a, b, c);
        }
    }
}

/// Every triangle the depth pass draws for a scene, in world space and in the
/// order Metal draws them:
///
/// - a landscape's grid (`DrawnGrid`), decimated by its step
///   (`Grid2D::decimated_extent`: node k at index min(k step, n - 1), placed
///   on the decimated domain at k over its count less one), at the drawn
///   height, tile by tile, two triangles a cell, (00, 10, 01) and
///   (10, 11, 01), a cell kept only when all four of its corners are in the
///   region, as `kv_height_vertex` cuts it; then the occluder's triangles
///   (`kv_mesh_vertex`);
/// - a solid's faces fanned into triangles (`Solid::mesh`);
/// - a parametric surface's grid, two triangles a cell, a periodic axis
///   closing onto its first sample (`kv_param_vertex`).
pub fn drawn_triangles(scene: &Scene, mut body: impl FnMut(P3<WorldSpace>, P3<WorldSpace>, P3<WorldSpace>)) {
    match scene.geometry() {
        SceneGeometry::Heightfield(h) => {
            let drawn = DrawnGrid::of(h);
            let (grid, step) = (&*drawn.grid, drawn.step);
            let ceiling = drawn.ceiling.unwrap_or(f64::INFINITY);
            let (w, hh) = (grid.width, grid.height);
            let (dw, dh, domain) = grid.decimated_extent(step);
            if dw >= 2 && dh >= 2 {
                let at = |a: usize, b: usize| {
                    let x = domain.real.lo + domain.real.length() * a as f64 / (dw - 1) as f64;
                    let y = domain.imag.lo + domain.imag.length() * b as f64 / (dh - 1) as f64;
                    let index = (b * step).min(hh - 1) * w + (a * step).min(w - 1);
                    (x, y, index)
                };
                for tile in &h.tiles {
                    let corner = |a: usize, b: usize| -> Option<P3<WorldSpace>> {
                        let (x, y, index) = at(a, b);
                        let world = tile.apply(P3::new(x, y, 0.0));
                        if !h.region.contains(P2::new(world.x(), world.y())) {
                            return None;
                        }
                        let z = swift::min(grid.values[index] as f64, ceiling);
                        Some(P3::new(world.x(), world.y(), z))
                    };
                    for b in 0..dh - 1 {
                        for a in 0..dw - 1 {
                            let (Some(c00), Some(c10), Some(c01), Some(c11)) =
                                (corner(a, b), corner(a + 1, b), corner(a, b + 1), corner(a + 1, b + 1))
                            else {
                                continue;
                            };
                            body(c00, c10, c01);
                            body(c10, c11, c01);
                        }
                    }
                }
            }
            for_each_triangle(&h.occluder.vertices, &h.occluder.triangles, body);
        }
        SceneGeometry::Solid(solid) => {
            let m = solid.mesh();
            for_each_triangle(&m.vertices, &m.triangles, body);
        }
        SceneGeometry::Parametric(s) => {
            let (nu, nv) = (s.u.cells() as i64, s.v.cells() as i64);
            for j in 0..nv {
                for i in 0..nu {
                    let (c00, c10, c01, c11) =
                        (s.position(i, j), s.position(i + 1, j), s.position(i, j + 1), s.position(i + 1, j + 1));
                    body(c00, c10, c01);
                    body(c10, c11, c01);
                }
            }
        }
    }
}

/// The depth pass of a scene into `frame` (`MetalRenderer.renderDepth`): the
/// greatest view z of `drawn_triangles` at each of the frame's samples, as a
/// float32, `-inf` where nothing is drawn. A frame with an axis of one sample
/// or of no length draws nothing, where Metal cannot make the texture.
pub fn render_depth(scene: &Scene, view: &Transform<WorldSpace, ViewSpace>, frame: DepthFrame) -> DepthImage {
    let rows = frame.rows.max(0);
    let depth = render_rows(scene, view, frame, 0, rows);
    DepthImage::new(frame, depth, f32::NEG_INFINITY)
}

/// The depth pass's coverage of `frame`, +1 where it draws and -1 where it
/// does not, rendered `strip` rows at a time: what an outline is traced from
/// (`DepthImage::outline`), without the whole frame's depth. Swift bounds its
/// pass by Metal's texture limit and stitches the tiles into one buffer; the
/// strips bound the depth held here to a strip, and every sample is the
/// whole pass's, the same arithmetic on the same frame, whatever the strip.
pub fn coverage(scene: &Scene, view: &Transform<WorldSpace, ViewSpace>, frame: DepthFrame, strip: i64) -> Vec<f32> {
    let rows = frame.rows.max(0);
    let mut out = Vec::with_capacity((rows * frame.cols.max(0)) as usize);
    let mut lo = 0;
    while lo < rows {
        let hi = (lo + strip.max(1)).min(rows);
        out.extend(
            render_rows(scene, view, frame, lo, hi).into_iter().map(|z| if z > f32::NEG_INFINITY { 1.0 } else { -1.0 }),
        );
        lo = hi;
    }
    out
}

/// Rows `first..end` of the depth pass of `frame`, row major.
fn render_rows(
    scene: &Scene,
    view: &Transform<WorldSpace, ViewSpace>,
    frame: DepthFrame,
    first: i64,
    end: i64,
) -> Vec<f32> {
    let cols = frame.cols.max(0);
    let mut depth = vec![f32::NEG_INFINITY; ((end - first).max(0) * cols) as usize];
    if !(frame.rows >= 2 && frame.cols >= 2 && frame.axis0.length() != 0.0 && frame.axis1.length() != 0.0) {
        return depth;
    }
    let rows = frame.rows;
    let row_scale = (rows - 1) as f64 / frame.axis0.length();
    let col_scale = (cols - 1) as f64 / frame.axis1.length();
    // A world point in continuous pixel units: (row, column, view z).
    let pixel = |p: P3<WorldSpace>| -> Vec3 {
        let q = view.apply(p);
        let (row, col) = frame.components(q.xy());
        Vec3::new((row - frame.axis0.lo) * row_scale, (col - frame.axis1.lo) * col_scale, q.z())
    };
    let edge = |a: Vec3, b: Vec3, r: f64, c: f64| -> f64 { (b.x - a.x) * (c - a.y) - (r - a.x) * (b.y - a.y) };
    let min3 = |a: f64, b: f64, c: f64| swift::min(swift::min(a, b), c);
    let max3 = |a: f64, b: f64, c: f64| swift::max(swift::max(a, b), c);
    drawn_triangles(scene, |a, b, c| {
        let (p0, p1, p2) = (pixel(a), pixel(b), pixel(c));
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
        let (r0, r1, c0, c1) = (r0.max(first), r1.min(end - 1), c0.max(0), c1.min(cols - 1));
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
                let z = (w0 * p0.z + w1 * p1.z + w2 * p2.z) as f32;
                let k = ((r - first) * cols + c) as usize;
                if z > depth[k] {
                    depth[k] = z;
                }
            }
        }
    });
    depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::manifest::{Interval, ObliqueProjection};
    use crate::polyline::PolylineSet;
    use crate::solid::Solid;

    /// A unit cube seen from above covers a square: the pass's coverage is
    /// the square's pixels, at the cube's top, and its outline the square's
    /// edge.
    #[test]
    fn a_cube_from_above_covers_its_square_at_its_top() {
        let cube = Solid::named("cube", 1.0, None, None).expect("a cube");
        let camera = Camera::oblique(&ObliqueProjection {
            shear: 0.0,
            x_angle: 0.0,
            z_angle: 0.0,
            flip_x: false,
            y_scale: None,
        });
        let scene = Scene::solid_scene(cube, Vec::new(), camera, crate::scene::PreviewMode::Drawing, 0.0);
        let bounds = scene.view_bounds().unwrap().unwrap();
        let pad = 0.25 * (bounds.hi.x - bounds.lo.x);
        let frame = DepthFrame::new(
            Interval::new(bounds.lo.x - pad, bounds.hi.x + pad),
            Interval::new(bounds.lo.y - pad, bounds.hi.y + pad),
            101,
            101,
        );
        let image = render_depth(&scene, &scene.camera.view, frame);
        let mut covered = 0;
        let mut worst: f64 = 0.0;
        for r in 0..frame.rows {
            for c in 0..frame.cols {
                let p = frame.coordinate(r, c);
                let inside =
                    p.x() >= bounds.lo.x && p.x() <= bounds.hi.x && p.y() >= bounds.lo.y && p.y() <= bounds.hi.y;
                let v = image.values[(r * frame.cols + c) as usize];
                assert_eq!(v > image.empty, inside, "pixel ({r}, {c}) at {p:?}");
                if inside {
                    covered += 1;
                    worst = worst.max((f64::from(v) - bounds.hi.z).abs());
                }
            }
        }
        assert!(covered > 2500, "{covered} pixels covered");
        assert!(worst < 1e-6, "the cube's top, to the float: {worst:e}");
        let outline: PolylineSet<_> = image.outline();
        assert!(outline.count() >= 1 && !outline.vertices.is_empty());
        // In strips of any height, the coverage is the whole pass's.
        let whole: Vec<f32> = image.values.iter().map(|&v| if v > image.empty { 1.0 } else { -1.0 }).collect();
        for strip in [1, 7, 100, 1000] {
            assert_eq!(coverage(&scene, &scene.camera.view, frame, strip), whole, "strips of {strip}");
        }
        assert_eq!(crate::hidden_line::outline_of_coverage(frame, whole), outline);
    }
}
