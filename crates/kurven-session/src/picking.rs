//! Picking a face: the port of `Sources/KurvenApp/Picking.swift`.
//!
//! A click reads a world point off the preview's depth buffer,
//! `SceneGeometry::face_at` names the face under it, and its boundary is
//! drawn over the scene in the accent colour, through the stroke path the
//! curves take, tested as they are.
//!
//! The depth buffer is the GPU preview's (Phase 7). Until the page has one,
//! `preview_depth` stands in for `MetalRenderer.previewDepth(atPixel:)`: the
//! triangles the depth pass draws, cast at the texel's centre on the CPU in
//! doubles, where Metal rasterizes them in floats. A pick reads the point
//! only to name the face within a few pixels of it, so the two name the same
//! face but at a face's very edge.

use std::sync::Arc;

use kurven_core::face::Face;
use kurven_core::manifest::{HeightPolicy, LayerRole, LayerSource, LayerSpec, Truncation};
use kurven_core::navigation::{Navigator, Viewport};
use kurven_core::pick::{FacePick, PickedBoundary};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{ContentID, Scene, SceneGeometry};
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, Transform, ViewSpace, WorldSpace};

use crate::document::{Document, Selection};
use crate::jobs::{BoundaryStatus, Job};

/// The stroke colour of a picked face's boundary. Swift reads the system's
/// accent colour; the page has its own, and this is macOS's default blue.
pub const ACCENT_COLOR: &str = "#007aff";

impl Document {
    /// Pick the face under a point of space, the point a click reads off the
    /// preview, or clear the pick where there is none: a click off the
    /// drawing gives no point, and a point no face is near picks nothing.
    /// The boundary is made as a job; `picking` is set until the latest
    /// click's lands, and a boundary that lands after a later click or clear
    /// is dropped.
    pub fn pick(&mut self, point: Option<P3<WorldSpace>>) {
        self.pick_generation = self.pick_generation.wrapping_add(1);
        let (Some(scene), Some(navigator), Some(point)) = (self.scene.clone(), self.navigator, point) else {
            self.clear_pick("picked nothing".to_string());
            return;
        };
        let faces = self.faces_of(&scene);
        let geometry = scene.geometry();
        let tolerance = geometry.pick_tolerance(point, navigator.framing.units_per_pixel);
        let Some(pick) = geometry.face_at(point, tolerance, Some(&faces)) else {
            self.clear_pick(format!("picked nothing at {}", vector(point.v)));
            return;
        };
        self.picked = Some(pick);
        self.draw_boundary(pick, &scene, BoundaryStatus::Picked { point: [point.x(), point.y(), point.z()] });
    }

    /// Pick at a pixel of the viewport as a click there does: the depth the
    /// preview left at its texel (`preview_depth`), taken back through the
    /// camera at the texel's centre.
    pub fn pick_at_pixel(&mut self, pixel: Vec2) {
        let point = self.world_point_at_pixel(pixel);
        self.pick(point);
    }

    /// The world point under a pixel: Swift's
    /// `MetalRenderer.worldPoint(atPixel:navigator:viewport:)`, the depth
    /// read from `preview_depth`.
    pub fn world_point_at_pixel(&self, pixel: Vec2) -> Option<P3<WorldSpace>> {
        let navigator = self.navigator?;
        let scene = self.framed_scene()?;
        let texel = Navigator::texel(pixel);
        let z = preview_depth(&scene, &navigator, self.viewport, texel)?;
        Some(navigator.world_point_at_texel(texel, z, self.viewport))
    }

    /// The scene's faces, made once while its geometry is kept.
    pub(crate) fn faces_of(&mut self, scene: &Scene) -> Arc<Vec<Face>> {
        if self.pick_faces.as_ref().map(|(c, _)| *c) != Some(scene.content()) {
            self.pick_faces = Some((scene.content(), Arc::new(scene.geometry().faces())));
        }
        self.pick_faces.as_ref().map(|(_, f)| f.clone()).unwrap_or_default()
    }

    /// Make the boundary of `pick` on `scene` as a job, drawn over the scene
    /// when it lands, unless a later click or clear came first.
    fn draw_boundary(&mut self, pick: FacePick, scene: &Scene, status: BoundaryStatus) {
        let faces = self.faces_of(scene);
        self.picking = true;
        self.jobs.push_back(Job::Boundary {
            pick,
            geometry: scene.geometry().clone(),
            faces,
            generation: self.pick_generation,
            content: scene.content(),
            status,
        });
    }

    /// A boundary landed.
    pub(crate) fn bounded(
        &mut self,
        pick: FacePick,
        geometry: &SceneGeometry,
        boundary: PickedBoundary,
        generation: u64,
        content: ContentID,
        status: BoundaryStatus,
    ) {
        if generation != self.pick_generation {
            return;
        }
        self.picking = false;
        let Some(scene) = self.scene.clone() else { return };
        if scene.content() != content {
            return;
        }
        let widest = scene.layers().iter().map(|l| l.spec.width).fold(None, |m: Option<f64>, w| {
            Some(match m {
                Some(m) => kurven_math::swift::max(m, w),
                None => w,
            })
        });
        let mut spec = LayerSpec::new(
            "selection",
            LayerRole::Scaffold,
            LayerSource::File { vertices: String::new(), offsets: String::new() },
            widest.unwrap_or(1.0),
            HeightPolicy::Surface,
        );
        spec.color = ACCENT_COLOR.to_string();
        let (paths, coords) = boundary;
        let count = paths.len();
        let paths = if matches!(geometry, SceneGeometry::Parametric(_)) {
            PolylineSet::from_paths_with_coords(&paths, &coords)
        } else {
            PolylineSet::from_paths(&paths)
        };
        let layer = Layer { spec, paths };
        let mut layers = scene.layers().to_vec();
        layers.push(layer.clone());
        self.selection = Some(Selection { pick, layer, scene: scene.drawing(layers), base: scene.curves() });
        self.pick_status = Some(match status {
            BoundaryStatus::Picked { point } => {
                format!("picked {pick} at ({:.4}, {:.4}, {:.4}), {count} curves round it", point[0], point[1], point[2])
            }
            BoundaryStatus::Kept => format!("picked {pick}, {count} curves round it"),
        });
    }

    /// No pick: what a click off the drawing leaves.
    fn clear_pick(&mut self, status: String) {
        self.selection = None;
        self.picked = None;
        self.picking = false;
        self.pick_status = Some(status);
    }

    /// Carry the pick across a new scene: its boundary drawn over the new
    /// curves where the geometry is the same. Where the geometry is new -- a
    /// surface edited, a landscape restyled -- the face is kept if the new
    /// geometry has it, and its boundary made again; if not, the pick is
    /// cleared.
    pub(crate) fn keep_selection(&mut self) {
        let Some(picked) = self.picked else { return };
        let Some(scene) = self.scene.clone() else {
            self.clear_pick("picked nothing".to_string());
            return;
        };
        if let Some(selection) = &self.selection
            && scene.content() == selection.scene.content()
        {
            if scene.curves() != selection.base {
                let mut layers = scene.layers().to_vec();
                layers.push(selection.layer.clone());
                self.selection = Some(Selection {
                    pick: selection.pick,
                    layer: selection.layer.clone(),
                    scene: scene.drawing(layers),
                    base: scene.curves(),
                });
            }
            return;
        }
        self.selection = None;
        let faces = self.faces_of(&scene);
        if !faces.iter().any(|f| f.id == picked.face && picked.copy < f.copies.len()) {
            self.pick_generation = self.pick_generation.wrapping_add(1);
            self.clear_pick("the picked face is gone".to_string());
            return;
        }
        self.pick_generation = self.pick_generation.wrapping_add(1);
        self.draw_boundary(picked, &scene, BoundaryStatus::Kept);
    }
}

/// Swift's `SIMD3<Double>` description, as a status prints a point.
fn vector(v: Vec3) -> String {
    format!("SIMD3<Double>({}, {}, {})", d(v.x), d(v.y), d(v.z))
}

fn d(x: f64) -> String {
    kurven_math::format::description(x)
}

/// The view depth the preview's depth pass leaves at a texel, or None where
/// it draws nothing: the stand-in for `MetalRenderer.previewDepth(atPixel:)`
/// until the page has a depth pass. The texel's centre is its view point
/// (`Navigator::world_point_at_texel`); the depth is the front-most, the
/// greatest view z, of the triangles the pass draws there: a landscape's
/// grid, decimated by its step, at the drawn height, per tile and cut to its
/// region, with its occluder's triangles; a parametric surface's grid; a
/// solid's faces fanned into triangles (`Solid::mesh`). A sample on an edge
/// belongs to both triangles, which the depth does not mind.
pub fn preview_depth(scene: &Scene, navigator: &Navigator, viewport: Viewport, texel: [i64; 2]) -> Option<f64> {
    if texel[0] < 0 || texel[1] < 0 || texel[0] >= viewport.width || texel[1] >= viewport.height {
        return None;
    }
    let view = navigator.camera().view;
    let q = navigator.framing.view_point(Vec2::new(texel[0] as f64 + 0.5, texel[1] as f64 + 0.5), viewport);
    let mut cast = Cast { at: Vec2::new(q.x(), q.y()), depth: None };
    match scene.geometry() {
        SceneGeometry::Heightfield(h) => {
            let banded = matches!(h.surface.truncation, Truncation::RealBands { .. });
            let grid = if banded { h.surface.clamped() } else { (*h.surface.height).clone() };
            let ceiling = match &h.surface.truncation {
                Truncation::Uniform(height) if !banded => *height,
                _ => f64::INFINITY,
            };
            // The decimated grid the shader draws (`Grid2D::decimated_extent`):
            // node k at index min(k step, n - 1), placed on the decimated
            // domain at k over its count less one.
            let step = h.step.max(1);
            let (w, hh) = (grid.width, grid.height);
            let (dw, dh, domain) = grid.decimated_extent(step);
            if dw < 2 || dh < 2 {
                return None;
            }
            let at = |a: usize, b: usize| -> (Vec2, usize) {
                let d = Vec2::new(
                    domain.real.lo + domain.real.length() * a as f64 / (dw - 1) as f64,
                    domain.imag.lo + domain.imag.length() * b as f64 / (dh - 1) as f64,
                );
                let index = (b * step).min(hh - 1) * w + (a * step).min(w - 1);
                (d, index)
            };
            for tile in &h.tiles {
                // A cell survives only when all four of its corners are in
                // the region: the notch is an absence of geometry.
                let corner = |a: usize, b: usize| -> Option<Vec3> {
                    let (d, index) = at(a, b);
                    let world = tile.apply(P3::new(d.x, d.y, 0.0));
                    if !h.region.contains(P2::new(world.x(), world.y())) {
                        return None;
                    }
                    let z = kurven_math::swift::min(grid.values[index] as f64, ceiling);
                    Some(view.apply(P3::new(world.x(), world.y(), z)).v)
                };
                for b in 0..dh - 1 {
                    for a in 0..dw - 1 {
                        let (Some(c00), Some(c10), Some(c01), Some(c11)) =
                            (corner(a, b), corner(a + 1, b), corner(a, b + 1), corner(a + 1, b + 1))
                        else {
                            continue;
                        };
                        cast.triangle(c00, c10, c01);
                        cast.triangle(c10, c11, c01);
                    }
                }
            }
            cast.mesh(&h.occluder.vertices, &h.occluder.triangles, &view);
        }
        SceneGeometry::Parametric(s) => {
            let (nu, nv) = (s.u.cells() as i64, s.v.cells() as i64);
            let at = |i: i64, j: i64| view.apply(s.position(i, j)).v;
            for j in 0..nv {
                for i in 0..nu {
                    let (c00, c10, c01, c11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
                    cast.triangle(c00, c10, c01);
                    cast.triangle(c10, c11, c01);
                }
            }
        }
        SceneGeometry::Solid(solid) => {
            let mesh = solid.mesh();
            cast.mesh(&mesh.vertices, &mesh.triangles, &view);
        }
    }
    cast.depth
}

/// The front-most depth of triangles at one view point.
struct Cast {
    at: Vec2,
    depth: Option<f64>,
}

impl Cast {
    fn mesh(&mut self, vertices: &[P3<WorldSpace>], triangles: &[[i32; 3]], view: &Transform<WorldSpace, ViewSpace>) {
        let points: Vec<Vec3> = vertices.iter().map(|&v| view.apply(v).v).collect();
        for t in triangles {
            let get = |k: i32| usize::try_from(k).ok().and_then(|k| points.get(k).copied());
            if let (Some(a), Some(b), Some(c)) = (get(t[0]), get(t[1]), get(t[2])) {
                self.triangle(a, b, c);
            }
        }
    }

    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        let p = self.at;
        let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if !(area.abs() > 1e-300) {
            return;
        }
        let wa = ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y)) / area;
        let wb = ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y)) / area;
        let wc = 1.0 - wa - wb;
        const EDGE: f64 = -1e-12;
        if !(wa >= EDGE && wb >= EDGE && wc >= EDGE) {
            return;
        }
        let z = wa * a.z + wb * b.z + wc * c.z;
        if z.is_finite() && self.depth.is_none_or(|d| z > d) {
            self.depth = Some(z);
        }
    }
}
