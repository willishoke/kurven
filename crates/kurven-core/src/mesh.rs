//! Indexed triangle meshes, and the meshes a landscape's depth pass draws
//! besides its heightfield: the port of `Sources/KurvenCore/Mesh.swift`.

use std::fmt;

use crate::manifest::{Affine2, BoundaryPerimeter, PerimeterEdge, Region, Truncation};
use crate::simd::Vec2;
use crate::spaces::{AABB, P2, P3, Transform, WorldSpace};
use crate::surface::Surface;

/// An indexed triangle mesh.
///
/// Only the *walls* are ever a mesh. The heightfield is a texture plus a step,
/// rasterized implicitly (see `Occluder`), because an explicit mesh for the
/// elliptic plate is six million triangles and for gamma's bake two hundred
/// million. Wall curtains are a few thousand, so they are values.
pub struct Mesh<S> {
    pub vertices: Vec<P3<S>>,
    pub triangles: Vec<[i32; 3]>,
}

impl<S> Clone for Mesh<S> {
    fn clone(&self) -> Self {
        Mesh { vertices: self.vertices.clone(), triangles: self.triangles.clone() }
    }
}
impl<S> PartialEq for Mesh<S> {
    fn eq(&self, o: &Self) -> bool {
        self.vertices == o.vertices && self.triangles == o.triangles
    }
}
impl<S> fmt::Debug for Mesh<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("vertices", &self.vertices.len())
            .field("triangles", &self.triangles.len())
            .finish()
    }
}

impl<S> Mesh<S> {
    pub fn new(vertices: Vec<P3<S>>, triangles: Vec<[i32; 3]>) -> Mesh<S> {
        Mesh { vertices, triangles }
    }

    pub fn empty() -> Mesh<S> {
        Mesh { vertices: Vec::new(), triangles: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    pub fn bounds(&self) -> Option<AABB<S>> {
        AABB::of(&self.vertices)
    }

    /// Concatenate, offsetting triangle indices into the combined vertex
    /// array. The offset wraps as Swift's `&+` does.
    pub fn concat(meshes: &[Mesh<S>]) -> Mesh<S> {
        let mut verts: Vec<P3<S>> = Vec::new();
        let mut tris: Vec<[i32; 3]> = Vec::new();
        for m in meshes {
            let offset = verts.len() as i32;
            verts.extend(m.vertices.iter().copied());
            tris.extend(m.triangles.iter().map(|t| t.map(|i| i.wrapping_add(offset))));
        }
        Mesh { vertices: verts, triangles: tris }
    }

    pub fn mapped<T>(&self, f: &Transform<S, T>) -> Mesh<T> {
        Mesh { vertices: self.vertices.iter().map(|&p| f.apply(p)).collect(), triangles: self.triangles.clone() }
    }
}

impl Mesh<WorldSpace> {
    /// A vertical ruled strip along a boundary polyline, from `base` up to the
    /// surface -- `occluder.wall_curtain`.
    ///
    /// Derived walls sit at the top edge the *grid* says, where the Python
    /// plate's dumped walls sit at the top edge the analytic function says.
    /// For a bundle that ships `Walls.mesh` this is unused; for
    /// `Walls.perimeter` it is the definition, and the difference between the
    /// two is the grid's interpolation error, which is a thing to measure
    /// rather than assume.
    ///
    /// The curtain stands on the footprint's side of its edge: `inward` is
    /// the edge's normal into the footprint, and where the edge stands on a
    /// band boundary, or ends on one, its top is the band's on that side
    /// (`Surface::wall_height`), not the band the boundary starts. Where the
    /// edge crosses a tile boundary between two samples and the copies' tops
    /// there differ, it has two more there, one read through the tile on
    /// each side (`Surface::tile_boundary_crossings`), so its top steps down
    /// the riser there rather than ramping across.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn wall_curtain(
        a: P2<WorldSpace>,
        b: P2<WorldSpace>,
        inward: Vec2,
        samples: usize,
        surface: &Surface,
        base: f64,
        tiles: &[Affine2],
    ) -> Mesh<WorldSpace> {
        if samples < 2 {
            return Mesh::empty();
        }
        let mut top: Vec<P3<WorldSpace>> = Vec::with_capacity(samples);
        let mut bottom: Vec<P3<WorldSpace>> = Vec::with_capacity(samples);
        let edge = PerimeterEdge { start: a, end: b, density: samples as i64 };
        let station =
            |t: f64, through: Option<Affine2>, top: &mut Vec<P3<WorldSpace>>, bottom: &mut Vec<P3<WorldSpace>>| {
                let p = Surface::point_from_to(a, b, t);
                top.push(P3::new(p.x(), p.y(), surface.wall_height(p, t, a, b, inward, tiles, through)));
                bottom.push(P3::new(p.x(), p.y(), base));
            };
        for i in 0..samples {
            let t = i as f64 / (samples - 1) as f64;
            if i > 0 {
                let previous = (i - 1) as f64 / (samples - 1) as f64;
                for (ct, near_back, far_back) in surface.tile_boundary_crossings(&edge, previous, t, inward, tiles) {
                    let p = Surface::point_from_to(a, b, ct);
                    let height = |back: Affine2| surface.wall_height(p, ct, a, b, inward, tiles, Some(back));
                    let (near, far) = (height(near_back), height(far_back));
                    if !(Surface::stands(near, far) || Surface::stands(far, near)) {
                        continue;
                    }
                    if ct != previous {
                        station(ct, Some(near_back), &mut top, &mut bottom);
                    }
                    if ct != t {
                        station(ct, Some(far_back), &mut top, &mut bottom);
                    }
                }
            }
            station(t, None, &mut top, &mut bottom);
        }
        let samples = top.len();
        let n = samples as i32;
        let mut tris: Vec<[i32; 3]> = Vec::with_capacity(2 * (samples - 1));
        for i in 0..(samples as i32 - 1) {
            tris.push([i, i + 1, i + n]);
            tris.push([i + 1, i + 1 + n, i + n]);
        }
        let mut vertices = top;
        vertices.extend(bottom);
        Mesh { vertices, triangles: tris }
    }

    /// Every edge of a boundary as a curtain.
    pub fn walls(perimeter: &BoundaryPerimeter, surface: &Surface, base: f64, tiles: &[Affine2]) -> Mesh<WorldSpace> {
        let curtains: Vec<Mesh<WorldSpace>> = (0..perimeter.edges().len())
            .map(|k| {
                let edge = perimeter.edges()[k];
                Mesh::wall_curtain(
                    edge.start,
                    edge.end,
                    perimeter.inward(k),
                    edge.density.max(0) as usize,
                    surface,
                    base,
                    tiles,
                )
            })
            .collect();
        Mesh::concat(&curtains)
    }

    /// The risers at the tile boundaries where two copies meet with different
    /// tops (`Surface::tile_boundary_risers`), as the depth pass draws the
    /// copies either side: each copy's top along the tile boundary the height
    /// its rasterized grid has there (`Surface::drawn_height`), every
    /// `step`-th sample, and between the two a vertical strip, a pair of
    /// triangles between each two stops where the riser stands
    /// (`Surface::tile_boundary_stretches`). The heightfield's instances end
    /// at the tile boundary, each at its own top, and without the strip the
    /// drop between them is a gap the depth pass sees through.
    pub fn tile_boundary_risers(
        surface: &Surface,
        step: usize,
        region: &Region,
        tiles: &[Affine2],
    ) -> Mesh<WorldSpace> {
        let mut vertices: Vec<P3<WorldSpace>> = Vec::new();
        let mut triangles: Vec<[i32; 3]> = Vec::new();
        for boundary in surface.tile_boundaries(tiles) {
            let (Some(first), Some(second)) = (tiles[boundary.first].inverse(), tiles[boundary.second].inverse())
            else {
                continue;
            };
            let backs = [first, second];
            let tops = |w: P2<WorldSpace>| -> Vec<f64> {
                backs
                    .iter()
                    .map(|back| {
                        surface.drawn_height(surface.onto_domain(back.apply(P3::new(w.x(), w.y(), 0.0)).xy()), step)
                    })
                    .collect()
            };
            let stretches = surface.tile_boundary_stretches(&boundary, &backs, region, step, &tops, &|q| {
                let drawn = surface.drawn_surface(q, step);
                drawn.interpolated - drawn.truncation_height
            });
            for stretch in stretches {
                for w in stretch.windows(2) {
                    let base = vertices.len() as i32;
                    for stop in [w[0], w[1]] {
                        let at = stop.at;
                        let z = tops(stop.read);
                        vertices.push(P3::new(at.x(), at.y(), sequence_min(&z)));
                        vertices.push(P3::new(at.x(), at.y(), sequence_max(&z)));
                    }
                    triangles.push([base, base + 2, base + 1]);
                    triangles.push([base + 1, base + 2, base + 3]);
                }
            }
        }
        Mesh { vertices, triangles }
    }

    /// The truncated surface where a grid cell straddles the truncation: the
    /// pieces the heightfield pass gets wrong.
    ///
    /// The heightfield is rasterized from heights clamped *at the grid
    /// vertices* and interpolated between them, so a cell with one corner
    /// below the truncation and another over it is drawn as one slope from the truncation
    /// down to the low corner. The corner where the wall meets the truncation is cut
    /// off, by up to the cell's whole rise -- half a world unit on a pole's
    /// graph -- and the curves sit where the *interpolated* height reaches the
    /// truncation, out over the notch: the truncation edge, the hatch across the truncation, a phase
    /// line carried up to it. Curves from behind show through the notches, and
    /// curves on the outline are tested against a wall that is not there.
    ///
    /// What a cell's samples describe is min(interpolated height, truncation): each
    /// of the cell's two triangles cut by the truncation along the line where the
    /// interpolated excess is zero, which is the line marching squares draws
    /// the truncation edge on. This mesh is those pieces, for the cells that need them and
    /// no others. It is drawn with the heightfield under the same MAX blend,
    /// and since the clamped-at-vertices triangle lies under the true pieces
    /// everywhere (min is concave), the union is the pieces exactly: nothing
    /// has to be skipped in the heightfield pass and nothing is drawn wrong
    /// twice.
    ///
    /// The grid is the rasterizer's: every `step`-th sample, a cell kept
    /// only when all four corners lie in the region, once per tile. A banded
    /// truncation is interpolated across a cell the way the pre-clamped grid is.
    pub fn truncation_edge_mesh(
        surface: &Surface,
        step: usize,
        region: &Region,
        tiles: &[Affine2],
    ) -> Mesh<WorldSpace> {
        if matches!(surface.truncation, Truncation::None) {
            return Mesh::empty();
        }
        let g = &*surface.height;
        let step = step.max(1);
        let nx = g.width.div_ceil(step);
        let ny = g.height.div_ceil(step);
        if nx < 2 || ny < 2 {
            return Mesh::empty();
        }
        let xs: Vec<f64> = (0..nx).map(|i| g.position(i * step, 0).x()).collect();
        let ys: Vec<f64> = (0..ny).map(|j| g.position(0, j * step).y()).collect();
        let col_truncation: Vec<f64> = xs.iter().map(|&x| surface.truncation_height_at(x)).collect();

        let corner = |i: usize, j: usize| -> TruncationCorner {
            TruncationCorner {
                x: xs[i],
                y: ys[j],
                h: f64::from(g.at((i * step).min(g.width - 1), (j * step).min(g.height - 1))),
                truncation_height: col_truncation[i],
            }
        };

        // The pieces of one triangle that the truncation plane cuts it into;
        // none when it lies wholly on one side, where the rasterizer is right.
        let pieces = |t: [TruncationCorner; 3]| -> Vec<[P3<WorldSpace>; 3]> {
            let over = t.iter().filter(|c| c.excess() > 0.0).count();
            let under = t.iter().filter(|c| c.excess() < 0.0).count();
            if !(over > 0 && under > 0) {
                return Vec::new();
            }
            // The lone corner is the one on its own side of the truncation; a corner
            // exactly on the plateau joins whichever side leaves one alone.
            let lone = if over == 1 {
                t.iter().position(|c| c.excess() > 0.0)
            } else {
                t.iter().position(|c| c.excess() < 0.0)
            }
            .unwrap_or(0);
            let (a, b, c) = (t[lone], t[(lone + 1) % 3], t[(lone + 2) % 3]);
            let (p, q) = (a.crossing(&b), a.crossing(&c));
            // On the flat side a point sits on the plateau; on the sloped side on
            // the interpolated height. At P and Q the two agree.
            let flat = |k: &TruncationCorner| P3::new(k.x, k.y, k.truncation_height);
            let sloped = |k: &TruncationCorner| P3::new(k.x, k.y, k.h);
            let (a_side, other_side): (Placing, Placing) =
                if a.excess() > 0.0 { (&flat, &sloped) } else { (&sloped, &flat) };
            vec![
                [a_side(&a), a_side(&p), a_side(&q)],
                [other_side(&p), other_side(&b), other_side(&c)],
                [other_side(&p), other_side(&c), other_side(&q)],
            ]
        };

        // Straddling cells are found once; they lie only along the truncation edges. Each
        // keeps its corners, for the region rule below.
        let mut cells: Vec<Straddling> = Vec::new();
        for j in 0..(ny - 1) {
            for i in 0..(nx - 1) {
                let (c00, c10) = (corner(i, j), corner(i + 1, j));
                let (c01, c11) = (corner(i, j + 1), corner(i + 1, j + 1));
                let corners = [c00, c10, c01, c11];
                if !(corners.iter().any(|c| c.excess() > 0.0) && corners.iter().any(|c| c.excess() < 0.0)) {
                    continue;
                }
                // The rasterizer's two triangles: (00, 10, 01) and (10, 11, 01).
                let mut found = pieces([c00, c10, c01]);
                found.extend(pieces([c10, c11, c01]));
                if found.is_empty() {
                    continue;
                }
                cells.push((corners.map(|c| P2::new(c.x, c.y)), found));
            }
        }
        if cells.is_empty() {
            return Mesh::empty();
        }

        let mut vertices: Vec<P3<WorldSpace>> = Vec::new();
        let mut triangles: Vec<[i32; 3]> = Vec::new();
        for tile in tiles {
            for (corners, found) in &cells {
                // The rule the heightfield pass culls by: all four corners in.
                if !corners.iter().all(|c| region.contains(tile.apply(P3::new(c.x(), c.y(), 0.0)).xy())) {
                    continue;
                }
                for piece in found {
                    let base = vertices.len() as i32;
                    vertices.extend(piece.iter().map(|&v| tile.apply(v)));
                    triangles.push([base, base + 1, base + 2]);
                }
            }
        }
        Mesh { vertices, triangles }
    }
}

/// Swift's `.min()` on a sequence of doubles: the first of the least by
/// `<`, NaN where empty (`?? .nan`).
fn sequence_min(z: &[f64]) -> f64 {
    let mut best: Option<f64> = None;
    for &v in z {
        if best.is_none_or(|b| v < b) {
            best = Some(v);
        }
    }
    best.unwrap_or(f64::NAN)
}

/// Swift's `.max()` on a sequence of doubles: the first of the greatest by
/// `<`, NaN where empty.
fn sequence_max(z: &[f64]) -> f64 {
    let mut best: Option<f64> = None;
    for &v in z {
        if best.is_none_or(|b| b < v) {
            best = Some(v);
        }
    }
    best.unwrap_or(f64::NAN)
}

/// A corner's point in space, on the plateau or on the interpolated height.
type Placing<'a> = &'a dyn Fn(&TruncationCorner) -> P3<WorldSpace>;
/// A straddling cell's four corners, and the pieces it is cut into.
type Straddling = ([P2<WorldSpace>; 4], Vec<[P3<WorldSpace>; 3]>);

/// A grid corner of a cell on a truncation edge: where it is, how high the sample
/// is, and the truncation there. Its excess over the truncation is what
/// `Mesh::truncation_edge_mesh` solves the cut on.
#[derive(Debug, Clone, Copy)]
struct TruncationCorner {
    x: f64,
    y: f64,
    h: f64,
    truncation_height: f64,
}

impl TruncationCorner {
    fn excess(&self) -> f64 {
        self.h - self.truncation_height
    }

    /// Along the edge to `other`, the point where the excess is zero.
    fn crossing(&self, other: &TruncationCorner) -> TruncationCorner {
        let t = self.excess() / (self.excess() - other.excess());
        TruncationCorner {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            h: self.h + (other.h - self.h) * t,
            truncation_height: self.truncation_height + (other.truncation_height - self.truncation_height) * t,
        }
    }
}
