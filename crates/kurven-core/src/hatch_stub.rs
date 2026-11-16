//! Temporary scaffolding: the parts of `Sources/KurvenCore/Hatch.swift` that
//! Surface, Mesh, Face, FaceContour and Bundle call. Hatch.swift is a later
//! wave of Phase 3; until it is ported this module stands in for it, and the
//! whole file goes when `hatch.rs` lands.
//!
//! Two kinds of stand-in. The small pure helpers -- the grid's `cell` and
//! `edgeTolerance`, a wall's foot along its edge, `clip`, `replicate`,
//! `HatchContext` and `tileBoundaryCrossings` -- are ported as Swift has
//! them, so what calls them is right meanwhile. The rest answers empty:
//! `topEdge`, `tileBoundaryStretches`, `plateauContours`, `riserStretches`,
//! the hatchings and integral curves of `curves`, and `graphTrimLoops` past
//! its domain edge (an untruncated graph's loops are exactly its domain
//! edge; a truncated graph's lack their plateaus' loops until the port).

use std::sync::Arc;

use kurven_math::swift;

use crate::manifest::{Affine2, BoundaryPerimeter, LayerSource, LayerSpec, Occluder, PerimeterEdge, Region, Walls};
use crate::polyline::PolylineSet;
use crate::simd::Vec2;
use crate::spaces::{DomainSpace, P2, P3, WorldSpace};
use crate::surface::{ContourRefine, Surface, TileBoundary};

/// What the hatching needs beyond the grids: where the walls stand and
/// how far down, what the footprint is, and how the landscape is tiled.
// PORT: phase 3 stub (Hatch.swift) — replaced on merge
#[derive(Debug, Clone, PartialEq)]
pub struct HatchContext {
    pub perimeter: Option<BoundaryPerimeter>,
    pub region: Region,
    pub tiles: Vec<Affine2>,
    /// The height the walls stand on, where a contour on them starts.
    pub base: f64,
}

impl HatchContext {
    /// The context a manifest describes. The wall kinds hatch the perimeter
    /// the walls are *derived* from, referenced rather than repeated -- so a
    /// bundle whose walls are a dumped mesh has no perimeter to hatch, and
    /// says so by deriving nothing.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn new(occluder: &Occluder) -> HatchContext {
        let (perimeter, base) = match &occluder.walls {
            Walls::Perimeter { perimeter, base } => (Some(perimeter.clone()), *base),
            _ => (None, 0.0),
        };
        HatchContext { perimeter, region: occluder.region.clone(), tiles: occluder.tiles.clone(), base }
    }
}

/// `graphTrimLoops(refine:)`'s answer.
// PORT: phase 3 stub (Hatch.swift) — replaced on merge
#[derive(Debug, Clone, Default)]
pub struct GraphTrimLoops {
    pub loops: Vec<Vec<P2<DomainSpace>>>,
    pub truncation_edge: Vec<Vec<P3<WorldSpace>>>,
    pub riser_tops: Vec<(f64, Vec<P3<WorldSpace>>)>,
    pub open: usize,
}

/// One stretch of a riser where it stands, in place (`Surface.RiserStretch`).
// PORT: phase 3 stub (Hatch.swift) — replaced on merge
#[derive(Clone)]
pub struct RiserStretch {
    /// Where the riser meets the face above: the higher top.
    pub top: Vec<P3<WorldSpace>>,
    /// Where it meets the face below: the lower top, a hair onto the
    /// lower side.
    pub bottom: Vec<P3<WorldSpace>>,
    /// The line the riser stands on, from end to end: its length, and
    /// the point `t` of the way along it.
    pub length: f64,
    pub point: Arc<dyn Fn(f64) -> P2<WorldSpace> + Send + Sync>,
    /// The stretch's part of the line, and whether it begins the run.
    pub lo: f64,
    pub hi: f64,
    pub opens: bool,
    /// The riser's outward normal at a point of the line, toward the lower side.
    pub lower: Arc<dyn Fn(P2<WorldSpace>) -> Vec2 + Send + Sync>,
    /// The map back to the domain at a point of the line.
    pub back: Arc<dyn Fn(P2<WorldSpace>) -> Affine2 + Send + Sync>,
}

/// One stop along a tile boundary's riser (`tileBoundaryStretches`).
// PORT: phase 3 stub (Hatch.swift) — replaced on merge
#[derive(Debug, Clone, Copy)]
pub struct TileBoundaryStop {
    pub s: f64,
    pub at: P2<WorldSpace>,
    pub read: P2<WorldSpace>,
}

impl Surface {
    /// The grid's spacing, the larger of the two directions: what the
    /// refiner's tolerance is measured against.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn cell(&self) -> f64 {
        let h = &*self.height;
        swift::max(
            h.domain.real.length().abs() / h.width.saturating_sub(1).max(1) as f64,
            h.domain.imag.length().abs() / h.height.saturating_sub(1).max(1) as f64,
        )
    }

    /// How near the domain's edge a truncation edge's end is taken to lie
    /// on it: a billionth of a cell.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn edge_tolerance(&self) -> f64 {
        1e-9 * self.cell()
    }

    /// `Mesh::wall_curtain`'s parameterization, which is not `a + (b - a)t`
    /// in floating point: the hatch has to stand on the wall, not a rounding
    /// error away from it.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn point_on(edge: &PerimeterEdge, t: f64) -> P2<DomainSpace> {
        Surface::point_from_to(edge.start, edge.end, t)
    }

    /// The foot `t` of the way along a wall from `a` to `b`, as
    /// `a(1 - t) + bt`, but with a vertical edge's x its own.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn point_from_to(a: P2<WorldSpace>, b: P2<WorldSpace>, t: f64) -> P2<DomainSpace> {
        P2::new(if a.x() == b.x() { a.x() } else { a.x() * (1.0 - t) + b.x() * t }, a.y() * (1.0 - t) + b.y() * t)
    }

    /// Split each path at the region boundary, keeping the runs inside.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn clip(paths: Vec<Vec<P3<WorldSpace>>>, region: &Region) -> Vec<Vec<P3<WorldSpace>>> {
        if !matches!(region, Region::Inside(_)) {
            return paths;
        }
        let mut out = Vec::new();
        for path in paths {
            let mut run: Vec<P3<WorldSpace>> = Vec::new();
            for v in path {
                if region.contains(v.xy()) {
                    run.push(v);
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
        out
    }

    /// Every path once per tile.
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn replicate(paths: &[Vec<P3<WorldSpace>>], tiles: &[Affine2]) -> Vec<Vec<P3<WorldSpace>>> {
        tiles.iter().flat_map(|tile| paths.iter().map(move |p| p.iter().map(|&v| tile.apply(v)).collect())).collect()
    }

    /// Where a wall's edge, from `t = a` to `t = b`, passes from one tile's
    /// copy of the domain into another's (`tileBoundaryCrossings`).
    // PORT: phase 3 stub (Hatch.swift) — replaced on merge
    pub fn tile_boundary_crossings(
        &self,
        edge: &PerimeterEdge,
        a: f64,
        b: f64,
        inward: Vec2,
        tiles: &[Affine2],
    ) -> Vec<(f64, Affine2, Affine2)> {
        if tiles.len() <= 1 {
            return Vec::new();
        }
        let along = edge.end.v - edge.start.v;
        let length = along.length();
        if !(length > 0.0) {
            return Vec::new();
        }
        let d = self.domain();
        let (xlo, xhi) = (swift::min(d.real.lo, d.real.hi), swift::max(d.real.lo, d.real.hi));
        let (ylo, yhi) = (swift::min(d.imag.lo, d.imag.hi), swift::max(d.imag.lo, d.imag.hi));
        let pb = Surface::point_on(edge, b);
        let mut back = self.wall_tile(Surface::point_on(edge, a), a, along, inward, tiles).1;
        let last = self.wall_tile(pb, b, along, inward, tiles).1;
        let mut t = a;
        let mut out: Vec<(f64, Affine2, Affine2)> = Vec::new();
        while back != last && out.len() < tiles.len() {
            let pa = Surface::point_on(edge, t);
            let qa = back.apply(P3::new(pa.x(), pa.y(), 0.0));
            let qb = back.apply(P3::new(pb.x(), pb.y(), 0.0));
            // Where the segment leaves the rectangle; at `b` where it does
            // not, the foot there on the boundary.
            let mut s = 1.0;
            for (u, v, lo, hi) in [(qa.x(), qb.x(), xlo, xhi), (qa.y(), qb.y(), ylo, yhi)] {
                if v == u {
                    continue;
                }
                if v > hi {
                    s = swift::min(s, (hi - u) / (v - u));
                }
                if v < lo {
                    s = swift::min(s, (lo - u) / (v - u));
                }
            }
            let crossing = if s >= 1.0 { b } else { t + (b - t) * swift::max(s, 0.0) };
            let p = Surface::point_on(edge, crossing);
            let next = if crossing == b {
                last
            } else {
                self.tile_under(P2::from_vec(p.v + self.edge_tolerance() * along / length), tiles).1
            };
            if next == back {
                break;
            }
            out.push((crossing, back, next));
            back = next;
            t = crossing;
        }
        out
    }

    /// The graph's trim loops: the domain's edge and the plateaus' loops.
    /// The stub has the domain's edge alone, which is all of them for a
    /// graph with no truncation.
    pub fn graph_trim_loops(&self, _refine: Option<&ContourRefine>) -> GraphTrimLoops {
        let g = &*self.height;
        let corner0 = g.position(0, 0);
        let corner1 = g.position(g.width - 1, g.height - 1);
        let (x0, x1) = (swift::min(corner0.x(), corner1.x()), swift::max(corner0.x(), corner1.x()));
        let (y0, y1) = (swift::min(corner0.y(), corner1.y()), swift::max(corner0.y(), corner1.y()));
        // `PlateauLoops.Rectangle.corners`, counter-clockwise from (x0, y0).
        let domain_edge = vec![P2::new(x0, y0), P2::new(x1, y0), P2::new(x1, y1), P2::new(x0, y1), P2::new(x0, y0)];
        // PORT: phase 3 stub (Hatch.swift) — replaced on merge: the plateaus' loops and the truncation edge of a truncated graph (truncationExcess, PlateauLoops, bandedTruncationEdge).
        GraphTrimLoops { loops: vec![domain_edge], ..GraphTrimLoops::default() }
    }

    /// The top edge of one wall (`topEdge(of:inward:tiles:refine:)`).
    pub fn top_edge(
        &self,
        _edge: &PerimeterEdge,
        _inward: Vec2,
        _tiles: &[Affine2],
        _refine: Option<&ContourRefine>,
    ) -> Vec<P3<WorldSpace>> {
        // PORT: phase 3 stub (Hatch.swift) — replaced on merge: the wall's top edge, f's profile along the edge with its truncation crossings and minima.
        Vec::new()
    }

    /// The stretches along a tile boundary where a riser stands
    /// (`tileBoundaryStretches(_:backs:region:step:tops:excess:)`).
    pub fn tile_boundary_stretches(
        &self,
        _boundary: &TileBoundary,
        _backs: &[Affine2],
        _region: &Region,
        _step: usize,
        _tops: &dyn Fn(P2<WorldSpace>) -> Vec<f64>,
        _excess: &dyn Fn(P2<DomainSpace>) -> f64,
    ) -> Vec<Vec<TileBoundaryStop>> {
        // PORT: phase 3 stub (Hatch.swift) — replaced on merge: the stretches of a tile boundary where one copy's top stands over the other's.
        Vec::new()
    }

    /// The contours of u or v on the plateaus (`plateauContours(of:_:refine:)`).
    pub fn plateau_contours(
        &self,
        _field: crate::manifest::ContourField,
        _levels: &[f64],
        _refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        // PORT: phase 3 stub (Hatch.swift) — replaced on merge: lines of u or v on the plateaus, cut band by band where |f| leaves each band's height.
        Vec::new()
    }

    /// The risers' stretches (`riserStretches(tiles:tiled:region:refine:)`).
    pub fn riser_stretches(
        &self,
        _tiles: &[Affine2],
        _tiled: bool,
        _region: &Region,
        _refine: Option<&ContourRefine>,
    ) -> Vec<RiserStretch> {
        // PORT: phase 3 stub (Hatch.swift) — replaced on merge: the risers at the band boundaries and the tile boundaries, a stretch at a time.
        Vec::new()
    }

    /// Any described layer's curves: a contour family, a hatching, or
    /// integral curves (`curves(_:occluder:refine:)`).
    pub fn curves(
        &self,
        spec: &LayerSpec,
        occluder: &Occluder,
        refine: Option<&ContourRefine>,
    ) -> PolylineSet<WorldSpace> {
        let context = HatchContext::new(occluder);
        match &spec.source {
            LayerSource::Hatching { .. } | LayerSource::WallEdges { .. } | LayerSource::TruncationEdge { .. } => {
                // PORT: phase 3 stub (Hatch.swift) — replaced on merge: the hatchings, the wall edges and the truncation edge (`hatch(_:in:refine:)`).
                return PolylineSet::empty();
            }
            LayerSource::IntegralCurves { .. } => {
                // PORT: phase 3 stub (IntegralCurves.swift) — replaced on merge: integral curves on the graph.
                return PolylineSet::empty();
            }
            LayerSource::Contour { field, levels, restriction, faces, light } if !field.is_field_of_f() => {
                return PolylineSet::from_paths(&self.contours(
                    *field,
                    levels,
                    faces,
                    restriction,
                    &context,
                    refine,
                    *light,
                ));
            }
            _ => {}
        }
        self.derive(&spec.source, spec.height_policy, &occluder.region, &occluder.tiles, refine)
    }
}
