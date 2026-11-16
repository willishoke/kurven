//! Deriving the hatching: the curves that shade a wall, a riser and a
//! plateau, and the edges of the walls, the risers and the truncation. The
//! port of `Sources/KurvenCore/Hatch.swift`.
//!
//! `tests/fixtures/hatch` pins every kind on one small grid, perimeter and
//! pair of truncations, derived from the grid alone, with no refiner, so a
//! stroke's top sits at the grid's top edge rather than f's. That difference
//! is the grid's interpolation error, half a cell's rise at most -- and a
//! whole cell's rise where a zero of f lies between two samples, since the
//! grid's chord over a pit bottoms out at the nearer sample. So with a
//! refiner in play the top edge gains a vertex at every minimum of |f|
//! between two nodes, solved on f, and the hatch stands under it
//! ([`Surface::top_edge`]).
//!
//! Why this is here at all, rather than a list of strokes in the bundle: the
//! truncation is a slider. Moving it moves every plateau, which moves every
//! plateau hatch line and every truncation edge and re-cuts every wall stroke
//! at the new top edge. Shipping the answer would make the slider a new
//! bundle; shipping the question makes it a few milliseconds of this.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;
use std::sync::Arc;

use kurven_math::{real, swift};

use crate::bisection::bisect;
use crate::contour;
use crate::grid::Grid2D;
use crate::hatching::{FaceSelector, Hatching};
use crate::manifest::{
    Affine2, BoundaryPerimeter, ContourField, DomainAxis, LayerSource, LayerSpec, Occluder, PerimeterEdge, RealBand,
    Region, Restriction, Side, Truncation, Walls,
};
use crate::polyline::PolylineSet;
use crate::scalar_field::{self, ScalarField, Trim};
use crate::simd::Vec2;
use crate::spaces::{DomainSpace, P2, P3, WorldSpace};
use crate::surface::{ContourRefine, Surface, TileBoundary};

/// What the hatching needs beyond the grids: where the walls stand and
/// how far down, what the footprint is, and how the landscape is tiled.
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
    pub fn new(occluder: &Occluder) -> HatchContext {
        let (perimeter, base) = match &occluder.walls {
            Walls::Perimeter { perimeter, base } => (Some(perimeter.clone()), *base),
            _ => (None, 0.0),
        };
        HatchContext { perimeter, region: occluder.region.clone(), tiles: occluder.tiles.clone(), base }
    }

    /// Swift's `init(perimeter:region:tiles:base:)` with its defaults: the
    /// whole region, the one identity tile, walls standing on 0.
    pub fn with_perimeter(perimeter: Option<BoundaryPerimeter>) -> HatchContext {
        HatchContext { perimeter, region: Region::Full, tiles: vec![Affine2::IDENTITY], base: 0.0 }
    }
}

/// `graphTrimLoops(refine:)`'s answer: the loops, the truncation edge drawn
/// on them, the risers' tops (each with its boundary), and how many chains
/// could not be closed.
#[derive(Debug, Clone, Default)]
pub struct GraphTrimLoops {
    pub loops: Vec<Vec<P2<DomainSpace>>>,
    pub truncation_edge: Vec<Vec<P3<WorldSpace>>>,
    pub riser_tops: Vec<(f64, Vec<P3<WorldSpace>>)>,
    pub open: usize,
}

/// One stretch of a riser where it stands, in place: what the truncation
/// edge's layer draws of it, its top and bottom edges, and what its
/// hatching and contours need (`Surface.RiserStretch`).
#[derive(Clone)]
pub struct RiserStretch {
    /// Where the riser meets the face above: the higher top.
    pub top: Vec<P3<WorldSpace>>,
    /// Where it meets the face below: the lower top, a hair onto the lower
    /// side, so that the line of sight from it starts outside the higher
    /// side's solid.
    pub bottom: Vec<P3<WorldSpace>>,
    /// The line the riser stands on, from end to end: its length, and the
    /// point `t` of the way along it. A band boundary across the domain, or
    /// a tile boundary.
    pub length: f64,
    pub point: Arc<dyn Fn(f64) -> P2<WorldSpace> + Send + Sync>,
    /// The stretch's part of the line, as fractions of the way along it,
    /// and whether it begins the run of stretches there, rather than
    /// starting where another ends.
    pub lo: f64,
    pub hi: f64,
    pub opens: bool,
    /// The riser's outward normal at a point of the line, in the ground
    /// plane: toward the lower side.
    pub lower: Arc<dyn Fn(P2<WorldSpace>) -> Vec2 + Send + Sync>,
    /// The map back to the domain at a point of the line, through the tile
    /// of the copy the riser faces there: where a restriction is read.
    pub back: Arc<dyn Fn(P2<WorldSpace>) -> Affine2 + Send + Sync>,
}

/// One stop of a stretch along a tile boundary (`tileBoundaryStretches`):
/// the distance `s` along the tile boundary, the point `at` it is, and the
/// point its tops are read at.
#[derive(Debug, Clone, Copy)]
pub struct TileBoundaryStop {
    pub s: f64,
    pub at: P2<WorldSpace>,
    pub read: P2<WorldSpace>,
}

/// A point of a wall's edge as the top edge reads it (Swift's `Station`):
/// `t` of the way along it, nil between the ends; or where it meets a
/// riser, the riser's boundary in the domain and the side of it the
/// truncation is read from; or where it crosses a tile boundary, the tile it
/// is read through on one side.
#[derive(Debug, Clone, Copy)]
struct Station {
    p: P2<DomainSpace>,
    t: Option<f64>,
    riser: Option<(f64, Side)>,
    back: Option<Affine2>,
}

impl Station {
    fn at(p: P2<DomainSpace>) -> Station {
        Station { p, t: None, riser: None, back: None }
    }
}

/// Swift's `sort { $0 < $1 }` on doubles, as a total order Rust's sort can
/// take: `<` for every number, -0 and 0 equal as Swift's `<` has them (so a
/// stable sort keeps their order), a NaN after everything.
fn order(a: f64, b: f64) -> Ordering {
    real::total_order(a + 0.0, b + 0.0)
}

/// Swift's `.max()` on a sequence of doubles: the first of the greatest by
/// `<` (a later element replaces the result only where the result is less).
fn sequence_max(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut best: Option<f64> = None;
    for v in values {
        if best.is_none_or(|b| b < v) {
            best = Some(v);
        }
    }
    best
}

/// Swift's `.min()` on a sequence of doubles: the first of the least by `<`.
fn sequence_min(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut best: Option<f64> = None;
    for v in values {
        if best.is_none_or(|b| v < b) {
            best = Some(v);
        }
    }
    best
}

/// A point as a dictionary key, as Swift's `Hashable` points compare: by
/// `==`, so -0 is 0. A point with a NaN is equal to nothing, Swift's
/// dictionary never finds it, and here it is not a key at all.
fn point_key<S>(p: P2<S>) -> Option<(u64, u64)> {
    if p.x().is_nan() || p.y().is_nan() {
        return None;
    }
    let bits = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
    Some((bits(p.x()), bits(p.y())))
}

/// A double as a dictionary key, by `==`, -0 being 0.
fn double_key(x: f64) -> u64 {
    if x == 0.0 { 0 } else { x.to_bits() }
}

fn domain_point(p: P2<WorldSpace>) -> P2<DomainSpace> {
    P2::from_vec(p.v)
}

impl Surface {
    /// Any described layer's curves: a contour family, a hatching, or
    /// integral curves.
    ///
    /// The one place that knows which kinds of description exist, so a
    /// reader, a re-derivation after an edit and a bake all produce the same
    /// strokes for the same spec.
    pub fn curves(
        &self,
        spec: &LayerSpec,
        occluder: &Occluder,
        refine: Option<&ContourRefine>,
    ) -> PolylineSet<WorldSpace> {
        let context = HatchContext::new(occluder);
        if let Some(hatched) = self.hatch(&spec.source, &context, refine) {
            return hatched;
        }
        match &spec.source {
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

    /// The landscape's hatchings and the edges of its walls and plateaus,
    /// as world-space polylines. `None` for a source this does not own, so a
    /// caller can use it as the test as well as the derivation. With a
    /// refiner, the walls take their heights from f. A plateau's hatching is
    /// derived as what it is, the contours of a coordinate
    /// (`plateau_hatching`); a wall's and a riser's keep their own routine,
    /// since their lines stand at whole gaps along each wall or riser, at
    /// levels that differ from one to the next, and on a wall's foot as its
    /// corners stand.
    pub fn hatch(
        &self,
        source: &LayerSource,
        context: &HatchContext,
        refine: Option<&ContourRefine>,
    ) -> Option<PolylineSet<WorldSpace>> {
        let paths: Vec<Vec<P3<WorldSpace>>> = match source {
            LayerSource::File { .. }
            | LayerSource::Contour { .. }
            | LayerSource::CoordinateCurves { .. }
            | LayerSource::LinearCurves { .. }
            | LayerSource::FoldCurves
            | LayerSource::Trajectory
            | LayerSource::IntegralCurves { .. }
            | LayerSource::Edges
            | LayerSource::Hatching { faces: FaceSelector::Solid, .. } => return None,
            LayerSource::Hatching { hatching, faces: FaceSelector::Graph { tiled } } => {
                // Along a coordinate, cut to the footprint already, every
                // width step's pieces in turn (`graph_hatching`).
                let paths: Vec<Vec<P3<WorldSpace>>> = self
                    .graph_hatching(hatching, *tiled, context, refine)
                    .into_iter()
                    .flat_map(|(_, paths)| paths)
                    .collect();
                return Some(PolylineSet::from_paths(&paths));
            }
            // A landscape has no parametric surface; decoding refuses it.
            LayerSource::Hatching { faces: FaceSelector::Surface, .. } => return Some(PolylineSet::empty()),
            LayerSource::Hatching { hatching, faces: FaceSelector::Walls { edges, end_lines, base, top_offset } } => {
                let Some(perimeter) = &context.perimeter else { return Some(PolylineSet::empty()) };
                self.wall_hatch_lines(
                    perimeter,
                    edges,
                    hatching.spacing,
                    *end_lines,
                    *base,
                    *top_offset,
                    &context.tiles,
                    refine,
                )
            }
            LayerSource::Hatching { hatching, faces: FaceSelector::Risers { tiled, end_lines } } => {
                // Cut to the footprint already.
                return Some(PolylineSet::from_paths(&self.riser_hatch_lines(
                    hatching.spacing,
                    *end_lines,
                    &context.tiles,
                    *tiled,
                    &context.region,
                    refine,
                )));
            }
            LayerSource::Hatching { hatching, faces: FaceSelector::Plateaus { tiled } } => {
                // Decoding holds the angle to an axis; one built in code at
                // any other draws nothing.
                let Some(axis) = DomainAxis::from_angle(hatching.angle) else { return Some(PolylineSet::empty()) };
                let LayerSource::Contour { field, levels, restriction, faces, .. } =
                    self.plateau_hatching(axis, hatching.spacing, *tiled)
                else {
                    return Some(PolylineSet::empty());
                };
                return Some(PolylineSet::from_paths(&self.contours(
                    field,
                    &levels,
                    &faces,
                    &restriction,
                    context,
                    refine,
                    None,
                )));
            }
            LayerSource::WallEdges { edges, base } => {
                let Some(perimeter) = &context.perimeter else { return Some(PolylineSet::empty()) };
                self.wall_edges(perimeter, edges, *base, &context.tiles, refine)
            }
            LayerSource::TruncationEdge { tiled } => {
                // Cut to the footprint already, the risers' tops and bottoms
                // exactly.
                return Some(PolylineSet::from_paths(&self.drawn_truncation_edge(
                    &context.tiles,
                    *tiled,
                    &context.region,
                    refine,
                )));
            }
        };
        Some(PolylineSet::from_paths(&Surface::clip(paths, &context.region)))
    }

    /// Hatching along a coordinate of the graph, by width step in increasing
    /// order, in world space (`Surface.graphHatching(_:tiled:in:refine:)`,
    /// which Swift keeps in Isophotes.swift beside `Face.coordinateHatching`).
    pub fn graph_hatching(
        &self,
        _hatching: &Hatching,
        _tiled: bool,
        _context: &HatchContext,
        _refine: Option<&ContourRefine>,
    ) -> Vec<(i64, Vec<Vec<P3<WorldSpace>>>)> {
        // PORT: phase 5 -- each copy of the graph's `Face.coordinateHatching(_:on:)` (Isophotes.swift: the coordinate's contours at `Face.hatchLevels`, cut by width step with `Face.graded` and `Solid.graded`, a half-Lambert tone read off the copy's outward normal), lifted to |f| and carried to its tile; refused, nothing drawn, until Isophotes.swift is ported.
        Vec::new()
    }
}

// MARK: - the shared rules

impl Surface {
    /// Samples along an edge of this length, hatched at this spacing.
    ///
    /// `floor(L / spacing + 0.5)` rather than `.rounded()`: Swift rounds a
    /// half away from zero and Python rounds it to even, and an edge that is
    /// an exact multiple of the spacing would otherwise be hatched
    /// differently by the two sides of the contract. Where Swift's `Int`
    /// traps, on a count that is not a number or does not fit, 2.
    pub fn stroke_count(length: f64, spacing: f64) -> usize {
        if !(spacing.is_finite() && spacing > 0.0) {
            return 2;
        }
        swift::int((length / spacing + 0.5).floor()).map_or(2, |k| k.saturating_add(1).max(2) as usize)
    }

    /// `Mesh::wall_curtain`'s parameterization, which is not `a + (b - a)t`
    /// in floating point: the hatch has to stand on the wall, not a rounding
    /// error away from it.
    pub fn point_on(edge: &PerimeterEdge, t: f64) -> P2<DomainSpace> {
        Surface::point_from_to(edge.start, edge.end, t)
    }

    /// The foot `t` of the way along a wall from `a` to `b`, as
    /// `a(1 - t) + bt`, but with a vertical edge's x its own: interpolated,
    /// a(1 - t) + at lands a rounding either side of a, and a wall standing
    /// on a band boundary would read the band on whichever side its foot
    /// fell rather than its footprint's.
    pub fn point_from_to(a: P2<WorldSpace>, b: P2<WorldSpace>, t: f64) -> P2<DomainSpace> {
        P2::new(if a.x() == b.x() { a.x() } else { a.x() * (1.0 - t) + b.x() * t }, a.y() * (1.0 - t) + b.y() * t)
    }

    /// One vertical stroke: its two ends.
    ///
    /// A vertex between them would test nothing the ends do not. The solid
    /// is everything under the surface, and a line of sight rises toward the
    /// eye at every elevation `Orbit` allows, so lowering a point lowers its
    /// line of sight into the solid: what is hidden of a vertical stroke is
    /// one piece from its bottom up, and the bake, which bisects between a
    /// hidden vertex and a kept one (`Heightfield.visibleRuns`), finds where
    /// that piece ends from the two ends alone. A vertex every step of
    /// height was a vertex a cell up to |f|, which untruncated has no bound.
    ///
    /// One piece whether the line of sight leaves the footprint at once or
    /// meets terrain -- another copy, across a notch -- since the traversal
    /// forgives a point on a wall below its top edge nothing
    /// (`Heightfield.visibleCurves(_:on:view:depthTolerance:resolution:)`):
    /// forgiven the depth tolerance inside terrain the stroke was not placed
    /// on, the verdict along a stroke turned more than once.
    pub fn vertical(x: f64, y: f64, base: f64, top: f64) -> Option<Vec<P3<WorldSpace>>> {
        if !(top > base) {
            return None;
        }
        Some(vec![P3::new(x, y, base), P3::new(x, y, top)])
    }

    /// Split each path at the region boundary, keeping the runs inside --
    /// the rule the contours already follow, and for the same reason:
    /// keeping the survivors as one polyline welds a stroke's far side to
    /// its near side across ground the region removed.
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
    pub fn replicate(paths: &[Vec<P3<WorldSpace>>], tiles: &[Affine2]) -> Vec<Vec<P3<WorldSpace>>> {
        tiles.iter().flat_map(|tile| paths.iter().map(move |p| p.iter().map(|&v| tile.apply(v)).collect())).collect()
    }
}

// MARK: - cut faces

impl Surface {
    /// The top edge of one wall: the surface's profile along the edge,
    /// sampled once per cell at the grid's nodes and truncated, with a
    /// vertex wherever it reaches the truncation between two nodes and, with
    /// a refiner, wherever it has a minimum between two nodes.
    ///
    /// Without a refiner the truncation crossing is solved on the grid's
    /// heights linearly, as the truncation edge is contoured (as the Python
    /// original solved it, operation for operation), so the top edge meets
    /// the truncation edge where the truncation edge ends rather than a
    /// sample later, and everything else is the grid's.
    ///
    /// With one, the top edge is f's own profile along the edge: the nodes
    /// at f's height, the truncation crossing bisected on f along the edge
    /// (the truncation edge, refined, ends on the same bisection:
    /// `truncation_edge`), a vertex at each minimum of |f| between two nodes
    /// (`with_minima`: the grid has no sample at a zero of f, so its chord
    /// across a pit bottoms out at the nearer node, a whole cell's rise above
    /// the floor the phase lines fan down to), and between any two of these
    /// the profile subdivided until no chord departs from f by more than the
    /// refiner's tolerance (`subdivided_profile`). The rings are placed on f,
    /// the fold curves are traced on f and the curves are tested against f,
    /// so the top edge has to be f's too, or the rings end on a fold the top
    /// edge parts from.
    ///
    /// The top edge reads the truncation as the curtain does (`wall_foot`):
    /// at each point's own x, through the tile under it, from the
    /// footprint's side, `inward` the edge's normal into it
    /// (`Surface::footprint_side`). A wall standing on a band boundary reads
    /// the band its footprint is in, all along, and a wall ending on one
    /// reads its own band at the end.
    pub fn top_edge(
        &self,
        edge: &PerimeterEdge,
        inward: Vec2,
        tiles: &[Affine2],
        refine: Option<&ContourRefine>,
    ) -> Vec<P3<WorldSpace>> {
        let n = edge.density.max(2) as usize;
        let magnitude_of_f = refine.and_then(|r| r.magnitude.as_ref());
        let along = edge.end.v - edge.start.v;
        // |f| over a station and the truncation there, both read through
        // the tile under it, or the station's own.
        let read = |station: &Station| -> (f64, f64) {
            let (point, foot_truncation) =
                self.wall_foot(station.p, station.t.unwrap_or(0.5), along, inward, tiles, station.back);
            let truncation_height =
                station.riser.map_or(foot_truncation, |(x, side)| self.truncation_height_at_from(x, side));
            (magnitude_of_f.map_or_else(|| self.magnitude_at(point), |m| m(point)), truncation_height)
        };
        // The surface over a point between the stations, truncated.
        let profile = |p: P2<DomainSpace>| -> f64 {
            let (u, truncation_height) = read(&Station::at(p));
            swift::min(u, truncation_height)
        };
        let mut top_edge: Vec<P3<WorldSpace>> = Vec::with_capacity(n);
        // The nodes along the edge; with the function, where the edge meets a
        // riser between two nodes, two vertices there as well, one either
        // side of the boundary at its own band's height -- the boundary
        // itself belongs to the band beyond it, so the near side is a
        // rounding step short of it.
        let mut stations: Vec<Station> = (0..n)
            .map(|k| {
                let t = k as f64 / (n - 1) as f64;
                Station { p: Surface::point_on(edge, t), t: Some(t), riser: None, back: None }
            })
            .collect();
        // Where the edge crosses a tile boundary between two nodes and the
        // copies' tops there differ (`stands`), two stations at the
        // crossing, each read through the tile on its side
        // (`tile_boundary_crossings`): the top steps there, down a riser, as
        // at a band boundary. One at an end of the segment, which is read
        // through the other already. Where the tops agree, none: the top runs
        // on across, as it would on one copy.
        if tiles.len() > 1 {
            let mut with_crossings: Vec<Station> = Vec::new();
            for pair in stations.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                with_crossings.push(a);
                for (ct, near_back, far_back) in
                    self.tile_boundary_crossings(edge, a.t.unwrap_or(0.0), b.t.unwrap_or(1.0), inward, tiles)
                {
                    let p = Surface::point_on(edge, ct);
                    let near = Station { p, t: Some(ct), riser: None, back: Some(near_back) };
                    let far = Station { p, t: Some(ct), riser: None, back: Some(far_back) };
                    let (zn, zf) = (read(&near), read(&far));
                    let (tn, tf) = (swift::min(zn.0, zn.1), swift::min(zf.0, zf.1));
                    if !(Surface::stands(tn, tf) || Surface::stands(tf, tn)) {
                        continue;
                    }
                    if Some(ct) != a.t {
                        with_crossings.push(near);
                    }
                    if Some(ct) != b.t {
                        with_crossings.push(far);
                    }
                }
            }
            with_crossings.push(stations[stations.len() - 1]);
            stations = with_crossings;
        }
        if magnitude_of_f.is_some()
            && let Truncation::RealBands { bands, .. } = &self.truncation
        {
            // The boundaries inside the domain, a band that starts on its
            // right edge starting beyond it, that the edge crosses in the
            // domain, through the tile under the crossing: one it ends on it
            // meets in its own band (`Surface::footprint_side`).
            let real = self.domain().real;
            let boundaries: Vec<f64> = bands
                .iter()
                .map(|b| b.below)
                .filter(|&b| b > swift::min(real.lo, real.hi) && b < swift::max(real.lo, real.hi))
                .collect();
            let mut with_risers: Vec<Station> = Vec::new();
            for pair in stations.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                with_risers.push(a);
                let backs: [Affine2; 2] = [a, b]
                    .map(|s| s.back.unwrap_or_else(|| self.wall_tile(s.p, s.t.unwrap_or(0.5), along, inward, tiles).1));
                let mut met: Vec<(f64, Station, Station)> = Vec::new();
                let distinct: Vec<Affine2> =
                    if backs[1] != backs[0] { vec![backs[0], backs[1]] } else { vec![backs[0]] };
                for back in distinct {
                    let Some(forward) = back.inverse() else { continue };
                    let qa = back.apply(P3::new(a.p.x(), a.p.y(), 0.0));
                    let qb = back.apply(P3::new(b.p.x(), b.p.y(), 0.0));
                    // An end of the edge on a boundary, a rounding off it
                    // where it came back through a tile.
                    let xa = if a.t == Some(0.0) { self.wall_foot_x(qa.x(), 0.0, along, back) } else { qa.x() };
                    let xb = if b.t == Some(1.0) { self.wall_foot_x(qb.x(), 1.0, along, back) } else { qb.x() };
                    for &x in &boundaries {
                        if !(((xa < x && xb >= x) || (xb < x && xa >= x))
                            && xb != xa
                            && !(a.t == Some(0.0) && xa == x)
                            && !(b.t == Some(1.0) && xb == x))
                        {
                            continue;
                        }
                        let s = (x - xa) / (xb - xa);
                        let y = qa.y() + (qb.y() - qa.y()) * s;
                        // Met where this tile is the one under it.
                        let crossing = forward.apply(P3::new(x, y, 0.0));
                        if !(backs[0] == backs[1]
                            || self.wall_tile(P2::new(crossing.x(), crossing.y()), 0.5, along, inward, tiles).1 == back)
                            || met.iter().any(|m| m.0 == s)
                        {
                            continue;
                        }
                        // Carried forward, on the edge: a vertical or a
                        // horizontal edge keeps its own x or y, which a
                        // turned tile carries a rounding off.
                        let on_edge = |q: P3<WorldSpace>| -> P2<DomainSpace> {
                            P2::new(
                                if a.p.x() == b.p.x() { a.p.x() } else { q.x() },
                                if a.p.y() == b.p.y() { a.p.y() } else { q.y() },
                            )
                        };
                        let near = forward.apply(P3::new(if xa < x { x.next_down() } else { x }, y, 0.0));
                        let far = forward.apply(P3::new(if xa < x { x } else { x.next_down() }, y, 0.0));
                        let (near_side, far_side) =
                            if xa < x { (Side::Left, Side::Right) } else { (Side::Right, Side::Left) };
                        met.push((
                            s,
                            Station { p: on_edge(near), t: None, riser: Some((x, near_side)), back: None },
                            Station { p: on_edge(far), t: None, riser: Some((x, far_side)), back: None },
                        ));
                    }
                }
                met.sort_by(|m, o| order(m.0, o.0));
                for (_, near, far) in met {
                    with_risers.push(near);
                    with_risers.push(far);
                }
            }
            with_risers.push(stations[stations.len() - 1]);
            stations = with_risers;
        }
        // The station before: its point, its excess, and the riser it is beside.
        type Previous = (P2<DomainSpace>, f64, Option<(f64, Side)>);
        let mut previous: Option<Previous> = None;
        for station in &stations {
            let p = station.p;
            let (u, truncation_height) = read(station);
            let excess = u - truncation_height;
            // No crossing between the two stations either side of one riser:
            // they stand a rounding step apart, and what changes between them
            // is the truncation, not |f|, so the top steps there from one to
            // the other and reaches neither band's truncation between.
            let across_riser = match (previous.and_then(|a| a.2), station.riser) {
                (Some(pr), Some(sr)) => pr.0 == sr.0 && pr.1 != sr.1,
                _ => false,
            };
            if let Some((ap, a_excess, _)) = previous
                && excess.is_finite()
                && a_excess.is_finite()
                && (ap.x() != p.x() || ap.y() != p.y())
                && !across_riser
            {
                let here = (p, excess);
                let there = (ap, a_excess);
                let mut under: Option<(P2<DomainSpace>, f64)> = None;
                let mut over: Option<(P2<DomainSpace>, f64)> = None;
                if a_excess <= 0.0 && excess > 0.0 {
                    under = Some(there);
                    over = Some(here);
                }
                if excess <= 0.0 && a_excess > 0.0 {
                    under = Some(here);
                    over = Some(there);
                }
                if let (Some(under), Some(over)) = (under, over) {
                    if magnitude_of_f.is_some() {
                        if let Some(q) = self.truncation_crossing_between(under.0, under.1, over.0, over.1, |q| {
                            let r = read(&Station::at(q));
                            r.0 - r.1
                        }) {
                            top_edge.push(P3::new(q.x(), q.y(), read(&Station::at(q)).1));
                        }
                    } else {
                        let t = under.1 / (under.1 - over.1);
                        if t > 0.0 && t < 1.0 {
                            let x = under.0.x() + (over.0.x() - under.0.x()) * t;
                            let y = under.0.y() + (over.0.y() - under.0.y()) * t;
                            top_edge.push(P3::new(x, y, read(&Station::at(P2::new(x, y))).1));
                        }
                    }
                }
            }
            top_edge.push(P3::new(p.x(), p.y(), swift::min(u, truncation_height)));
            previous = Some((p, excess, station.riser));
        }
        let Some(refine) = refine else { return top_edge };
        if refine.magnitude.is_none() {
            return top_edge;
        }
        self.subdivided_profile(&self.with_minima(&top_edge, &profile), &profile, refine.tolerance, 5)
    }

    /// Where a wall's edge crosses a tile boundary between `a` and `b` of
    /// the way along it, in order, each with the tiles either side: the tile
    /// under the foot at `a` (`wall_tile`), its copy of the domain left where
    /// the edge carried back through it leaves the rectangle, and the tile
    /// under the edge just beyond, until the one under the foot at `b`. A
    /// foot on a tile boundary read through the tile on its footprint's side
    /// meets it at that foot. Nothing for one tile, or a stretch under one.
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

    /// The grid's spacing, the larger of the two directions: what the
    /// refiner's tolerance is measured against.
    pub fn cell(&self) -> f64 {
        let h = &*self.height;
        swift::max(
            h.domain.real.length().abs() / h.width.saturating_sub(1).max(1) as f64,
            h.domain.imag.length().abs() / h.height.saturating_sub(1).max(1) as f64,
        )
    }

    /// How near the domain's edge a truncation edge's end is taken to lie
    /// on it: a billionth of a cell.
    pub fn edge_tolerance(&self) -> f64 {
        1e-9 * self.cell()
    }

    /// Where |f| crosses the truncation on the straight segment from `a` to
    /// `b`, given the excess over the truncation at each end: bisection on f
    /// to the last bit, `None` when both ends lie on one side. One routine
    /// for the top edge and the truncation edge's ends, so the two meet to
    /// rounding. `excess` is |f| less the truncation at a point of the
    /// segment, as the caller reads it: the top edge through the tile under
    /// the point, from the footprint's side. (Swift's
    /// `truncationCrossing(from:_:to:_:excess:)`; `truncation_crossing` is
    /// Surface.swift's on a lifted segment.)
    pub fn truncation_crossing_between(
        &self,
        a: P2<DomainSpace>,
        ea: f64,
        b: P2<DomainSpace>,
        eb: f64,
        excess: impl Fn(P2<DomainSpace>) -> f64,
    ) -> Option<P2<DomainSpace>> {
        if !(ea.is_finite() && eb.is_finite() && ((ea > 0.0) != (eb > 0.0))) {
            return None;
        }
        let (mut lo, mut hi) = (a.v, b.v);
        let mut elo = ea;
        for _ in 0..60 {
            let m = 0.5 * (lo + hi);
            let em = excess(P2::from_vec(m));
            if !em.is_finite() {
                return None;
            }
            if (em > 0.0) == (elo > 0.0) {
                lo = m;
                elo = em;
            } else {
                hi = m;
            }
        }
        Some(P2::from_vec(0.5 * (lo + hi)))
    }

    /// The top edge with f's profile between its vertices: each chord's
    /// midpoint is taken onto the profile where the chord departs from it
    /// by more than `tolerance` cells, and the halves likewise, `max_depth`
    /// deep: five for a wall's top edge. The top edge is a function of the
    /// distance along its straight edge, so the vertices keep their order.
    /// `profile` is the truncated surface over a point of the edge, as the
    /// top edge reads it.
    pub fn subdivided_profile(
        &self,
        top_edge: &[P3<WorldSpace>],
        profile: &dyn Fn(P2<DomainSpace>) -> f64,
        tolerance: f64,
        max_depth: usize,
    ) -> Vec<P3<WorldSpace>> {
        if top_edge.len() < 2 {
            return top_edge.to_vec();
        }
        let cell = self.cell();
        #[allow(clippy::too_many_arguments)]
        fn subdivide(
            a: P3<WorldSpace>,
            b: P3<WorldSpace>,
            depth: usize,
            out: &mut Vec<P3<WorldSpace>>,
            profile: &dyn Fn(P2<DomainSpace>) -> f64,
            tolerance: f64,
            max_depth: usize,
            cell: f64,
        ) {
            if !(depth < max_depth && a.z().is_finite() && b.z().is_finite()) {
                return;
            }
            // The two vertices either side of a riser stand at one point.
            if !((a.x() - b.x()).abs() + (a.y() - b.y()).abs() > 1e-9 * cell) {
                return;
            }
            let mid = P2::<DomainSpace>::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
            let z = profile(mid);
            if !(z.is_finite() && (z - 0.5 * (a.z() + b.z())).abs() > tolerance * cell) {
                return;
            }
            let v = P3::<WorldSpace>::new(mid.x(), mid.y(), z);
            subdivide(a, v, depth + 1, out, profile, tolerance, max_depth, cell);
            out.push(v);
            subdivide(v, b, depth + 1, out, profile, tolerance, max_depth, cell);
        }
        let mut out: Vec<P3<WorldSpace>> = vec![top_edge[0]];
        for pair in top_edge.windows(2) {
            subdivide(pair[0], pair[1], 0, &mut out, profile, tolerance, max_depth, cell);
            out.push(pair[1]);
        }
        out
    }

    /// The top edge with a vertex at every minimum of |f| that falls between
    /// two of its vertices. The edge is straight, so the top edge is a
    /// function of the distance along it and the new vertices keep that
    /// order. `profile` is the truncated surface over a point of the edge,
    /// as the top edge reads it.
    pub fn with_minima(
        &self,
        top_edge: &[P3<WorldSpace>],
        profile: &dyn Fn(P2<DomainSpace>) -> f64,
    ) -> Vec<P3<WorldSpace>> {
        if top_edge.len() < 3 {
            return top_edge.to_vec();
        }
        // A vertex's point on the edge.
        let dom = |v: P3<WorldSpace>| -> P2<DomainSpace> { P2::new(v.x(), v.y()) };
        let between = |a: P2<DomainSpace>, b: P2<DomainSpace>, t: f64| -> P2<DomainSpace> {
            P2::new(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)
        };
        let apart = |a: P2<DomainSpace>, b: P2<DomainSpace>| -> f64 {
            ((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y())).sqrt()
        };
        let mut out: Vec<P3<WorldSpace>> = vec![top_edge[0]];
        for i in 1..(top_edge.len() - 1) {
            let v = top_edge[i];
            out.push(v);
            if !(v.z().is_finite()
                && v.z() <= top_edge[i - 1].z()
                && v.z() <= top_edge[i + 1].z()
                && (v.z() < top_edge[i - 1].z() || v.z() < top_edge[i + 1].z()))
            {
                continue;
            }
            let (a, b) = (dom(top_edge[i - 1]), dom(top_edge[i + 1]));
            let (mut lo, mut hi) = (0.0, 1.0);
            let phi = (5.0f64.sqrt() - 1.0) / 2.0;
            let mut c = hi - phi * (hi - lo);
            let mut d = lo + phi * (hi - lo);
            let mut fc = profile(between(a, b, c));
            let mut fd = profile(between(a, b, d));
            for _ in 0..90 {
                if fc < fd {
                    hi = d;
                    d = c;
                    fd = fc;
                    c = hi - phi * (hi - lo);
                    fc = profile(between(a, b, c));
                } else {
                    lo = c;
                    c = d;
                    fc = fd;
                    d = lo + phi * (hi - lo);
                    fd = profile(between(a, b, d));
                }
                if hi - lo < 1e-15 {
                    break;
                }
            }
            let q = between(a, b, (lo + hi) / 2.0);
            let z = profile(q);
            if !(z.is_finite()
                && z < v.z() - 1e-12 * swift::max(v.z().abs(), 1.0)
                && apart(q, dom(v)) > 1e-12 * apart(a, b))
            {
                continue;
            }
            // Two nodes of equal height either side of a dip both lead here;
            // the vertex is added once.
            if out[out.len().saturating_sub(2)..].iter().any(|&o| apart(dom(o), q) <= 1e-9 * apart(a, b)) {
                continue;
            }
            let vertex = P3::<WorldSpace>::new(q.x(), q.y(), z);
            // Before the node or after it, by where it lies along the edge.
            let before = (q.x() - a.x()) * (b.x() - a.x()) + (q.y() - a.y()) * (b.y() - a.y())
                < (v.x() - a.x()) * (b.x() - a.x()) + (v.y() - a.y()) * (b.y() - a.y());
            if before {
                let at = out.len() - 1;
                out.insert(at, vertex);
            } else {
                out.push(vertex);
            }
        }
        out.push(top_edge[top_edge.len() - 1]);
        out
    }

    /// The top edge's height at a point of its edge, by linear interpolation
    /// between its vertices: what a hatch stroke rises to. Without a refiner
    /// that is the grid's own interpolation along the edge, which
    /// `height_at_tiles` computes bit for bit as `kurven.hatch` does and the
    /// fixtures hold it to; with one, the top edge has vertices the grid does
    /// not, and the strokes stand under them.
    pub fn top_edge_height(&self, top_edge: &[P3<WorldSpace>], p: P2<DomainSpace>) -> Option<f64> {
        if top_edge.len() < 2 {
            return None;
        }
        let (a0, b0) = (top_edge[0], top_edge[top_edge.len() - 1]);
        let (dx, dy) = (b0.x() - a0.x(), b0.y() - a0.y());
        let length = dx * dx + dy * dy;
        if !(length > 0.0) {
            return None;
        }
        let along = |x: f64, y: f64| -> f64 { ((x - a0.x()) * dx + (y - a0.y()) * dy) / length };
        let s = along(p.x(), p.y());
        for pair in top_edge.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (sa, sb) = (along(a.x(), a.y()), along(b.x(), b.y()));
            if !(sa <= s && s <= sb) {
                continue;
            }
            let t = if sb > sa { (s - sa) / (sb - sa) } else { 0.0 };
            return Some(a.z() + (b.z() - a.z()) * t);
        }
        None
    }

    /// Vertical strokes on the walls along `edges`, at whole gaps along each
    /// wall, from `base` up to the top edge plus `top_offset`.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn wall_hatch_lines(
        &self,
        perimeter: &BoundaryPerimeter,
        edges: &[i64],
        spacing: f64,
        end_lines: bool,
        base: f64,
        top_offset: f64,
        tiles: &[Affine2],
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let mut out = Vec::new();
        for &index in edges {
            if !(index >= 0 && (index as usize) < perimeter.edges().len()) {
                continue;
            }
            let index = index as usize;
            let edge = perimeter.edges()[index];
            let inward = perimeter.inward(index);
            let length = (edge.end.v - edge.start.v).length();
            let n = Surface::stroke_count(length, spacing);
            let range = if end_lines { 0..n } else { 1..(n - 1).max(1) };
            let top_edge = if refine.is_some_and(|r| r.magnitude.is_some()) {
                Some(self.top_edge(&edge, inward, tiles, refine))
            } else {
                None
            };
            for k in range {
                let t = k as f64 / (n - 1).max(1) as f64;
                let p = Surface::point_on(&edge, t);
                let top = top_edge
                    .as_ref()
                    .and_then(|e| self.top_edge_height(e, p))
                    .unwrap_or_else(|| self.wall_height(p, t, edge.start, edge.end, inward, tiles, None))
                    + top_offset;
                if let Some(stroke) = Surface::vertical(p.x(), p.y(), base, top) {
                    out.push(stroke);
                }
            }
        }
        out
    }

    /// The top and bottom edges of every wall, and a vertical edge at every
    /// corner.
    ///
    /// A corner's vertical edge is where two walls meet, up to the lower of
    /// their tops. They differ only on a band boundary, where one wall
    /// stands on it, reading its footprint's band, and the other leaves it
    /// in its own band, and a riser ends at the corner, in the plane of the
    /// wall on the boundary: the edge rises to the top of the wall that
    /// leaves, past the other's between it and the riser, or short of it,
    /// where the riser and the wall on the boundary are one plane.
    pub fn wall_edges(
        &self,
        perimeter: &BoundaryPerimeter,
        edges: &[i64],
        base: f64,
        tiles: &[Affine2],
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let mut out = Vec::new();
        let mut corners: Vec<P2<WorldSpace>> = Vec::new();
        for &index in edges {
            if !(index >= 0 && (index as usize) < perimeter.edges().len()) {
                continue;
            }
            let index = index as usize;
            let edge = perimeter.edges()[index];
            let n = edge.density.max(2) as usize;
            out.push(self.top_edge(&edge, perimeter.inward(index), tiles, refine));
            out.push(
                (0..n)
                    .map(|k| {
                        let p = Surface::point_on(&edge, k as f64 / (n - 1) as f64);
                        P3::new(p.x(), p.y(), base)
                    })
                    .collect(),
            );
            // Deduplicated by exact coordinate, so a closed traversal stands
            // one vertical edge at each corner rather than two.
            for corner in [edge.start, edge.end] {
                if !corners.contains(&corner) {
                    corners.push(corner);
                }
            }
        }
        for corner in corners {
            // Every wall ending at the corner, and its top there; those that
            // are not parallel to the imaginary axis in the domain, through
            // the tile the wall reads its foot there through, where any is
            // not.
            let foot = P2::<DomainSpace>::new(corner.x(), corner.y());
            let mut tops: Vec<(f64, bool)> = Vec::new();
            for (k, e) in perimeter.edges().iter().enumerate() {
                let along = e.end.v - e.start.v;
                for (end, t) in [(e.start, 0.0), (e.end, 1.0)] {
                    if end != corner {
                        continue;
                    }
                    let back = self.wall_tile(foot, t, along, perimeter.inward(k), tiles).1;
                    tops.push((
                        self.wall_height(foot, t, e.start, e.end, perimeter.inward(k), tiles, None),
                        !Surface::is_parallel_to_imaginary_axis(along, back),
                    ));
                }
            }
            let leaving: Vec<f64> = tops.iter().filter(|t| t.1).map(|t| t.0).collect();
            let top = if leaving.len() == 1 { sequence_max(leaving) } else { sequence_max(tops.iter().map(|t| t.0)) }
                .unwrap_or_else(|| self.height_at_tiles(foot, tiles));
            if let Some(vertical_edge) = Surface::vertical(corner.x(), corner.y(), base, top) {
                out.push(vertical_edge);
            }
        }
        out
    }
}

// MARK: - plateaus

impl Surface {
    /// A hatching on the plateaus, as what it is: the contours of the
    /// coordinate across its lines, Im z for lines along the real axis and
    /// Re z for lines along the imaginary, on the plateaus.
    ///
    /// Its levels divide the domain across the lines the way
    /// `wall_hatch_lines` divides an edge: the spacing asks for a count, and
    /// the lines are spread to fit it, the two on the domain's edges left
    /// out. The hatching then meets the plateau's edges with whole gaps, and
    /// each line stands where the walls that run across it stand theirs, so
    /// it turns the corner down the wall instead of missing it.
    pub fn plateau_hatching(&self, axis: DomainAxis, spacing: f64, tiled: bool) -> LayerSource {
        let across = if axis == DomainAxis::Real { self.height.domain.imag } else { self.height.domain.real };
        let n = Surface::stroke_count(across.hi - across.lo, spacing);
        let levels: Vec<f64> = if spacing.is_finite() && spacing > 0.0 {
            (1..(n - 1).max(1)).map(|k| across.lo + (across.hi - across.lo) * k as f64 / (n - 1) as f64).collect()
        } else {
            Vec::new()
        };
        LayerSource::Contour {
            field: if axis == DomainAxis::Real { ContourField::V } else { ContourField::U },
            levels,
            restriction: Restriction::All,
            faces: FaceSelector::Plateaus { tiled },
            light: None,
        }
    }

    /// The contours of u or v on the plateaus: lines along the other axis,
    /// placed by the plain routine (`scalar_field::levels`) with a vertex
    /// wherever they cross a grid line, on the level exactly, and each cut
    /// where it leaves a plateau. The height is constant on a plateau, and
    /// has none.
    ///
    /// A line is cut band by band, each band's plateau at its own height,
    /// with a vertex where it enters and leaves the band, and kept where |f|
    /// reaches the band's height. With the function at hand it is ruled on f
    /// and ends where |f| reaches the height along it, bisected: the grid's
    /// chords lie above f on a convex graph, so its truncation region reaches
    /// past the truncation edge drawn on f by a fraction of a cell, and a line
    /// ruled on it poked out past the truncation edge. Without, the end is
    /// solved linearly on the grid, as the truncation edge is contoured.
    pub fn plateau_contours(
        &self,
        field: ContourField,
        levels: &[f64],
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        if !(field == ContourField::U || field == ContourField::V) {
            return Vec::new();
        }
        let Some(rectangle) = self.parameter_rectangle() else { return Vec::new() };
        // A line of v runs along the real axis, one of u along the imaginary.
        let axis = if field == ContourField::V { DomainAxis::Real } else { DomainAxis::Imag };
        let grid = &*self.height;
        let along_interval = if axis == DomainAxis::Real { grid.domain.real } else { grid.domain.imag };
        let coordinate = ScalarField::new(None, move |p| if field == ContourField::U { p.x() } else { p.y() });
        let lines = scalar_field::levels(&coordinate, levels, rectangle, &Trim::default());
        let magnitude = refine.and_then(|r| r.magnitude.as_ref());
        let mut out = Vec::new();
        for (_, paths) in &lines {
            for path in paths {
                let Some(first) = path.first() else { continue };
                // The coordinate across the line, and its vertices in order
                // along it.
                let c = if axis == DomainAxis::Real { first.y() } else { first.x() };
                let mut samples: Vec<f64> =
                    path.iter().map(|p| if axis == DomainAxis::Real { p.x() } else { p.y() }).collect();
                samples.sort_by(|a, b| order(*a, *b));
                // The truncation varies with Re and with nothing else: a line
                // along the real axis crosses the bands and is solved piece by
                // piece, and one along the imaginary axis stays in a single
                // band for its length.
                let intervals: Vec<(f64, f64)> = if axis == DomainAxis::Real {
                    self.truncation.bands(along_interval.lo, along_interval.hi).iter().map(|s| (s.lo, s.hi)).collect()
                } else {
                    vec![(along_interval.lo, along_interval.hi)]
                };
                for (a, b) in intervals {
                    let truncation_height = self.truncation_height_at(if axis == DomainAxis::Real { a } else { c });
                    if !truncation_height.is_finite() {
                        continue;
                    }
                    let mut coord = vec![a];
                    coord.extend(samples.iter().copied().filter(|&s| s > a && s < b));
                    coord.push(b);
                    let excess_at = |t: f64| -> f64 {
                        let p = if axis == DomainAxis::Real { P2::new(t, c) } else { P2::new(c, t) };
                        if let Some(f) = magnitude {
                            let u = f(p);
                            return (if u.is_nan() { f64::INFINITY } else { u }) - truncation_height;
                        }
                        self.magnitude_at(p) - truncation_height
                    };
                    let excess: Vec<f64> = coord.iter().map(|&t| excess_at(t)).collect();
                    let runs = if magnitude.is_none() {
                        Surface::runs_at_or_above(&coord, &excess, None)
                    } else {
                        Surface::runs_at_or_above(
                            &coord,
                            &excess,
                            Some(&|lo, hi| Surface::bisected(lo, hi, &excess_at)),
                        )
                    };
                    for run in runs {
                        out.push(
                            run.iter()
                                .map(|&t| {
                                    if axis == DomainAxis::Real {
                                        P3::new(t, c, truncation_height)
                                    } else {
                                        P3::new(c, t, truncation_height)
                                    }
                                })
                                .collect(),
                        );
                    }
                }
            }
        }
        out
    }

    /// The stretches where `excess >= 0`, with their ends solved linearly.
    ///
    /// `t = e0 / (e0 - e1)` is where the same linear interpolation puts the
    /// truncation edge contour, so a stroke ends exactly on the truncation
    /// edge rather than at the last sample inside it.
    pub fn runs_at_or_above(
        coord: &[f64],
        excess: &[f64],
        crossing: Option<&dyn Fn(f64, f64) -> f64>,
    ) -> Vec<Vec<f64>> {
        let solve = |i0: usize, i1: usize| -> f64 {
            if let Some(crossing) = crossing {
                return crossing(coord[i0], coord[i1]);
            }
            let (e0, e1) = (excess[i0], excess[i1]);
            let t = if e0 != e1 { e0 / (e0 - e1) } else { 0.0 };
            coord[i0] + t * (coord[i1] - coord[i0])
        };
        let mut runs = Vec::new();
        let mut start: Option<usize> = None;
        for i in 0..=coord.len() {
            let inside = i < coord.len() && excess[i] >= 0.0;
            if inside && start.is_none() {
                start = Some(i);
            }
            if !inside && let Some(a) = start {
                let mut line = Vec::new();
                if a > 0 {
                    line.push(solve(a - 1, a));
                }
                line.extend_from_slice(&coord[a..i]);
                if i < coord.len() {
                    line.push(solve(i - 1, i));
                }
                if line.len() >= 2 {
                    runs.push(line);
                }
                start = None;
            }
        }
        runs
    }

    /// The coordinate between `lo` and `hi` where `f` changes sign, bisected
    /// to the last bit; `lo` when it does not.
    pub fn bisected(lo: f64, hi: f64, f: &dyn Fn(f64) -> f64) -> f64 {
        let flo = f(lo);
        if (flo < 0.0) == (f(hi) < 0.0) {
            return lo;
        }
        let (a, b) = bisect(lo, hi, |t| (f(t) < 0.0) == (flo < 0.0));
        0.5 * (a + b)
    }

    /// |f| - truncation, as a grid, in float32 -- the same subtraction the
    /// Python side performs, on the same float32 samples, so the truncation
    /// edge lands in the same place. The truncation edge is this grid's zero
    /// contour.
    pub fn truncation_excess(&self) -> Option<Grid2D<f32>> {
        let grid = &*self.height;
        let mut values = grid.values.clone();
        let mut any_finite = false;
        for x in 0..grid.width {
            let truncation_height = self.truncation_height_at(grid.position(x, 0).x()) as f32;
            if !truncation_height.is_finite() {
                for y in 0..grid.height {
                    values[y * grid.width + x] = f32::NEG_INFINITY;
                }
                continue;
            }
            any_finite = true;
            for y in 0..grid.height {
                values[y * grid.width + x] = grid.at(x, y) - truncation_height;
            }
        }
        if !any_finite {
            return None;
        }
        Some(Grid2D::new(grid.width, grid.height, grid.domain, values))
    }

    /// The truncation edge, as the graph's trim loops draw it
    /// (`graph_trim_loops`).
    pub fn truncation_edge(&self, refine: Option<&ContourRefine>) -> Vec<Vec<P3<WorldSpace>>> {
        self.graph_trim_loops(refine).truncation_edge
    }

    /// The truncation edge as its layer draws it: on every tile when
    /// `tiled`, and in place otherwise, cut to the footprint, with each
    /// riser's top and bottom edges where the riser stands
    /// (`band_riser_stretches`). Tiled, the edges of the risers at the tile
    /// boundaries follow, where those risers stand
    /// (`tile_boundary_riser_stretches`).
    ///
    /// The placed pieces are cut at their vertices, as every layer is
    /// (`clip`).
    pub fn drawn_truncation_edge(
        &self,
        tiles: &[Affine2],
        tiled: bool,
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let trim = self.graph_trim_loops(refine);
        let pieces = &trim.truncation_edge[..trim.truncation_edge.len().saturating_sub(trim.riser_tops.len())];
        let at_tile_boundaries =
            if tiled { self.tile_boundary_riser_stretches(tiles, region, refine) } else { Vec::new() };
        let copies: Vec<Option<Affine2>> = if tiled { tiles.iter().copied().map(Some).collect() } else { vec![None] };
        let mut out = Vec::new();
        for tile in copies {
            let placed = |v: P3<WorldSpace>| -> P3<WorldSpace> { tile.map_or(v, |t| t.apply(v)) };
            out.extend(Surface::clip(pieces.iter().map(|p| p.iter().map(|&v| placed(v)).collect()).collect(), region));
            for stretch in self.band_riser_stretches(&trim.riser_tops, tile, region, refine) {
                out.push(stretch.top);
                out.push(stretch.bottom);
            }
        }
        for stretch in at_tile_boundaries {
            out.push(stretch.top);
            out.push(stretch.bottom);
        }
        out
    }

    /// The risers at the band boundaries on the copy on `tile`, or in place
    /// for `None`, from their tops (`graph_trim_loops`), a stretch at a
    /// time: where each stands, where the footprint holds both sides of its
    /// boundary, read a hair either side of it where the copy stands, as
    /// `risers` reads it. Where the footprint holds one side there is no
    /// riser: a wall on the boundary rises to the top on that side, and an
    /// edge drawn there would stand over the wall, or along its top edge.
    ///
    /// A stretch is cut exactly, where the footprint's perimeter crosses the
    /// two lines either side of its boundary, the top's height read there:
    /// a flat top has a vertex at each end and nothing between, and cut at
    /// its vertices it went missing where the riser stands, or ran across a
    /// notch where none does. Its bottom is at the lesser band's height,
    /// under each of the top's vertices.
    pub fn band_riser_stretches(
        &self,
        riser_tops: &[(f64, Vec<P3<WorldSpace>>)],
        tile: Option<Affine2>,
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<RiserStretch> {
        if riser_tops.is_empty() {
            return Vec::new();
        }
        let side = 1e-9 * self.cell();
        let magnitude = refine.and_then(|r| r.magnitude.as_ref());
        let g = &*self.height;
        let corner0 = g.position(0, 0);
        let corner1 = g.position(g.width - 1, g.height - 1);
        let (y0, y1) = (swift::min(corner0.y(), corner1.y()), swift::max(corner0.y(), corner1.y()));
        let placed = move |v: P3<WorldSpace>| -> P3<WorldSpace> { tile.map_or(v, |t| t.apply(v)) };
        let mut out = Vec::new();
        for (b, top) in riser_tops {
            let b = *b;
            let (Some(&first), Some(&last)) = (top.first(), top.last()) else { continue };
            let left = self.truncation.height_at_from(b, Side::Left);
            let right = self.truncation.height_at_from(b, Side::Right);
            let (lesser, greater) = (swift::min(left, right), swift::max(left, right));
            let height = |y: f64| -> f64 {
                let Some(magnitude) = magnitude else { return first.z() };
                let u = magnitude(P2::new(b, y));
                swift::min(if u.is_nan() { f64::INFINITY } else { u }, greater)
            };
            // The stretches the footprint holds on both sides of the
            // boundary: with no footprint, the whole top.
            let mut stretches: Vec<(f64, f64)> = Vec::new();
            if let Region::Inside(perimeter) = region {
                if !(first.y() < last.y()) {
                    continue;
                }
                // Where the perimeter crosses the lines a hair either side
                // of the boundary, in the copy's place: w(y) = o + y d.
                let mut stops = vec![first.y(), last.y()];
                for x in [b - side, b + side] {
                    let o = placed(P3::new(x, 0.0, 0.0)).xy().v;
                    let d = placed(P3::new(x, 1.0, 0.0)).xy().v - o;
                    for edge in perimeter.edges() {
                        let p = edge.start.v;
                        let e = edge.end.v - edge.start.v;
                        let det = e.x * d.y - e.y * d.x;
                        if det == 0.0 {
                            continue;
                        }
                        let r = p - o;
                        let y = (e.x * r.y - e.y * r.x) / det;
                        let s = (d.x * r.y - d.y * r.x) / det;
                        if (0.0..=1.0).contains(&s) && y > first.y() && y < last.y() {
                            stops.push(y);
                        }
                    }
                }
                stops.sort_by(|a, b| order(*a, *b));
                let holds = |y: f64| -> bool {
                    region.contains(placed(P3::new(b - side, y, 0.0)).xy())
                        && region.contains(placed(P3::new(b + side, y, 0.0)).xy())
                };
                // The stretches between the stops the footprint holds on
                // both sides, run together.
                for pair in stops.windows(2) {
                    let (a, c) = (pair[0], pair[1]);
                    if !(c > a && holds(0.5 * (a + c))) {
                        continue;
                    }
                    match stretches.last_mut() {
                        Some(previous) if previous.1 == a => previous.1 = c,
                        _ => stretches.push((a, c)),
                    }
                }
            } else {
                stretches = vec![(first.y(), last.y())];
            }
            // The lower side, its line a hair from the boundary, and the way
            // out of the riser toward it in the copy's place, square to the
            // placed boundary.
            let x = if left < right { b - side } else { b + side };
            let a = placed(P3::new(b, y0, 0.0)).xy().v;
            let c = placed(P3::new(b, y1, 0.0)).xy().v;
            let along = (c - a).normalized();
            let across = placed(P3::new(if left < right { b - 1.0 } else { b + 1.0 }, y0, 0.0)).xy().v - a;
            let normal = Vec2::new(along.y, -along.x);
            let outward = normal * (if normal.dot(across) < 0.0 { -1.0 } else { 1.0 });
            for (lo, hi) in stretches {
                // Drawn from its first stop to its last through the top's
                // vertices between.
                let piece: Vec<P3<WorldSpace>> = if lo == first.y() && hi == last.y() {
                    top.clone()
                } else {
                    let mut piece = vec![if lo == first.y() { first } else { P3::new(b, lo, height(lo)) }];
                    piece.extend(top.iter().copied().filter(|v| v.y() > lo && v.y() < hi));
                    piece.push(if hi == last.y() { last } else { P3::new(b, hi, height(hi)) });
                    piece
                };
                let back = tile.map_or(Affine2::IDENTITY, |t| t.inverse().unwrap_or(Affine2::IDENTITY));
                out.push(RiserStretch {
                    top: piece.iter().map(|&v| placed(v)).collect(),
                    bottom: piece.iter().map(|v| placed(P3::new(x, v.y(), lesser))).collect(),
                    length: y1 - y0,
                    point: Arc::new(move |t| {
                        let p = Surface::point_from_to(P2::new(b, y0), P2::new(b, y1), t);
                        placed(P3::new(p.x(), p.y(), 0.0)).xy()
                    }),
                    lo: (lo - y0) / (y1 - y0),
                    hi: (hi - y0) / (y1 - y0),
                    opens: true,
                    lower: Arc::new(move |_| outward),
                    back: Arc::new(move |_| back),
                });
            }
        }
        out
    }

    /// The risers at the tile boundaries (`tile_boundary_risers`) where they
    /// stand (`tile_boundary_stretches`), a stretch at a time, in world
    /// space: its top at the higher of the two copies' tops, its bottom at
    /// the lower, a hair onto the lower copy's side, each read on the tile
    /// boundary through its tile (`tile_boundary_top`). Each edge has a
    /// vertex at each of the stretch's stops and, with a refiner, its chords
    /// are subdivided at their midpoints until none departs there from its
    /// top by more than the refiner's tolerance, seven deep. Unlike a
    /// riser's top at a band boundary (`banded_truncation_edge`), it seeks no
    /// minimum of |f| between two stops, so a dip whose midpoints all stand
    /// on the higher top is drawn across. A stretch runs on through a
    /// crossing of the copies' tops, where the riser turns to face the other
    /// way: its bottom and its outward normal follow the lower copy.
    pub fn tile_boundary_riser_stretches(
        &self,
        tiles: &[Affine2],
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<RiserStretch> {
        let magnitude = self.face_magnitude(refine);
        let cell = self.cell();
        let side = 1e-9 * cell;
        let d = self.domain();
        let centre = P3::<WorldSpace>::new(0.5 * (d.real.lo + d.real.hi), 0.5 * (d.imag.lo + d.imag.hi), 0.0);
        let subdividing = refine.and_then(|r| if r.magnitude.is_some() { Some(r.tolerance) } else { None });
        let mut out = Vec::new();
        for boundary in self.tile_boundaries(tiles) {
            let (Some(first), Some(second)) = (tiles[boundary.first].inverse(), tiles[boundary.second].inverse())
            else {
                continue;
            };
            let surface = self.clone();
            let m = magnitude.clone();
            let tops = Arc::new(move |w: P2<WorldSpace>| -> [f64; 2] {
                [first, second].map(|back| surface.tile_boundary_top(w, back, &|p| m(p)))
            });
            // Which side of the tile boundary each copy is on: its tile's
            // image of the domain's centre.
            let n = boundary.normal();
            let sides = [boundary.first, boundary.second]
                .map(|k| if (tiles[k].apply(centre).xy().v - boundary.start.v).dot(n) > 0.0 { n } else { -n });
            let tops_lower = tops.clone();
            let lower: Arc<dyn Fn(P2<WorldSpace>) -> Vec2 + Send + Sync> = Arc::new(move |w| {
                let z = tops_lower(w);
                if z[1] < z[0] { sides[1] } else { sides[0] }
            });
            let tops_vec = |w: P2<WorldSpace>| -> Vec<f64> { tops(w).to_vec() };
            let stretches = self.tile_boundary_stretches(&boundary, &[first, second], region, 1, &tops_vec, &|q| {
                magnitude(q) - self.truncation_height_at(q.x())
            });
            for (k, stretch) in stretches.iter().enumerate() {
                // The edge at one copy's top, the one `pick` takes of the two.
                let edge = |pick: &dyn Fn([f64; 2]) -> f64| -> Vec<P3<WorldSpace>> {
                    let mut curve: Vec<P3<WorldSpace>> = Vec::new();
                    let vertex = |s: f64, z: f64| -> P3<WorldSpace> {
                        let w = boundary.point(s);
                        P3::new(w.x(), w.y(), z)
                    };
                    let top = |s: f64| -> f64 { pick(tops(boundary.point(s))) };
                    for (j, stop) in stretch.iter().enumerate() {
                        let z = pick(tops(stop.read));
                        let at = P3::<WorldSpace>::new(stop.at.x(), stop.at.y(), z);
                        if j > 0
                            && let Some(tolerance) = subdividing
                        {
                            let a = stretch[j - 1].s;
                            let za = curve[curve.len() - 1].z();
                            subdivide_stop(&mut curve, &top, &vertex, (a, za), (stop.s, z), 0, cell, tolerance);
                        }
                        curve.push(at);
                    }
                    curve
                };
                let top = edge(&|z| sequence_max(z).unwrap_or(f64::NAN));
                let (Some(start), Some(end)) = (stretch.first(), stretch.last()) else { continue };
                if top.len() < 2 {
                    continue;
                }
                let bottom: Vec<P3<WorldSpace>> = edge(&|z| sequence_min(z).unwrap_or(f64::NAN))
                    .into_iter()
                    .map(|v| {
                        let o = side * lower(P2::new(v.x(), v.y()));
                        P3::new(v.x() + o.x, v.y() + o.y, v.z())
                    })
                    .collect();
                let length = boundary.length();
                // Where a copy's top steps the stretch before ends where this
                // one starts.
                let follows = k > 0 && stretches[k - 1].last().map(|s| s.s) == Some(start.s);
                let tops_back = tops.clone();
                out.push(RiserStretch {
                    top,
                    bottom,
                    length,
                    point: Arc::new(move |t| boundary.point(t * length)),
                    lo: start.s / length,
                    hi: end.s / length,
                    opens: !follows,
                    lower: lower.clone(),
                    back: Arc::new(move |w| {
                        let z = tops_back(w);
                        if z[1] < z[0] { second } else { first }
                    }),
                });
            }
        }
        out
    }

    /// The risers where they stand, a stretch at a time, as the truncation
    /// edge's layer draws their edges (`drawn_truncation_edge`): on every
    /// tile when `tiled`, each copy cut to the footprint where it stands,
    /// and at the tile boundaries; in place otherwise. A band boundary's only
    /// with a refiner, as their tops are drawn (`graph_trim_loops`); a tile
    /// boundary's either way.
    pub fn riser_stretches(
        &self,
        tiles: &[Affine2],
        tiled: bool,
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<RiserStretch> {
        let banded =
            matches!(self.truncation, Truncation::RealBands { .. }) && refine.is_some_and(|r| r.magnitude.is_some());
        if !(banded || (tiled && tiles.len() > 1)) {
            return Vec::new();
        }
        let riser_tops = if banded { self.graph_trim_loops(refine).riser_tops } else { Vec::new() };
        let copies: Vec<Option<Affine2>> = if tiled { tiles.iter().copied().map(Some).collect() } else { vec![None] };
        let mut out: Vec<RiserStretch> =
            copies.into_iter().flat_map(|tile| self.band_riser_stretches(&riser_tops, tile, region, refine)).collect();
        if tiled {
            out.extend(self.tile_boundary_riser_stretches(tiles, region, refine));
        }
        out
    }

    /// The hatch lines on the risers (`riser_stretches`), as on the walls
    /// (`wall_hatch_lines`): vertical, from a riser's bottom edge up to its
    /// top, a hair onto the lower side, at whole gaps along the line it
    /// stands on, the line's two ends drawn only with `end_lines`, and kept
    /// where the riser stands. A band boundary's line runs across the domain,
    /// as the plateaus' hatching divides it, so a plateau's lines along Re z
    /// meet a riser's on its edges at the same spacing.
    ///
    /// A stretch holds the lines from its start to its end, but for one at
    /// its start where another ends: at a tile boundary a stretch breaks
    /// where a copy's top steps, and the next starts there
    /// (`tile_boundary_stretches`), and a line on that stop is the first's.
    pub fn riser_hatch_lines(
        &self,
        spacing: f64,
        end_lines: bool,
        tiles: &[Affine2],
        tiled: bool,
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let side = 1e-9 * self.cell();
        let mut out = Vec::new();
        for riser in self.riser_stretches(tiles, tiled, region, refine) {
            let n = Surface::stroke_count(riser.length, spacing);
            let range = if end_lines { 0..n } else { 1..(n - 1).max(1) };
            for k in range {
                let t = k as f64 / (n - 1).max(1) as f64;
                if !((t > riser.lo || riser.opens) && t <= riser.hi) {
                    continue;
                }
                let w = (riser.point)(t);
                let (Some(top), Some(base)) = (
                    self.top_edge_height(&riser.top, domain_point(w)),
                    self.top_edge_height(&riser.bottom, domain_point(w)),
                ) else {
                    continue;
                };
                let o = side * (riser.lower)(w);
                if let Some(stroke) = Surface::vertical(w.x() + o.x, w.y() + o.y, base, top) {
                    out.push(stroke);
                }
            }
        }
        out
    }

    /// The stretches of a tile boundary a riser stands on: where the
    /// footprint holds both sides, read a hair either side of the tile
    /// boundary, and one copy's top stands over the other's (`stands`),
    /// `tops` reading each copy's over a point of the tile boundary, `backs`
    /// the two tiles' inverses. Each stretch is its stops in order: the
    /// distance `s` along the tile boundary, the point `at` it is, and the
    /// point its tops are read at, at its ends a hair inside it, since a
    /// copy's top can step there, or the footprint end. The stops: each
    /// copy's grid nodes along the tile boundary (every `step`-th); where the
    /// footprint's perimeter crosses the lines either side of the tile
    /// boundary, exactly, on the edge itself where the edge or the boundary
    /// runs along an axis; where a copy's top steps, at a band boundary its
    /// side of the domain crosses; where a copy's `excess` over its
    /// truncation changes sign; and where the copies' tops cross, or the
    /// riser stops standing otherwise, bisected to the last bit. Stops
    /// nearer than a billionth of a cell are one, a step or the footprint's
    /// end kept over the others: a band boundary on one copy's grid node
    /// lands a rounding from the other copy's node there. A stretch runs on
    /// through a crossing of the tops and breaks where a copy's top steps.
    pub fn tile_boundary_stretches(
        &self,
        boundary: &TileBoundary,
        backs: &[Affine2],
        region: &Region,
        step: usize,
        tops: &dyn Fn(P2<WorldSpace>) -> Vec<f64>,
        excess: &dyn Fn(P2<DomainSpace>) -> f64,
    ) -> Vec<Vec<TileBoundaryStop>> {
        let cell = self.cell();
        let (side, nudge) = (1e-9 * cell, 1e-9 * cell);
        let length = boundary.length();
        let n = boundary.normal();
        let g = &*self.height;
        let held = |s: f64| -> bool {
            let w = boundary.point(s).v;
            region.contains(P2::from_vec(w - side * n)) && region.contains(P2::from_vec(w + side * n))
        };
        let stands = |s: f64| -> bool {
            let z = tops(boundary.point(s));
            held(s) && (Surface::stands(z[0], z[1]) || Surface::stands(z[1], z[0]))
        };
        let copy = |k: usize, s: f64| -> P2<DomainSpace> {
            let w = boundary.point(s);
            self.onto_domain(backs[k].apply(P3::new(w.x(), w.y(), 0.0)).xy())
        };
        // Each stop with what it is: a grid node, or bisected, 0; where the
        // footprint ends, 1; where a copy's top steps, 2.
        type Stop = (f64, u8, Option<P2<WorldSpace>>);
        let mut stops: Vec<Stop> = vec![(0.0, 0, None), (length, 0, None)];
        let stride = step.max(1);
        for k in 0..backs.len() {
            let (q0, q1) = (copy(k, 0.0), copy(k, length));
            let along = |c: f64, c0: f64, c1: f64| -> Option<f64> {
                let s = length * (c - c0) / (c1 - c0);
                if s > 0.0 && s < length { Some(s) } else { None }
            };
            if q0.x() != q1.x() {
                for i in (0..g.width).step_by(stride) {
                    if let Some(s) = along(g.position(i, 0).x(), q0.x(), q1.x()) {
                        stops.push((s, 0, None));
                    }
                }
                if let Truncation::RealBands { bands, .. } = &self.truncation {
                    for b in bands {
                        if let Some(s) = along(b.below, q0.x(), q1.x()) {
                            stops.push((s, 2, None));
                        }
                    }
                }
            }
            if q0.y() != q1.y() {
                for j in (0..g.height).step_by(stride) {
                    if let Some(s) = along(g.position(0, j).y(), q0.y(), q1.y()) {
                        stops.push((s, 0, None));
                    }
                }
            }
        }
        // Where the perimeter crosses the lines either side of the tile
        // boundary, o + s u, at the edge's p + r e.
        if let Region::Inside(perimeter) = region {
            let u = (boundary.end.v - boundary.start.v) / length;
            for o in [boundary.start.v - side * n, boundary.start.v + side * n] {
                for edge in perimeter.edges() {
                    let p = edge.start.v;
                    let e = edge.end.v - edge.start.v;
                    let det = e.x * u.y - e.y * u.x;
                    if det == 0.0 {
                        continue;
                    }
                    let r = p - o;
                    let mut s = (e.x * r.y - e.y * r.x) / det;
                    let t = (u.x * r.y - u.y * r.x) / det;
                    if !(0.0..=1.0).contains(&t) {
                        continue;
                    }
                    // Along an axis, the crossing's coordinate there is the
                    // edge's or the boundary's own, not a ratio a rounding
                    // off it.
                    let mut at: Option<P2<WorldSpace>> = None;
                    if u.x == 0.0 {
                        let y = if e.y == 0.0 { p.y } else { p.y + t * e.y };
                        at = Some(P2::new(boundary.start.x(), y));
                        s = (y - boundary.start.y()) / u.y;
                    } else if u.y == 0.0 {
                        let x = if e.x == 0.0 { p.x } else { p.x + t * e.x };
                        at = Some(P2::new(x, boundary.start.y()));
                        s = (x - boundary.start.x()) / u.x;
                    }
                    if s > 0.0 && s < length {
                        stops.push((s, 1, at));
                    }
                }
            }
        }
        // The stops in order, those nearer than `nudge` one, of the greater
        // kind.
        let merged = |stops: &[Stop]| -> Vec<Stop> {
            let mut sorted = stops.to_vec();
            sorted.sort_by(|a, b| order(a.0, b.0));
            let mut out: Vec<Stop> = Vec::new();
            for stop in sorted {
                match out.last_mut() {
                    Some(last) if stop.0 - last.0 <= nudge => {
                        if stop.1 > last.1 {
                            *last = stop;
                        }
                    }
                    _ => out.push(stop),
                }
            }
            out
        };
        let sorted: Vec<f64> = merged(&stops).iter().map(|s| s.0).collect();
        let mut more: Vec<Stop> = Vec::new();
        for pair in sorted.windows(2) {
            let (ia, ic) = (pair[0] + nudge, pair[1] - nudge);
            if !(ic > ia) {
                continue;
            }
            for k in 0..backs.len() {
                let (ea, ec) = (excess(copy(k, ia)), excess(copy(k, ic)));
                if ea.is_finite() && ec.is_finite() && ((ea > 0.0) != (ec > 0.0)) {
                    let (holds, fails) = bisect(ia, ic, |s| (excess(copy(k, s)) > 0.0) == (ea > 0.0));
                    more.push((0.5 * (holds + fails), 0, None));
                }
            }
            let (da, dc) = (tops(boundary.point(ia)), tops(boundary.point(ic)));
            if (da[0] - da[1]).is_finite() && (dc[0] - dc[1]).is_finite() && ((da[0] > da[1]) != (dc[0] > dc[1])) {
                let (holds, fails) = bisect(ia, ic, |s| {
                    let z = tops(boundary.point(s));
                    (z[0] > z[1]) == (da[0] > da[1])
                });
                more.push((0.5 * (holds + fails), 0, None));
            } else if stands(ia) != stands(ic) {
                let standing = stands(ia);
                let (holds, fails) = bisect(ia, ic, |s| stands(s) == standing);
                more.push((0.5 * (holds + fails), 0, None));
            }
        }
        let mut all = merged(&stops);
        all.extend(more);
        let all = merged(&all);
        let mut stretches: Vec<Vec<TileBoundaryStop>> = Vec::new();
        let mut open = false;
        for pair in all.windows(2) {
            let (a, c) = (pair[0], pair[1]);
            if !stands(0.5 * (a.0 + c.0)) {
                open = false;
                continue;
            }
            // Read from within at the stretch's ends; where it runs on, a
            // stop between is read where it is.
            let start = TileBoundaryStop {
                s: a.0,
                at: a.2.unwrap_or_else(|| boundary.point(a.0)),
                read: boundary.point(a.0 + nudge),
            };
            let end = TileBoundaryStop {
                s: c.0,
                at: c.2.unwrap_or_else(|| boundary.point(c.0)),
                read: boundary.point(c.0 - nudge),
            };
            if open && a.1 != 2 {
                let last = stretches.last_mut().expect("an open stretch");
                let n = last.len();
                last[n - 1].read = boundary.point(a.0);
                last.push(end);
            } else {
                stretches.push(vec![start, end]);
            }
            open = true;
        }
        stretches
    }
}

/// `tileBoundaryRiserStretches`' `subdivide`: the chord of a riser's edge
/// from `(s0, z0)` to `(s1, z1)` along the tile boundary, split at its
/// midpoint onto `top` where it departs from it by more than `tolerance`
/// cells, seven deep.
#[allow(clippy::too_many_arguments)]
fn subdivide_stop(
    curve: &mut Vec<P3<WorldSpace>>,
    top: &dyn Fn(f64) -> f64,
    vertex: &dyn Fn(f64, f64) -> P3<WorldSpace>,
    (s0, z0): (f64, f64),
    (s1, z1): (f64, f64),
    depth: usize,
    cell: f64,
    tolerance: f64,
) {
    if !(depth < 7 && s1 - s0 > 1e-9 * cell) {
        return;
    }
    let sm = 0.5 * (s0 + s1);
    let zm = top(sm);
    if !(zm.is_finite() && (zm - 0.5 * (z0 + z1)).abs() > tolerance * cell) {
        return;
    }
    subdivide_stop(curve, top, vertex, (s0, z0), (sm, zm), depth + 1, cell, tolerance);
    curve.push(vertex(sm, zm));
    subdivide_stop(curve, top, vertex, (sm, zm), (s1, z1), depth + 1, cell, tolerance);
}

/// A placed stretch of a band's truncation edge with an end on a column
/// beside a cut cell (`bandedTruncationEdge`'s `Stretch`): its points, and
/// at each such end the crossing there as the grid placed it, which the
/// stretch it meets shares. A run's are the refiner's; a part's line's are
/// solved here.
#[derive(Debug, Clone)]
struct Stretch {
    points: Vec<P2<DomainSpace>>,
    ends: (Option<P2<DomainSpace>>, Option<P2<DomainSpace>>),
    refined: bool,
}

impl Stretch {
    fn reverse(&mut self) {
        self.points.reverse();
        self.ends = (self.ends.1, self.ends.0);
    }
}

/// What `bandedTruncationEdge`'s `emit` writes to: the curves returned, the
/// placed pieces by band, and the ends of placed pieces that lie on a
/// boundary, by boundary, with the truncation they belong to.
struct Emitter<'a> {
    boundaries: &'a [f64],
    out: Vec<Vec<P3<WorldSpace>>>,
    pieces: &'a mut HashMap<usize, Vec<Vec<P2<DomainSpace>>>>,
    ends_on_boundary: HashMap<u64, Vec<(f64, P2<DomainSpace>)>>,
}

impl Emitter<'_> {
    fn emit(&mut self, piece: Vec<P2<DomainSpace>>, band: usize, truncation_height: f64) {
        if piece.first() != piece.last() {
            for p in [piece[0], piece[piece.len() - 1]] {
                if self.boundaries.contains(&p.x()) {
                    self.ends_on_boundary.entry(double_key(p.x())).or_default().push((truncation_height, p));
                }
            }
        }
        self.out.push(piece.iter().map(|p| P3::new(p.x(), p.y(), truncation_height)).collect());
        self.pieces.entry(band).or_default().push(piece);
    }
}

/// `bandedTruncationEdge`'s `subdivided`: a part's line with its chords
/// taken onto f.
struct PartSubdivider<'a> {
    surface: &'a Surface,
    magnitude: &'a dyn Fn(P2<DomainSpace>) -> f64,
    boundaries: &'a [f64],
    h: f64,
    l: f64,
    r: f64,
    cell: f64,
    tolerance: f64,
}

impl PartSubdivider<'_> {
    fn excess(&self, p: P2<DomainSpace>) -> f64 {
        (self.magnitude)(p) - self.h
    }

    fn subdivide(&self, pieces: &mut Vec<Vec<P2<DomainSpace>>>, a: P2<DomainSpace>, b: P2<DomainSpace>, depth: usize) {
        let cell = self.cell;
        let step = 1e-4 * cell;
        let chord = (b.v - a.v).length();
        if !(depth < 5 && chord > 1e-9 * cell) {
            return;
        }
        let m = P2::<DomainSpace>::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
        let e = self.excess(m);
        let gx = (self.excess(P2::new(m.x() + step, m.y())) - self.excess(P2::new(m.x() - step, m.y()))) / (2.0 * step);
        let gy = (self.excess(P2::new(m.x(), m.y() + step)) - self.excess(P2::new(m.x(), m.y() - step))) / (2.0 * step);
        let slope = (gx * gx + gy * gy).sqrt();
        if !(e.is_finite() && slope > 0.0 && slope.is_finite() && e.abs() / slope > self.tolerance) {
            return;
        }
        // Toward the level: marching squares keeps the plateau on a line's
        // left, so from a midpoint on it the curve lies to the right, and
        // from one off it to the left, whichever way |f| falls there, as
        // beside a pole within the chord. As far as a chord's length, and no
        // further than the band's edges.
        let left = Vec2::new(a.y() - b.y(), b.x() - a.x()) / chord;
        let toward = if e > 0.0 { -left } else { left };
        let edge = if toward.x > 0.0 { self.r } else { self.l };
        let farthest = if toward.x == 0.0 { chord } else { swift::min(chord, (edge - m.x()) / toward.x) };
        let mut reach = e.abs() / slope;
        let mut q: Option<P2<DomainSpace>> = None;
        let mut end = m;
        let excess = |p: P2<DomainSpace>| self.excess(p);
        while reach > 0.0 && q.is_none() {
            end = P2::from_vec(m.v + swift::min(reach, farthest) * toward);
            q = self.surface.truncation_crossing_between(m, e, end, excess(end), excess);
            if !(reach < farthest) {
                break;
            }
            reach *= 2.0;
        }
        if let Some(q) = q {
            self.subdivide(pieces, a, q, depth + 1);
            pieces.last_mut().expect("a piece").push(q);
            self.subdivide(pieces, q, b, depth + 1);
        } else if farthest < chord
            && end.x() == edge
            && self.boundaries.contains(&edge)
            && let Some(ca) = self.surface.truncation_crossing_between(
                end,
                excess(end),
                P2::new(edge, a.y()),
                excess(P2::new(edge, a.y())),
                excess,
            )
            && let Some(cb) = self.surface.truncation_crossing_between(
                end,
                excess(end),
                P2::new(edge, b.y()),
                excess(P2::new(edge, b.y())),
                excess,
            )
        {
            self.subdivide(pieces, a, ca, depth + 1);
            pieces.last_mut().expect("a piece").push(ca);
            pieces.push(vec![cb]);
            self.subdivide(pieces, cb, b, depth + 1);
        }
    }

    /// A part's line with its chords taken onto f, as the refiner takes a
    /// run's: a chord whose midpoint is off |f| = h by more than the
    /// tolerance, to first order, is split where the level crosses its
    /// normal, the nearest crossing toward the level within the band,
    /// between `l` and `r`, and the halves likewise, five deep. The curve
    /// can leave the cut cell where the grid beside it cannot see it, as
    /// round a zero on the boundary whose hole is wider than the part. It
    /// can leave the band too, through a boundary between two rows: the
    /// line is then broken there, at the level's two crossings of the
    /// boundary either side of where the search met it, into pieces.
    fn subdivided(&self, line: &[P2<DomainSpace>]) -> Vec<Vec<P2<DomainSpace>>> {
        let Some(&first) = line.first() else { return Vec::new() };
        let mut pieces = vec![vec![first]];
        for pair in line.windows(2) {
            self.subdivide(&mut pieces, pair[0], pair[1], 0);
            pieces.last_mut().expect("a piece").push(pair[1]);
        }
        // A closed line broken is open, from one break to the next.
        if line.first() == line.last() && pieces.len() > 1 {
            let mut last = pieces.pop().expect("a piece");
            last.extend_from_slice(&pieces[0][1..]);
            pieces[0] = last;
        }
        pieces.into_iter().filter(|p| p.len() >= 2).collect()
    }
}

impl Surface {
    /// The graph's trim loops, and the truncation edge drawn on them.
    ///
    /// The truncation edge is the zero level of the grid's excess over the
    /// truncation, at the truncation height. With a refiner it is the
    /// contour |f| = h(x) placed on f, as the rings are, and an end of it on
    /// the domain's edge is re-solved by the top edge's own bisection along
    /// that edge (`truncation_crossing_between`), so the top edge meets the
    /// truncation edge to rounding. A banded truncation edge is placed band
    /// by band at each band's own level (`banded_truncation_edge`), with the
    /// top of each riser.
    ///
    /// The loops are the domain's edge and the plateaus' loops: the
    /// truncation edge, closed along the domain's edge where a plateau
    /// reaches it. Without a refiner they are the excess grid's, one set for
    /// all the bands. With one they are placed band by band, and each band's
    /// are closed along the band's own edges as well, across the risers: the
    /// top of a riser lies on the lower band's loops, and where both sides
    /// of a band boundary are plateau, both bands' loops run along it. The
    /// loops bound the graph even-odd, a stretch passed twice bounding
    /// nothing. The truncation edge lies on them, riser tops included; a
    /// closing stretch is where the domain's or the band's edge is plateau,
    /// read on the excess grid without a refiner and on f with one. Each
    /// loop ends where it begins; `open` counts the chains that could not
    /// be closed, at an end off the edges with no other end there, which
    /// are among the loops as they are. `riser_tops` are the risers' tops,
    /// which end `truncation_edge`, each with its boundary.
    pub fn graph_trim_loops(&self, refine: Option<&ContourRefine>) -> GraphTrimLoops {
        let g = &*self.height;
        let corner0 = g.position(0, 0);
        let corner1 = g.position(g.width - 1, g.height - 1);
        let rectangle = Rectangle {
            x0: swift::min(corner0.x(), corner1.x()),
            x1: swift::max(corner0.x(), corner1.x()),
            y0: swift::min(corner0.y(), corner1.y()),
            y1: swift::max(corner0.y(), corner1.y()),
        };
        let corners = rectangle.corners();
        let mut domain_edge: Vec<P2<DomainSpace>> = corners.iter().map(|c| c.1).collect();
        domain_edge.push(corners[0].1);
        let Some(excess) = self.truncation_excess() else {
            return GraphTrimLoops { loops: vec![domain_edge], ..GraphTrimLoops::default() };
        };
        let mut out: Vec<Vec<P3<WorldSpace>>> = Vec::new();
        let mut riser_tops: Vec<(f64, Vec<P3<WorldSpace>>)> = Vec::new();
        let mut plateaus: Vec<PlateauLoops<'_>> = Vec::new();
        let magnitude = refine.and_then(|r| r.magnitude.as_ref());
        let plateau = |magnitude: &'_ crate::surface::Magnitude, truncation_height: f64| {
            let magnitude = magnitude.clone();
            Box::new(move |p: P2<DomainSpace>| !(magnitude(p) < truncation_height))
                as Box<dyn Fn(P2<DomainSpace>) -> bool>
        };
        for (_, lines) in contour::levels(&excess, &[0.0]) {
            if let (Some(refine), Some(magnitude)) = (refine, magnitude) {
                match &self.truncation {
                    Truncation::Uniform(truncation_height) if truncation_height.is_finite() => {
                        let truncation_height = *truncation_height;
                        let mut whole = PlateauLoops::new(rectangle, plateau(magnitude, truncation_height));
                        let mut placed = (refine.contours)(
                            ContourField::Magnitude,
                            truncation_height,
                            &lines,
                            f64::INFINITY,
                            ContourRefine::EVERYWHERE,
                        );
                        if let Some(rings) = &refine.rings {
                            let more = rings(truncation_height, &placed);
                            placed.extend(more);
                        }
                        for mut line in placed {
                            if line.len() >= 2 {
                                let last = line.len() - 1;
                                for i in [0, last] {
                                    if let Some(q) = self.on_boundary_truncation_crossing(line[i], &**magnitude) {
                                        line[i] = q;
                                    }
                                }
                            }
                            out.push(line.iter().map(|p| P3::new(p.x(), p.y(), truncation_height)).collect());
                            whole.pieces.push(line);
                        }
                        plateaus.push(whole);
                        continue;
                    }
                    Truncation::RealBands { bands, beyond } => {
                        let mut pieces: HashMap<usize, Vec<Vec<P2<DomainSpace>>>> = HashMap::new();
                        out.extend(self.banded_truncation_edge(
                            &lines,
                            bands,
                            *beyond,
                            refine,
                            &**magnitude,
                            &mut pieces,
                            &mut riser_tops,
                        ));
                        out.extend(riser_tops.iter().map(|t| t.1.clone()));
                        let real = self.domain().real;
                        let (lo, hi) = (swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
                        for span in self.truncation.bands(lo, hi) {
                            if !span.height.is_finite() {
                                continue;
                            }
                            let mut band = PlateauLoops::new(
                                Rectangle {
                                    x0: if span.lo == lo { rectangle.x0 } else { span.lo },
                                    x1: if span.hi == hi { rectangle.x1 } else { span.hi },
                                    y0: rectangle.y0,
                                    y1: rectangle.y1,
                                },
                                plateau(magnitude, span.height),
                            );
                            // Every placed piece is filed under a band of
                            // finite truncation that the domain holds, its
                            // ends on that band's edges at most, so its band
                            // has a span here.
                            band.pieces = pieces.get(&span.band).cloned().unwrap_or_default();
                            plateaus.push(band);
                        }
                        continue;
                    }
                    _ => {}
                }
            }
            let excess = excess.clone();
            let mut whole = PlateauLoops::new(rectangle, Box::new(move |p| excess.sample(p) > 0.0));
            for line in lines {
                out.push(line.iter().map(|p| P3::new(p.x(), p.y(), self.truncation_height_at(p.x()))).collect());
                whole.pieces.push(line);
            }
            plateaus.push(whole);
        }
        let closed: Vec<(Vec<Vec<P2<DomainSpace>>>, usize)> =
            plateaus.iter().map(|p| p.loops(self.edge_tolerance())).collect();
        let open = closed.iter().map(|c| c.1).sum();
        let mut loops = vec![domain_edge];
        loops.extend(closed.into_iter().flat_map(|c| c.0));
        GraphTrimLoops { loops, truncation_edge: out, riser_tops, open }
    }

    /// The truncation edge of a banded truncation, on f: in each band the
    /// contour |f| = that band's truncation, placed at the band's own level
    /// and cut exactly at the band's edges.
    ///
    /// The grid's truncation edge lines are the zero level of the excess
    /// over each column's own band. In a cell a band boundary cuts -- one
    /// between its columns, or on the greater, which reads the band beyond
    /// -- that excess is interpolated between two truncations and means
    /// nothing, and a band narrower than a column has no column of its own
    /// at all. So the lines are cut out of those cells, and each such cell is
    /// contoured again band by band: cut at its boundaries into parts, each
    /// part over its own band's excess, read on the grid along a column of
    /// that band and on f along a boundary, every crossing in it solved on f
    /// at that band's level, one on a boundary on the boundary itself, and
    /// the chords between taken onto f as the refiner takes a run's. The runs
    /// of the grid's lines between those cells are placed by the refiner and
    /// joined end to end to the parts' lines where they meet: on a column
    /// beside a cut cell, where both crossed the same edge from the same two
    /// samples, the refiner's point standing for both. The rings round the
    /// zeros the grid could not see are seeded once a band, at its level,
    /// and those the band's truncation edge does not already draw are cut to
    /// the band exactly too.
    ///
    /// Where the truncation steps the surface has a riser, and its top is
    /// the truncation edge there: along the boundary, at f's own height up to
    /// the greater truncation, wherever the lesser band's plateau holds the
    /// boundary, from one end of that band's truncation edge on it to the
    /// next, or to the domain's edge.
    ///
    /// Each placed piece, as drawn but on the ground, goes into `pieces`
    /// under its band, as `band_at` numbers it; the risers' tops go into
    /// `riser_tops` instead, each with its boundary, and are not among the
    /// curves returned.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn banded_truncation_edge(
        &self,
        lines: &[Vec<P2<DomainSpace>>],
        bands: &[RealBand],
        _beyond: f64,
        refine: &ContourRefine,
        magnitude: &dyn Fn(P2<DomainSpace>) -> f64,
        pieces: &mut HashMap<usize, Vec<Vec<P2<DomainSpace>>>>,
        riser_tops: &mut Vec<(f64, Vec<P3<WorldSpace>>)>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        // The boundaries inside the domain: one on its edge has no riser,
        // the band beyond the far edge starting outside the domain.
        let real = self.domain().real;
        let (lo, hi) = (swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
        let mut boundaries: Vec<f64> = bands.iter().map(|b| b.below).filter(|&b| b > lo && b < hi).collect();
        boundaries.sort_by(|a, b| order(*a, *b));
        let boundaries = boundaries.as_slice();
        let g = &*self.height;
        let cell = self.cell();
        let tolerance = refine.tolerance * cell;
        let mag = |p: P2<DomainSpace>| -> f64 { magnitude(p) };
        let columns: Vec<f64> = (0..g.width).map(|x| g.position(x, 0).x()).collect();
        let rows: Vec<f64> = (0..g.height).map(|y| g.position(0, y).y()).collect();
        // The cells a boundary cuts: the lesser and greater of their
        // columns' x, with the columns' indices.
        let cut: Vec<(f64, f64, (usize, usize))> = (0..g.width.saturating_sub(1))
            .filter_map(|i| {
                let (l, h) = if columns[i] <= columns[i + 1] { (i, i + 1) } else { (i + 1, i) };
                if !boundaries.iter().any(|&b| b > columns[l] && b <= columns[h]) {
                    return None;
                }
                Some((columns[l], columns[h], (l, h)))
            })
            .collect();
        let in_cut = |x: f64| -> bool { cut.iter().any(|c| x > c.0 && x < c.1) };

        let mut emitter = Emitter { boundaries, out: Vec::new(), pieces, ends_on_boundary: HashMap::new() };
        let mut stretches: BTreeMap<usize, (f64, Vec<Stretch>)> = BTreeMap::new();
        // A run's end on a column beside a cut cell, where a part's line
        // meets it.
        let meets_part = |p: Option<P2<DomainSpace>>| -> Option<P2<DomainSpace>> {
            let p = p?;
            if cut.iter().any(|c| p.x() == c.0 || p.x() == c.1) { Some(p) } else { None }
        };

        for line in lines {
            // Runs of one band's vertices off the cut cells: a vertex in one
            // is dropped, and a run is broken across a segment through one.
            let mut runs: Vec<Vec<P2<DomainSpace>>> = Vec::new();
            let mut broken = true;
            for &v in line {
                if in_cut(v.x()) {
                    broken = true;
                    continue;
                }
                let joins = !broken
                    && runs.last().and_then(|r| r.last()).is_some_and(|&u| {
                        !in_cut(0.5 * (u.x() + v.x()))
                            && self.truncation_height_at(u.x()) == self.truncation_height_at(v.x())
                    });
                if joins {
                    runs.last_mut().expect("a run").push(v);
                } else {
                    runs.push(vec![v]);
                }
                broken = false;
            }
            // A closed line's last run goes on into its first, unbroken at
            // the line's start.
            if line.first() == line.last()
                && runs.len() >= 2
                && runs[0].first() == line.first()
                && runs[runs.len() - 1].last() == line.last()
            {
                let mut last = runs.pop().expect("a run");
                last.extend_from_slice(&runs[0][1..]);
                runs[0] = last;
            }
            for run in runs.iter().filter(|r| r.len() >= 2) {
                let truncation_height = self.truncation_height_at(run[0].x());
                let Some(band) = self.band_at(run[0].x()) else { continue };
                if !truncation_height.is_finite() {
                    continue;
                }
                let closed = run.first() == run.last();
                // Kept to its band, between the boundaries either side.
                let x = run[0].x();
                let span: RangeInclusive<f64> =
                    boundaries.iter().rev().copied().find(|&b| b <= x).unwrap_or(f64::NEG_INFINITY)
                        ..=boundaries.iter().copied().find(|&b| b > x).unwrap_or(f64::INFINITY);
                let refined = (refine.contours)(
                    ContourField::Magnitude,
                    truncation_height,
                    std::slice::from_ref(run),
                    f64::INFINITY,
                    span,
                );
                let count = refined.len();
                for (index, mut piece) in refined.into_iter().enumerate() {
                    if piece.len() < 2 {
                        continue;
                    }
                    // A closed piece has no ends.
                    for at_start in [true, false] {
                        if piece.first() == piece.last() {
                            continue;
                        }
                        let i = if at_start { 0 } else { piece.len() - 1 };
                        if let Some(q) = self.on_boundary_truncation_crossing(piece[i], magnitude) {
                            piece[i] = q;
                        }
                    }
                    let ends = (
                        if closed || index > 0 { None } else { meets_part(run.first().copied()) },
                        if closed || index < count - 1 { None } else { meets_part(run.last().copied()) },
                    );
                    if ends.0.is_none() && ends.1.is_none() {
                        emitter.emit(piece, band, truncation_height);
                    } else {
                        stretches.entry(band).or_insert((truncation_height, Vec::new())).1.push(Stretch {
                            points: piece,
                            ends,
                            refined: true,
                        });
                    }
                }
            }
        }

        // The cut cells, band by band.
        for c in &cut {
            let mut stops = vec![c.0];
            stops.extend(boundaries.iter().copied().filter(|&b| b > c.0 && b < c.1));
            stops.push(c.1);
            for pair in stops.windows(2) {
                let (l, r) = (pair[0], pair[1]);
                if !(r > l) {
                    continue;
                }
                let truncation_height = self.truncation_height_at(0.5 * (l + r));
                let Some(band) = self.band_at(0.5 * (l + r)) else { continue };
                if !truncation_height.is_finite() {
                    continue;
                }
                // Each side of the part: a column of this band, read on the
                // grid as the excess is, or a boundary, read on f.
                let side = |x: f64, i: usize| -> (f64, Option<usize>) {
                    (
                        x,
                        if x == columns[i] && self.truncation_height_at(x) == truncation_height {
                            Some(i)
                        } else {
                            None
                        },
                    )
                };
                let sides = [side(l, c.2.0), side(r, c.2.1)];
                let mut values = vec![0.0f32; 2 * g.height];
                for j in 0..g.height {
                    for (k, s) in sides.iter().enumerate() {
                        values[2 * j + k] = match s.1 {
                            Some(column) => g.at(column, j) - truncation_height as f32,
                            None => (mag(P2::new(s.0, rows[j])) - truncation_height) as f32,
                        };
                    }
                }
                let part = Grid2D::new(2, g.height, g.domain, values);
                // A crossing as marching squares places it, by the
                // arithmetic `contour::lines` does, so on a column it is the
                // grid's own; and solved on f.
                let ends = |a: [usize; 2], b: [usize; 2]| -> (P2<DomainSpace>, P2<DomainSpace>) {
                    (P2::new(sides[a[0]].0, rows[a[1]]), P2::new(sides[b[0]].0, rows[b[1]]))
                };
                let interpolated = |a: [usize; 2], b: [usize; 2]| -> P2<DomainSpace> {
                    let (p, q) = ends(a, b);
                    let za = f64::from(part.at(a[0], a[1]));
                    let zb = f64::from(part.at(b[0], b[1]));
                    let t = swift::min(swift::max((0.0 - za) / (zb - za), 0.0), 1.0);
                    P2::new(p.x() + (q.x() - p.x()) * t, p.y() + (q.y() - p.y()) * t)
                };
                let graded = contour::lines_placed(&part, 0.0, None, (false, false), |a, b| Some(interpolated(a, b)));
                let solved = contour::lines_placed(&part, 0.0, None, (false, false), |a, b| {
                    let (p, q) = ends(a, b);
                    Some(
                        self.truncation_crossing_between(
                            p,
                            mag(p) - truncation_height,
                            q,
                            mag(q) - truncation_height,
                            |x| mag(x) - truncation_height,
                        )
                        .unwrap_or_else(|| interpolated(a, b)),
                    )
                });
                // A line's end on a column of the band, where a run meets it.
                let meets_run = |p: P2<DomainSpace>| -> Option<P2<DomainSpace>> {
                    if sides.iter().any(|s| s.1.is_some() && s.0 == p.x()) { Some(p) } else { None }
                };
                let subdivider = PartSubdivider {
                    surface: self,
                    magnitude,
                    boundaries,
                    h: truncation_height,
                    l: boundaries.iter().rev().copied().find(|&b| b <= l).unwrap_or(lo),
                    r: boundaries.iter().copied().find(|&b| b >= r).unwrap_or(hi),
                    cell,
                    tolerance,
                };
                for (grid, solved) in graded.iter().zip(solved.iter()) {
                    let placed = subdivider.subdivided(solved);
                    let count = placed.len();
                    for (k, piece) in placed.into_iter().enumerate() {
                        let ends = if grid.first() == grid.last() {
                            (None, None)
                        } else {
                            (
                                if k == 0 { meets_run(grid[0]) } else { None },
                                if k == count - 1 { meets_run(grid[grid.len() - 1]) } else { None },
                            )
                        };
                        if ends.0.is_none() && ends.1.is_none() {
                            emitter.emit(piece, band, truncation_height);
                        } else {
                            stretches.entry(band).or_insert((truncation_height, Vec::new())).1.push(Stretch {
                                points: piece,
                                ends,
                                refined: false,
                            });
                        }
                    }
                }
            }
        }

        // The stretches joined where they meet, each band's in turn: from an
        // end that meets nothing, then round what is left, which closes.
        // Where a run meets a part's line its point stands for both.
        for (&band, (truncation_height, list)) in &stretches {
            let truncation_height = *truncation_height;
            let mut at: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
            for (k, s) in list.iter().enumerate() {
                for key in [s.ends.0, s.ends.1].into_iter().flatten() {
                    if let Some(key) = point_key(key) {
                        at.entry(key).or_default().push(k);
                    }
                }
            }
            let met = |key: Option<P2<DomainSpace>>, k: usize| -> bool {
                key.and_then(point_key).is_some_and(|key| at.get(&key).is_some_and(|ks| ks.iter().any(|&j| j != k)))
            };
            let mut used = vec![false; list.len()];
            for closing in [false, true] {
                for k0 in 0..list.len() {
                    if used[k0] {
                        continue;
                    }
                    let mut first = list[k0].clone();
                    if !closing {
                        if met(first.ends.0, k0) && met(first.ends.1, k0) {
                            continue;
                        }
                        if met(first.ends.0, k0) {
                            first.reverse();
                        }
                    }
                    used[k0] = true;
                    let mut points = first.points.clone();
                    let mut last = first.clone();
                    while let Some(key) = last.ends.1 {
                        let Some(k) = point_key(key)
                            .and_then(|pk| at.get(&pk))
                            .and_then(|ks| ks.iter().copied().find(|&j| !used[j]))
                        else {
                            break;
                        };
                        used[k] = true;
                        let mut next = list[k].clone();
                        if next.ends.0 != Some(key) {
                            next.reverse();
                        }
                        if next.refined && !last.refined {
                            points.pop();
                            points.extend_from_slice(&next.points);
                        } else {
                            points.extend_from_slice(&next.points[1..]);
                        }
                        last = next;
                    }
                    if closing && last.ends.1.is_some() && last.ends.1 == first.ends.0 {
                        let n = points.len();
                        if last.refined && !first.refined {
                            points[0] = points[n - 1];
                        } else {
                            points[n - 1] = points[0];
                        }
                    }
                    emitter.emit(points, band, truncation_height);
                }
            }
        }

        // Each band's rings, seeded at its level once, those its truncation
        // edge does not already draw, cut to the band where they cross a
        // boundary, exactly: on the boundary, where |f| reaches the band's
        // level beside the chord's crossing.
        if let Some(rings) = &refine.rings {
            let on_boundary =
                |b: f64, q: P2<DomainSpace>, p: P2<DomainSpace>, truncation_height: f64| -> P2<DomainSpace> {
                    if q.x() == b {
                        return q;
                    }
                    if p.x() == b {
                        return p;
                    }
                    let y = q.y() + (p.y() - q.y()) * (b - q.x()) / (p.x() - q.x());
                    let reach = (p.x() - q.x()).abs() + (p.y() - q.y()).abs();
                    let excess = |y: f64| -> f64 { mag(P2::new(b, y)) - truncation_height };
                    let mut s = 1e-6 * reach;
                    while s <= reach {
                        for (u, v) in [(y, y + s), (y - s, y)] {
                            if let Some(c) = self.truncation_crossing_between(
                                P2::new(b, u),
                                excess(u),
                                P2::new(b, v),
                                excess(v),
                                |x| mag(x) - truncation_height,
                            ) {
                                return c;
                            }
                        }
                        s *= 2.0;
                    }
                    P2::new(b, y)
                };
            for span in self.truncation.bands(lo, hi) {
                if !span.height.is_finite() {
                    continue;
                }
                let truncation_height = span.height;
                let edges: Vec<f64> = [span.lo, span.hi].into_iter().filter(|e| boundaries.contains(e)).collect();
                let inside = |p: P2<DomainSpace>| -> bool {
                    !(span.lo > lo && p.x() < span.lo) && !(span.hi < hi && p.x() >= span.hi)
                };
                // A ring is drawn already where the band's truncation edge
                // runs along it: nine in ten of its vertices in the band
                // within twice the tolerance of the edge. Not where some
                // piece turns round its zero, the refiner's test: an arc a cut
                // cell drew of a zero's ring in the band beside turns too
                // little round it, and a loop round two poles and the zero
                // between them turns round it but is not its ring. Nor where
                // the edge comes near it at a point: beside a saddle of |f|
                // just over the band's level, the edge round another zero
                // passes close to the ring and is not it.
                let drawn: Vec<(Vec2, Vec2)> = emitter
                    .pieces
                    .get(&span.band)
                    .map(|ps| ps.iter().flat_map(|p| p.windows(2).map(|w| (w[0].v, w[1].v))).collect())
                    .unwrap_or_default();
                let drawn_already = |ring: &[P2<DomainSpace>]| -> bool {
                    let near = 2.0 * tolerance;
                    let open = if ring.first() == ring.last() { &ring[..ring.len() - 1] } else { ring };
                    let held: Vec<Vec2> = open.iter().copied().filter(|&p| inside(p)).map(|p| p.v).collect();
                    let Some(&first) = held.first() else { return false };
                    let low = held.iter().fold(first, |m, &v| m.min(v)) - Vec2::splat(near);
                    let high = held.iter().fold(first, |m, &v| m.max(v)) + Vec2::splat(near);
                    let close: Vec<&(Vec2, Vec2)> = drawn
                        .iter()
                        .filter(|(a, b)| {
                            let (mx, mn) = (a.max(*b), a.min(*b));
                            mx.x >= low.x && mx.y >= low.y && mn.x <= high.x && mn.y <= high.y
                        })
                        .collect();
                    let covered = held
                        .iter()
                        .filter(|&&v| {
                            close.iter().any(|(a, b)| {
                                let d = *b - *a;
                                let t = swift::max(
                                    0.0,
                                    swift::min(
                                        1.0,
                                        (v - *a).dot(d) / swift::max(d.length_squared(), f64::MIN_POSITIVE),
                                    ),
                                );
                                (v - (*a + t * d)).length() <= near
                            })
                        })
                        .count();
                    10 * covered >= 9 * held.len()
                };
                for ring in rings(truncation_height, &[]) {
                    if !(ring.len() >= 2 && ring.iter().any(|&p| inside(p)) && !drawn_already(&ring)) {
                        continue;
                    }
                    if ring.iter().all(|&p| inside(p)) {
                        emitter.emit(ring, span.band, truncation_height);
                        continue;
                    }
                    // And a part of a ring cut to the band, ending on the
                    // boundary where a piece drawn in the band ends at both
                    // of its ends, is that piece: the level set crosses the
                    // boundary there once. Beside a saddle the grid's line
                    // can be the ring's curve drawn off it, as a chord across
                    // the zero's hole.
                    let ends_drawn =
                        |pieces: &HashMap<usize, Vec<Vec<P2<DomainSpace>>>>, part: &[P2<DomainSpace>]| -> bool {
                            let near = 2.0 * tolerance;
                            let (a, b) = (part[0].v, part[part.len() - 1].v);
                            pieces.get(&span.band).is_some_and(|ps| {
                                ps.iter().any(|p| {
                                    let (s, e) = (p[0].v, p[p.len() - 1].v);
                                    ((a - s).length() <= near && (b - e).length() <= near)
                                        || ((a - e).length() <= near && (b - s).length() <= near)
                                })
                            })
                        };
                    // A closed ring walked from a point outside the band.
                    let mut path = ring.clone();
                    if ring.first() == ring.last()
                        && let Some(s) = ring.iter().position(|&p| !inside(p))
                    {
                        path = ring[s..(ring.len() - 1)].to_vec();
                        path.extend_from_slice(&ring[0..=s]);
                    }
                    let mut part: Vec<P2<DomainSpace>> = Vec::new();
                    for (k, &p) in path.iter().enumerate() {
                        if k > 0 {
                            let q = path[k - 1];
                            let mut holds = inside(q);
                            let mut crossed: Vec<f64> =
                                edges.iter().copied().filter(|&e| (q.x() < e) != (p.x() < e)).collect();
                            crossed.sort_by(|a, b| order((a - q.x()).abs(), (b - q.x()).abs()));
                            for b in crossed {
                                let crossing = on_boundary(b, q, p, truncation_height);
                                if holds {
                                    part.push(crossing);
                                    if part.len() >= 2 && !ends_drawn(emitter.pieces, &part) {
                                        emitter.emit(part.clone(), span.band, truncation_height);
                                    }
                                    part = Vec::new();
                                } else {
                                    part = vec![crossing];
                                }
                                holds = !holds;
                            }
                        }
                        if inside(p) && part.last() != Some(&p) {
                            part.push(p);
                        }
                    }
                    if part.len() >= 2 && !ends_drawn(emitter.pieces, &part) {
                        emitter.emit(part, span.band, truncation_height);
                    }
                }
            }
        }

        // The top of each riser, at f's height truncated by the greater
        // band: along the boundary wherever the lesser band's plateau holds
        // it, from one end of that band's truncation edge on it to the next,
        // or to the domain's edge, read at the middle of each stretch
        // between. In order along x, then y.
        let corner0 = g.position(0, 0);
        let corner1 = g.position(g.width - 1, g.height - 1);
        let (y0, y1) = (swift::min(corner0.y(), corner1.y()), swift::max(corner0.y(), corner1.y()));
        for &b in boundaries {
            let left = self.truncation.height_at_from(b, Side::Left);
            let right = self.truncation.height_at_from(b, Side::Right);
            let (lesser, greater) = (swift::min(left, right), swift::max(left, right));
            if !(lesser.is_finite() && lesser < greater) {
                continue;
            }
            let ends = emitter.ends_on_boundary.get(&double_key(b)).cloned().unwrap_or_default();
            let mut lows: Vec<f64> = ends.iter().filter(|e| e.0 == lesser).map(|e| e.1.y()).collect();
            lows.sort_by(|a, b| order(*a, *b));
            // Where the greater band's truncation edge ends on the boundary
            // the top turns from f's height onto the greater plateau's: a
            // corner, which no midpoint lands on.
            let highs: Vec<f64> = ends.iter().filter(|e| e.0 == greater).map(|e| e.1.y()).collect();
            let mut stops = vec![y0];
            stops.extend(lows.iter().copied().filter(|&y| y > y0 && y < y1));
            stops.push(y1);
            for pair in stops.windows(2) {
                let (a, c) = (pair[0], pair[1]);
                if !(c > a && !(mag(P2::new(b, 0.5 * (a + c))) < lesser)) {
                    continue;
                }
                let top = |y: f64| -> f64 { swift::min(mag(P2::new(b, y)), greater) };
                let profile = |p: P2<DomainSpace>| -> f64 { top(p.y()) };
                // A vertex at every row the boundary crosses, as a wall's top
                // edge has one at every node, and at each corner; then each
                // dip between two found on f, and the rest subdivided onto f:
                // the top edge's own construction (`top_edge`), so a dip
                // narrower than the stretch is not drawn flat across.
                let mut ys = vec![a, c];
                ys.extend(rows.iter().copied().filter(|&y| y > a && y < c));
                ys.extend(highs.iter().copied().filter(|&y| y > a && y < c));
                ys.sort_by(|a, b| order(*a, *b));
                let mut nodes: Vec<P3<WorldSpace>> = Vec::new();
                for y in ys {
                    if nodes.last().map(|n| n.y()) != Some(y) {
                        nodes.push(P3::new(b, y, top(y)));
                    }
                }
                // To half the tolerance, so the curve meets it with room: a
                // chord's midpoint is not always where it departs furthest.
                // And ten halvings deep, a thousandth of a row: beside a
                // zero of f a hair from the boundary, |f| along it turns as
                // sharply as a hyperbola at its vertex, and at the top edge's
                // five the chords stopped short there.
                let curve =
                    self.subdivided_profile(&self.with_minima(&nodes, &profile), &profile, 0.5 * refine.tolerance, 10);
                riser_tops.push((b, curve));
            }
        }
        emitter.out
    }

    /// For a truncation edge vertex on the domain's edge, the truncation
    /// crossing between the two grid nodes beside it on that edge, by the
    /// top edge's bisection; `None` for a vertex off the edge or with no
    /// crossing beside it (and where Swift's `Int` would trap, on a vertex
    /// whose grid index is not a number).
    pub fn on_boundary_truncation_crossing(
        &self,
        p: P2<DomainSpace>,
        magnitude: &dyn Fn(P2<DomainSpace>) -> f64,
    ) -> Option<P2<DomainSpace>> {
        let g = &*self.height;
        let eps = self.edge_tolerance();
        let f = g.index(p);
        let on_row = (p.y() - g.domain.imag.lo).abs() <= eps || (p.y() - g.domain.imag.hi).abs() <= eps;
        let on_column = (p.x() - g.domain.real.lo).abs() <= eps || (p.x() - g.domain.real.hi).abs() <= eps;
        if !(on_row || on_column) {
            return None;
        }
        let excess = |q: P2<DomainSpace>| -> f64 { magnitude(q) - self.truncation_height_at(q.x()) };
        let clamp = |v: f64, count: usize| -> Option<usize> {
            let i = swift::int(v.floor())?;
            usize::try_from(i.max(0).min(count as i64 - 2)).ok()
        };
        let (a, b) = if on_row {
            let j = if (p.y() - g.domain.imag.lo).abs() <= eps { 0 } else { g.height - 1 };
            let i = clamp(f.x, g.width)?;
            (g.position(i, j), g.position(i + 1, j))
        } else {
            let i = if (p.x() - g.domain.real.lo).abs() <= eps { 0 } else { g.width - 1 };
            let j = clamp(f.y, g.height)?;
            (g.position(i, j), g.position(i, j + 1))
        };
        self.truncation_crossing_between(a, excess(a), b, excess(b), excess)
    }
}

// MARK: - the graph's trim loops

/// The rectangle a set of plateaus' loops is closed along
/// (`PlateauLoops.Rectangle`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    pub x0: f64,
    pub x1: f64,
    pub y0: f64,
    pub y1: f64,
}

impl Rectangle {
    pub fn perimeter(&self) -> f64 {
        2.0 * (self.x1 - self.x0) + 2.0 * (self.y1 - self.y0)
    }

    /// The corners, counter-clockwise from (x0, y0), with the distance round
    /// the edge to each.
    pub fn corners(&self) -> [(f64, P2<DomainSpace>); 4] {
        let (w, h) = (self.x1 - self.x0, self.y1 - self.y0);
        [
            (0.0, P2::new(self.x0, self.y0)),
            (w, P2::new(self.x1, self.y0)),
            (w + h, P2::new(self.x1, self.y1)),
            (2.0 * w + h, P2::new(self.x0, self.y1)),
        ]
    }

    /// The distance round the edge, counter-clockwise from (x0, y0), of a
    /// point within `eps` of it; `None` for one off it.
    pub fn distance(&self, p: P2<DomainSpace>, eps: f64) -> Option<f64> {
        let (w, h) = (self.x1 - self.x0, self.y1 - self.y0);
        if !(p.x() >= self.x0 - eps && p.x() <= self.x1 + eps && p.y() >= self.y0 - eps && p.y() <= self.y1 + eps) {
            return None;
        }
        let clamp = |v: f64, hi: f64| swift::min(swift::max(v, 0.0), hi);
        if (p.y() - self.y0).abs() <= eps {
            return Some(clamp(p.x() - self.x0, w));
        }
        if (p.x() - self.x1).abs() <= eps {
            return Some(w + clamp(p.y() - self.y0, h));
        }
        if (p.y() - self.y1).abs() <= eps {
            return Some(w + h + clamp(self.x1 - p.x(), w));
        }
        if (p.x() - self.x0).abs() <= eps {
            return Some(2.0 * w + h + clamp(self.y1 - p.y(), h));
        }
        None
    }

    /// The point of the edge a distance round it.
    pub fn point(&self, s: f64) -> P2<DomainSpace> {
        let (w, h) = (self.x1 - self.x0, self.y1 - self.y0);
        let s = real::fmod(s, self.perimeter());
        if s < w {
            return P2::new(self.x0 + s, self.y0);
        }
        if s < w + h {
            return P2::new(self.x1, self.y0 + (s - w));
        }
        if s < 2.0 * w + h {
            return P2::new(self.x1 - (s - w - h), self.y1);
        }
        P2::new(self.x0, self.y1 - (s - 2.0 * w - h))
    }
}

/// One set of plateaus' loops as they are closed: the truncation edge's
/// pieces over a rectangle, the domain or one band of it, joined end to end
/// along the rectangle's edge wherever the plateau holds it.
pub struct PlateauLoops<'a> {
    /// The rectangle the pieces are closed along.
    pub rectangle: Rectangle,
    /// Whether a point of the rectangle's edge is plateau.
    pub plateau: Box<dyn Fn(P2<DomainSpace>) -> bool + 'a>,
    pub pieces: Vec<Vec<P2<DomainSpace>>>,
}

impl<'a> PlateauLoops<'a> {
    pub fn new(rectangle: Rectangle, plateau: Box<dyn Fn(P2<DomainSpace>) -> bool + 'a>) -> PlateauLoops<'a> {
        PlateauLoops { rectangle, plateau, pieces: Vec::new() }
    }

    /// The pieces joined into loops: a closed piece is a loop of its own; an
    /// open one ending on the rectangle's edge is joined to the next end
    /// round it, through the corners between, wherever the plateau holds the
    /// stretch between, read at its middle; and pieces meeting end to end
    /// are joined. With no end on the edge the whole edge is a loop where it
    /// is plateau. Each loop ends where it begins; a chain that cannot be
    /// closed, at an end off the edge with nothing there, is kept as it is,
    /// and counted in `open`.
    pub fn loops(&self, eps: f64) -> (Vec<Vec<P2<DomainSpace>>>, usize) {
        let rectangle = self.rectangle;
        let mut loops: Vec<Vec<P2<DomainSpace>>> = Vec::new();
        let mut open = 0;
        let mut edges: Vec<Vec<P2<DomainSpace>>> = Vec::new();
        for piece in self.pieces.iter().filter(|p| p.len() >= 2) {
            if piece.first() == piece.last() {
                loops.push(piece.clone());
            } else {
                edges.push(piece.clone());
            }
        }
        let mut ends: Vec<(f64, P2<DomainSpace>)> = Vec::new();
        for piece in &edges {
            for p in [piece[0], piece[piece.len() - 1]] {
                if let Some(s) = rectangle.distance(p, eps) {
                    ends.push((s, p));
                }
            }
        }
        ends.sort_by(|a, b| order(a.0, b.0).then(order(a.1.x(), b.1.x())).then(order(a.1.y(), b.1.y())));
        let perimeter = rectangle.perimeter();
        let corners = rectangle.corners();
        if ends.is_empty() {
            if (self.plateau)(rectangle.point(0.5 * (rectangle.x1 - rectangle.x0))) {
                let mut lp: Vec<P2<DomainSpace>> = corners.iter().map(|c| c.1).collect();
                lp.push(corners[0].1);
                loops.push(lp);
            }
        } else {
            for k in 0..ends.len() {
                let (a, next) = (ends[k], ends[(k + 1) % ends.len()]);
                let sb = if k + 1 < ends.len() { next.0 } else { next.0 + perimeter };
                if !(sb > a.0 && (self.plateau)(rectangle.point(0.5 * (a.0 + sb)))) {
                    continue;
                }
                let mut edge = vec![a.1];
                edge.extend(
                    corners
                        .iter()
                        .copied()
                        .chain(corners.iter().map(|c| (c.0 + perimeter, c.1)))
                        .filter(|c| c.0 > a.0 && c.0 < sb)
                        .map(|c| c.1),
                );
                edge.push(next.1);
                edges.push(edge);
            }
        }
        // Chained end to end, each edge once, in the order they come.
        let mut at: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            for p in [e[0], e[e.len() - 1]] {
                if let Some(key) = point_key(p) {
                    at.entry(key).or_default().push(i);
                }
            }
        }
        let next_unused = |used: &[bool], p: P2<DomainSpace>| -> Option<usize> {
            at.get(&point_key(p)?)?.iter().copied().find(|&j| !used[j])
        };
        let mut used = vec![false; edges.len()];
        for i in 0..edges.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            let mut chain = edges[i].clone();
            while chain.first() != chain.last() {
                let Some(j) = next_unused(&used, chain[chain.len() - 1]) else { break };
                used[j] = true;
                let e = &edges[j];
                let tail = chain[chain.len() - 1];
                if e[0] == tail {
                    chain.extend_from_slice(&e[1..]);
                } else {
                    chain.extend(e.iter().rev().skip(1).copied());
                }
            }
            // Open at the far end: carried back from the near one as far as
            // it goes.
            while chain.first() != chain.last() {
                let Some(j) = next_unused(&used, chain[0]) else { break };
                used[j] = true;
                let e = &edges[j];
                let mut head: Vec<P2<DomainSpace>> =
                    if e[e.len() - 1] == chain[0] { e.clone() } else { e.iter().rev().copied().collect() };
                head.pop();
                head.extend(chain);
                chain = head;
            }
            if chain.first() != chain.last() {
                open += 1;
            }
            loops.push(chain);
        }
        (loops, open)
    }
}
