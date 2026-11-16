//! The traversal up a line of sight through a heightfield's solid: Swift's
//! `GridTraversal`, from `Sources/KurvenCore/Scene.swift`, in a module of its
//! own because `scene.rs` holds the rest of that file.

use kurven_math::swift;

use crate::manifest::{Affine2, Truncation, Walls};
use crate::scene::Heightfield;
use crate::simd::{Vec2, Vec3};
use crate::spaces::{DomainSpace, P2, P3, Transform, ViewSpace, WorldSpace};
use crate::surface::{DrawnSurface, Magnitude, Surface};

/// Where a traversal reports what it made of each sample: Swift's `trace`
/// closure, for asking why a vertex is hidden.
pub type Trace<'a> = Option<&'a mut dyn FnMut(String)>;

/// Per tile: world to the tile's grid index coordinates, and the grid's
/// last index each way.
#[derive(Debug, Clone, Copy)]
struct TileGrid {
    /// The linear part by its rows, as Swift's `simd_double2x2(rows:)`.
    row0: Vec2,
    row1: Vec2,
    offset: Vec2,
    extent: Vec2,
}

impl TileGrid {
    /// `linear * v`, simd's matrix-vector product, unfused.
    fn linear(&self, v: Vec2) -> Vec2 {
        Vec2::new(self.row0.x * v.x + self.row0.y * v.y, self.row1.x * v.x + self.row1.y * v.y)
    }
}

/// A band boundary Re z = b the truncation steps at inside the domain, as a
/// level of the grid's u, with the lesser and greater band heights either
/// side: a riser on every copy.
#[derive(Debug, Clone, Copy)]
struct BandStep {
    b: f64,
    level: f64,
    lesser: f64,
    greater: f64,
}

/// The line's height over the drawn surface at a distance along it: over
/// the interpolated triangle and over the truncation, the triangle's cosine,
/// and, with the function, the rate the gap over f changes along the line.
#[derive(Debug, Clone, Copy)]
struct Sample {
    over_triangle: f64,
    over_truncation: f64,
    cosine: f64,
    rate: f64,
}

/// A traversal up a line of sight through a heightfield's solid: the question the
/// depth pass answers for every pixel, asked for one point, exactly.
///
/// Exactly, because the drawn surface is piecewise linear: the grid's
/// cells, two triangles each, truncated. Along a straight line the gap between
/// the line and that surface is therefore linear between the places where
/// the line crosses a grid row, a grid column or a cell's diagonal, and
/// its least value lies at one of them. So the traversal visits those crossings
/// and no other points -- in the grid's own index coordinates, where the
/// rows, columns and diagonals are the integer levels of `u`, `v` and
/// `u + v` -- and a point is hidden when at some crossing beyond the depth tolerance
/// the line is under the surface. Nothing is sampled: a sliver thinner than
/// any step is still a crossing. The answer is a function of the geometry
/// alone, which is what makes a bake the same drawing at every depth
/// resolution.
///
/// The depth tolerance means two things, and the traversal keeps them apart. A point of
/// curves lie on a surface it was placed on by the function or by the grid,
/// and the drawn surface -- the grid's triangles -- misses that by an
/// interpolation gap, so the point may start a little inside the solid.
/// That is forgiven *perpendicular* to the triangle, by the depth tolerance: a vertical
/// gap times the triangle's cosine, still linear along the line over one
/// triangle. Measured vertically it would be a horizontal tolerance of
/// depth tolerance / slope, and on a cone of slope 24 the rings came out dashed;
/// measured along the line of sight it forgives nothing at a fold, where the
/// line runs along the surface for a long way within a hair of it, and the
/// folds vanished. Then, once the line has been outside the solid, anything
/// it enters is an occluder and is tested *along the line of sight*, as the
/// depth buffer tested it: hidden when the line is still inside the solid
/// at the depth tolerance's distance toward the eye, or beyond. A wedge of rock
/// that is thin across but long along the line -- the side of a pit seen
/// through its notch -- hides what is behind it, as it should.
///
/// A point strictly over the domain is tested where it stands as well. A
/// point on a wall -- at the domain's edge, or on a notch in the footprint
/// strictly over it -- is on the solid's boundary and is tested only from
/// where the line goes next, so the bottom edge of a wall that faces the eye,
/// whose line leaves the solid at once, is seen.
///
/// Built once per camera, since it scans the samples for the solid's extent
/// and inverts the tiles.
#[derive(Debug, Clone)]
pub struct GridTraversal {
    heightfield: Heightfield,
    pub toward: Vec3,
    /// The hidden-line depth tolerance, in view depth. Infinite disarms the traversal.
    pub depth_tolerance: f64,
    /// The depth tolerance as a distance along the line of sight: an occluder must
    /// stand in front by more than this.
    pub reach: f64,
    /// The drawn grid's spacing, the larger of the two directions.
    pub cell: f64,
    pub top: f64,
    pub floor: f64,
    /// The floor the walls stand on, a perimeter's base, or without one the
    /// lowest point drawn: under it is outside the slab, for a point of
    /// another solid (`unoccluded`).
    pub base: f64,
    /// Per tile: world to the tile's grid index coordinates, and the
    /// grid's last index each way.
    grids: Vec<TileGrid>,
    /// Per tile: world back to the fundamental domain, for the nearest-tile
    /// height read (`Surface::in_tile`, with the inverses taken once).
    backs: Vec<Affine2>,
    step: usize,
    /// The band boundaries Re z = b the truncation steps at inside the
    /// domain, each as a level of the grid's u, with the lesser and greater
    /// band heights either side: a riser on every copy.
    steps: Vec<BandStep>,
}

impl GridTraversal {
    pub fn new(
        heightfield: Heightfield,
        view: &Transform<WorldSpace, ViewSpace>,
        depth_tolerance: f64,
    ) -> GridTraversal {
        let toward = -view.projection_direction();
        // View depth per unit of travel along the line of sight: the camera's
        // third row dotted with the direction, which under a sheared drawing
        // camera is not one.
        let c = view.m.columns;
        let per_unit = Vec3::new(c[0].z, c[1].z, c[2].z).dot(toward).abs();
        let g = &*heightfield.surface.height;
        let step = heightfield.step.max(1);
        let nx = g.width.div_ceil(step);
        let ny = g.height.div_ceil(step);
        let dx = g.domain.real.length() / g.width.saturating_sub(1).max(1) as f64 * step as f64;
        let dy = g.domain.imag.length() / g.height.saturating_sub(1).max(1) as f64 * step as f64;
        let cell = swift::max(dx.abs(), dy.abs());
        let real = g.domain.real;
        let spans = heightfield.surface.truncation.bands(swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
        let steps = spans
            .iter()
            .zip(spans.iter().skip(1))
            .filter_map(|(left, right)| {
                let lesser = swift::min(left.height, right.height);
                let greater = swift::max(left.height, right.height);
                if !(lesser.is_finite() && lesser < greater) {
                    return None;
                }
                Some(BandStep { b: left.hi, level: (left.hi - real.lo) / dx, lesser, greater })
            })
            .collect();
        // The depth tolerance forgives a curve its distance from the solid it is
        // tested against. Against the grid that is the chord's departure
        // from f, the drawing's depth tolerance. Against f it is the refiner's own
        // tolerance: a vertex is on f to a hundredth of it, and a chord
        // between two vertices departs from the curve by at most the
        // tolerance of a cell, measured across the curve -- which is the
        // perpendicular distance the traversal tests by, since along a level
        // line f is flat. A drawing-scale depth tolerance would let a ring show
        // through a wall for the depth tolerance's reach past the top edge.
        let bounded = match &heightfield.refine {
            Some(r) => swift::min(depth_tolerance, r.tolerance * cell),
            None => depth_tolerance,
        };
        let reach = bounded / swift::max(per_unit, 1e-12);
        let mut top = f64::NEG_INFINITY;
        let mut floor = f64::INFINITY;
        heightfield.for_each_sample(|p| {
            top = swift::max(top, p.z());
            floor = swift::min(floor, p.z());
        });
        for v in &heightfield.occluder.vertices {
            floor = swift::min(floor, v.z());
        }
        let base = match &heightfield.walls {
            Walls::Perimeter { base, .. } => *base,
            _ => floor,
        };
        let mut grids = Vec::new();
        let mut backs = Vec::new();
        for tile in &heightfield.tiles {
            let back = tile.inverse().unwrap_or(Affine2::IDENTITY);
            backs.push(back);
            grids.push(TileGrid {
                row0: Vec2::new(back.a / dx, back.b / dx),
                row1: Vec2::new(back.c / dy, back.d / dy),
                offset: Vec2::new((back.tx - g.domain.real.lo) / dx, (back.ty - g.domain.imag.lo) / dy),
                extent: Vec2::new((nx as f64) - 1.0, (ny as f64) - 1.0),
            });
        }
        GridTraversal {
            heightfield,
            toward,
            depth_tolerance: bounded,
            reach,
            cell,
            top,
            floor,
            base,
            grids,
            backs,
            step,
            steps,
        }
    }

    pub fn heightfield(&self) -> &Heightfield {
        &self.heightfield
    }

    /// The same traversal for one thread: its heightfield's function forked.
    pub fn forked(&self) -> GridTraversal {
        GridTraversal { heightfield: self.heightfield.forked(), ..self.clone() }
    }

    /// The drawn surface over a world point, read through the nearest tile:
    /// the grid's interpolated height before the truncation, the truncation there, and
    /// the cosine of the triangle's tilt. The drawn height is the lesser of the
    /// first two.
    pub fn drawn_surface(&self, p: Vec2) -> DrawnSurface {
        let s = &self.heightfield.surface;
        if self.backs.len() <= 1 {
            return self.surface(P2::new(p.x, p.y));
        }
        let d = s.domain();
        let (xlo, xhi) = (swift::min(d.real.lo, d.real.hi), swift::max(d.real.lo, d.real.hi));
        let (ylo, yhi) = (swift::min(d.imag.lo, d.imag.hi), swift::max(d.imag.lo, d.imag.hi));
        let mut best = P2::<DomainSpace>::new(p.x, p.y);
        let mut best_distance = f64::INFINITY;
        for back in &self.backs {
            let q = back.apply(P3::new(p.x, p.y, 0.0));
            let distance = swift::max(xlo - q.x(), 0.0)
                + swift::max(q.x() - xhi, 0.0)
                + swift::max(ylo - q.y(), 0.0)
                + swift::max(q.y() - yhi, 0.0);
            if distance < best_distance {
                best_distance = distance;
                best = P2::new(q.x(), q.y());
                if distance == 0.0 {
                    break;
                }
            }
        }
        self.surface(best)
    }

    /// The surface over a point of the fundamental tile: f's own height and
    /// the cosine of its tilt, by differences a step small against the cell,
    /// when the function is at hand; the grid's triangle otherwise. The curves
    /// is placed on f, so the solid it is tested against is f: a ring seen
    /// through a notch is hidden where f's top edge hides it, not where the
    /// chord across the nearest two nodes would.
    pub fn surface(&self, q: P2<DomainSpace>) -> DrawnSurface {
        let s = &self.heightfield.surface;
        let Some(magnitude) = self.heightfield.refine.as_ref().and_then(|r| r.magnitude.as_ref()) else {
            return s.drawn_surface(q, self.step);
        };
        let (u, slope) = self.function_surface(q, magnitude);
        let cosine = 1.0 / (1.0 + slope.dot(slope)).sqrt();
        DrawnSurface {
            interpolated: u,
            truncation_height: s.truncation_height_at(q.x()),
            cosine: if cosine.is_nan() { 0.0 } else { cosine },
        }
    }

    /// f's height over a point of the fundamental tile and its slope there,
    /// by differences a step small against the cell.
    pub fn function_surface(&self, q: P2<DomainSpace>, magnitude: &Magnitude) -> (f64, Vec2) {
        let u = magnitude(q);
        let e = 1e-4 * self.cell;
        // One-sided, from the value already in hand: three evaluations of
        // f per sample rather than five, at a step small enough that the
        // difference from the centred slope is below what the cosine or
        // the rate is used for.
        let hx = (magnitude(P2::new(q.x() + e, q.y())) - u) / e;
        let hy = (magnitude(P2::new(q.x(), q.y() + e)) - u) / e;
        (if u.is_nan() { f64::INFINITY } else { u }, Vec2::new(hx, hy))
    }

    /// Whether nothing of the solid stands between `p` and the eye: Swift's
    /// `unoccluded(_:)` with its defaults, forgiven and of this solid.
    pub fn unoccluded_default(&self, p: P3<WorldSpace>) -> bool {
        self.unoccluded(p, true, false, None)
    }

    /// Whether nothing of the solid stands between `p` and the eye. A point
    /// not `forgiven` lies on a wall exactly, with no interpolation gap to
    /// forgive: it is on the solid's boundary, not tested where it stands,
    /// and outside from the start, so anything its line enters hides it. A
    /// point `elsewhere`, of another solid in a composite, is forgiven
    /// nothing either, and is tested where it stands as well: strictly
    /// inside this solid -- over the domain and the footprint, above the
    /// base and under the surface, each by more than a rounding -- it is
    /// hidden; on a wall, the floor or the top it is on the boundary, and
    /// hidden only where its line goes in. For such a point the slab has a
    /// floor: under the base is outside it, and a line of sight that comes
    /// up through the floor enters it there. With `trace`, every sample the
    /// traversal takes and what it made of it, for asking why a vertex is
    /// hidden.
    pub fn unoccluded(&self, p: P3<WorldSpace>, forgiven: bool, elsewhere: bool, mut trace: Trace<'_>) -> bool {
        let hf = &self.heightfield;
        let d = self.toward;
        let pxy = Vec2::new(p.x(), p.y());
        let dxy = Vec2::new(d.x, d.y);
        let cell = self.cell;
        // How far up the line there can be anything: once above the highest
        // sample, or below the lowest, nothing is.
        let mut t_max = f64::INFINITY;
        if d.z > 0.0 {
            t_max = swift::max((self.top - p.z()) / d.z, 0.0);
        } else if d.z < 0.0 {
            t_max = swift::max((self.floor - p.z()) / d.z, 0.0);
        } else if p.z() > self.top || p.z() < self.floor {
            return true;
        }
        let t_eps = 1e-9 * cell;
        // The line's height over the drawn surface at distance `t` along
        // it: over the interpolated triangle and over the truncation, and the triangle's
        // cosine; None where the point is outside the region, over which
        // nothing is drawn.
        // With the function, the sample also carries the rate the gap over
        // f changes along the line, from the same three evaluations, for
        // finding where the gap peaks or dips between two crossings.
        // The slope is f's, read in the domain, so the line's direction is
        // carried back into the domain through the tile under the point
        // before the two are multiplied: a reflected or turned copy's line
        // runs another way across f than across the world.
        let magnitude = hf.refine.as_ref().and_then(|r| r.magnitude.clone());
        let gaps = |t: f64| -> Option<Sample> {
            let q = pxy + t * dxy;
            if !hf.region.contains(P2::new(q.x, q.y)) {
                return None;
            }
            let z = p.z() + t * d.z;
            if elsewhere && z < self.base - t_eps {
                return None;
            }
            if let Some(magnitude) = &magnitude {
                let (dq, back) = hf.surface.tile_under(P2::new(q.x, q.y), &hf.tiles);
                let (u, slope) = self.function_surface(dq, magnitude);
                let cosine = 1.0 / (1.0 + slope.dot(slope)).sqrt();
                let truncation_height = hf.surface.truncation_height_at(dq.x());
                let along = Vec2::new(back.a * dxy.x + back.b * dxy.y, back.c * dxy.x + back.d * dxy.y);
                return Some(Sample {
                    over_triangle: z - u,
                    over_truncation: z - truncation_height,
                    cosine: if cosine.is_nan() { 0.0 } else { cosine },
                    rate: d.z - slope.dot(along),
                });
            }
            let drawn = self.drawn_surface(q);
            Some(Sample {
                over_triangle: z - drawn.interpolated,
                over_truncation: z - drawn.truncation_height,
                cosine: drawn.cosine,
                rate: 0.0,
            })
        };
        // The truncation's bands, if any, as levels of the grid's u where the
        // truncation steps: a crossing of the drawn surface too.
        let mut band_edges: Vec<f64> = Vec::new();
        if let Truncation::RealBands { bands, .. } = &hf.surface.truncation {
            let g = &*hf.surface.height;
            let dx = g.domain.real.length() / g.width.saturating_sub(1).max(1) as f64 * self.step as f64;
            band_edges = bands.iter().map(|b| (b.below - g.domain.real.lo) / dx).collect();
        }
        // Every crossing of a grid row, column or diagonal -- the
        // integer levels of u, v and u + v -- and of a truncation band's edge, over
        // every tile the line passes, with the ends of each passage; and
        // the depth tolerance's reach, which is where occlusion starts to count.
        let mut ts: Vec<f64> = Vec::new();
        let mut strictly_inside = false;
        let mut passages: Vec<(f64, f64, Affine2)> = Vec::new();
        // Where the line passes through a band boundary's riser: it crosses
        // Re z = b between the two sides' tops there, under the higher and
        // over the lower. The sample on the boundary reads the band beyond
        // it, and the solid steps up the riser between it and its neighbours,
        // so the line is in the solid at the crossing, however soon it
        // leaves: each is tested there, as the gap over the higher side's
        // top. (At a tile boundary the samples just inside each copy read
        // both sides.) Not where the line starts on the riser's line, within
        // half a billionth of a cell, as `Heightfield::riser_facing`
        // counts it -- a riser's top, a truncation edge's end at it, a wall's
        // top edge stepping across it -- which its own riser does not hide.
        let mut risers: Vec<(f64, Sample)> = Vec::new();
        let s: &Surface = &hf.surface;
        let region = &hf.region;
        let side = 1e-9 * s.cell();
        let z_eps = 1e-9 * cell;
        for (grid, back) in self.grids.iter().zip(&self.backs) {
            let u0 = grid.linear(pxy) + grid.offset;
            let du = grid.linear(dxy);
            let mut ta = 0.0;
            let mut tb = t_max;
            let mut misses = false;
            for axis in 0..2 {
                if du[axis].abs() < 1e-15 {
                    if u0[axis] < 0.0 || u0[axis] > grid.extent[axis] {
                        misses = true;
                    }
                    continue;
                }
                let t1 = -u0[axis] / du[axis];
                let t2 = (grid.extent[axis] - u0[axis]) / du[axis];
                ta = swift::max(ta, swift::min(t1, t2));
                tb = swift::min(tb, swift::max(t1, t2));
            }
            if misses || !(ta <= tb) || !tb.is_finite() {
                continue;
            }
            passages.push((ta, tb, *back));
            if u0.x > 1e-9 && u0.x < grid.extent.x - 1e-9 && u0.y > 1e-9 && u0.y < grid.extent.y - 1e-9 {
                strictly_inside = true;
            }
            ts.push(ta);
            ts.push(tb);
            // The diagonals split the grid's cells into the triangles the
            // depth pass draws; f has no triangles, and its brackets are the
            // rows and columns.
            let families = if magnitude.is_none() { 3 } else { 2 };
            for family in 0..families {
                let f0 = if family < 2 { u0[family] } else { u0.x + u0.y };
                let df = if family < 2 { du[family] } else { du.x + du.y };
                if !(df.abs() > 1e-15) {
                    continue;
                }
                let fa = f0 + ta * df;
                let fb = f0 + tb * df;
                // Swift's `Int` traps on a crossing that is not a number;
                // the port passes over the family there.
                if let (Some(k_lo), Some(k_hi)) =
                    (swift::int((swift::min(fa, fb) + 1e-9).ceil()), swift::int((swift::max(fa, fb) - 1e-9).floor()))
                    && k_lo <= k_hi
                {
                    for k in k_lo..=k_hi {
                        ts.push((k as f64 - f0) / df);
                    }
                }
                if family == 0 {
                    for &level in &band_edges {
                        let t = (level - f0) / df;
                        if t > ta && t < tb {
                            ts.push(t);
                        }
                    }
                }
            }
            if self.reach.is_finite() && self.reach > ta && self.reach < tb {
                ts.push(self.reach);
            }
            // A band boundary Re z = b in this copy's domain, crossed at a
            // point carried back through its tile: the riser there stands
            // from min(|f|, h) on the lesser band's side to min(|f|, h) on
            // the greater's, where the footprint holds both sides.
            let start = back.apply(P3::new(p.x(), p.y(), 0.0));
            let normal = Vec2::new(back.a, back.b).normalized();
            for step in &self.steps {
                if !(du.x.abs() > 1e-15 && (start.x() - step.b).abs() >= 0.5 * side) {
                    continue;
                }
                let t = (step.level - u0.x) / du.x;
                if !(t > ta && t < tb && t > t_eps) {
                    continue;
                }
                let q = pxy + t * dxy;
                let z = p.z() + t * d.z;
                if !(region.contains(P2::from_vec(q - side * normal))
                    && region.contains(P2::from_vec(q + side * normal)))
                {
                    continue;
                }
                let drawn = self.surface(s.onto_domain(P2::new(step.b, back.apply(P3::new(q.x, q.y, 0.0)).y())));
                if !(z > swift::min(drawn.interpolated, step.lesser) + z_eps
                    && z < swift::min(drawn.interpolated, step.greater) - z_eps)
                {
                    continue;
                }
                risers.push((
                    t,
                    Sample {
                        over_triangle: z - drawn.interpolated,
                        over_truncation: z - step.greater,
                        cosine: drawn.cosine,
                        rate: 0.0,
                    },
                ));
            }
        }
        // Where the line leaves one copy for another across a tile boundary
        // and the copies' tops there differ, a sample just inside each, read
        // through its own tile: the sample on the tile boundary itself reads
        // whichever tile comes first, and between the two the solid drops
        // down a riser, so a line passing the tile boundary under the higher
        // top is in it.
        for a in &passages {
            for b in &passages {
                if !((a.1 - b.0).abs() <= t_eps && b.1 > b.0) {
                    continue;
                }
                let q = pxy + a.1 * dxy;
                let tops = [a.2, b.2].map(|back| {
                    let drawn = self.surface(s.onto_domain(back.apply(P3::new(q.x, q.y, 0.0)).xy()));
                    swift::min(drawn.interpolated, drawn.truncation_height)
                });
                if !(Surface::stands(tops[0], tops[1]) || Surface::stands(tops[1], tops[0])) {
                    continue;
                }
                if a.1 - 2.0 * t_eps > a.0 {
                    ts.push(a.1 - 2.0 * t_eps);
                }
                if b.0 + 2.0 * t_eps < b.1 {
                    ts.push(b.0 + 2.0 * t_eps);
                }
            }
        }
        // A point of another solid: hidden where its line crosses the base,
        // which is the slab's floor, and sampled a hair along the line from
        // a start on the boundary, where the line goes in or out.
        let mut start_inside = false;
        if elsewhere && !ts.is_empty() {
            // Only over a copy's grid, where the slab is.
            let over_grid = |t: f64| passages.iter().any(|&(ta, tb, _)| t >= ta && t <= tb);
            // The floor is a sheet of the slab's boundary: a line that
            // crosses it over the grid and the footprint goes into the slab
            // however thin the slab is there -- where the surface comes down
            // to the base -- as the closed-form cut has it (`Slab`).
            if d.z.abs() > 1e-15 {
                let floor_crossing = (self.base - p.z()) / d.z;
                if floor_crossing > t_eps
                    && over_grid(floor_crossing)
                    && region.contains(P2::new(pxy.x + floor_crossing * dxy.x, pxy.y + floor_crossing * dxy.y))
                {
                    if let Some(trace) = trace.as_mut() {
                        trace(format!("t {} cells: through the floor", g4(floor_crossing / cell)));
                    }
                    return false;
                }
            }
            if over_grid(2.0 * t_eps) {
                ts.push(2.0 * t_eps);
            }
            let e = side;
            start_inside = strictly_inside
                && p.z() > self.base + t_eps
                && [Vec2::new(e, 0.0), Vec2::new(-e, 0.0), Vec2::new(0.0, e), Vec2::new(0.0, -e)]
                    .iter()
                    .all(|o| region.contains(P2::new(pxy.x + o.x, pxy.y + o.y)));
        }
        risers.sort_by(|a, b| swift_order(a.0, b.0));
        if ts.is_empty() {
            return true;
        }
        ts.sort_by(|a, b| swift_order(*a, *b));
        let mut unique: Vec<f64> = Vec::new();
        for &t in &ts {
            if unique.last().is_none_or(|&last| t - last > t_eps) {
                unique.push(t);
            }
        }
        let ts = unique;
        let values: Vec<Option<Sample>> = ts.iter().map(|&t| gaps(t)).collect();

        // The walk. In its own surface's neighbourhood -- from the start,
        // for as long as it has not yet been outside -- the line is
        // forgiven the depth tolerance perpendicular to the triangle over it. Once it
        // has been outside, being inside at or beyond the depth tolerance's reach is
        // occlusion. A point not forgiven has been outside from the start, and
        // being inside anywhere is occlusion.
        let mut walk = Walk {
            outside_yet: !forgiven || elsewhere,
            start_forgiven: false,
            forgiven,
            elsewhere,
            strictly_inside,
            start_inside,
            on_f: magnitude.is_some(),
            t_eps,
            cell,
            depth_tolerance: self.depth_tolerance,
            reach: self.reach,
        };
        if let Some(trace) = trace.as_mut() {
            trace(format!(
                "from ({:.4}, {:.4}, {:.4}), {} samples to {} cells",
                p.x(),
                p.y(),
                p.z(),
                ts.len(),
                g4(ts.last().copied().unwrap_or(0.0) / cell)
            ));
        }
        let mut previous_middle: Option<Sample> = None;
        let mut next_riser = 0;
        for k in 0..ts.len() {
            let t = ts[k];
            let g = values[k];
            // A point on a grid line lies under two triangles: tested by
            // the triangle behind it and the one ahead. Under f the cosine is
            // continuous and the sample's own serves. The sample at the
            // middle of the interval ahead is taken once, for this and for
            // the search of the interval.
            let middle = if k + 1 < ts.len() { gaps(0.5 * (t + ts[k + 1])) } else { None };
            if magnitude.is_none() {
                let cosines: Vec<f64> =
                    [previous_middle.map(|m| m.cosine), middle.map(|m| m.cosine)].into_iter().flatten().collect();
                let tried = if cosines.is_empty() { vec![g.map_or(1.0, |g| g.cosine)] } else { cosines };
                for cosine in tried {
                    if walk.hidden(t, g, cosine, &mut trace) {
                        return false;
                    }
                }
            } else if walk.hidden(t, g, g.map_or(1.0, |g| g.cosine), &mut trace) {
                return false;
            }
            previous_middle = middle;
            // The risers crossed here: each on a band boundary, and so a
            // sample of its own, or one within rounding.
            while next_riser < risers.len() && risers[next_riser].0 <= t + t_eps {
                let (tr, gr) = risers[next_riser];
                if let Some(trace) = trace.as_mut() {
                    trace(format!("t {} cells: through a riser", g4(tr / cell)));
                }
                if walk.hidden(tr, Some(gr), gr.cosine, &mut trace) {
                    return false;
                }
                next_riser += 1;
            }
            // With the function, the gap over f between this crossing and
            // the next is not the triangle's straight line: it can rise above
            // the surface's skin and fall back under it, or dip under and
            // come back out, between two samples that see neither. The gap
            // and its rate are read at both ends and at the middle; a
            // greatest value is bracketed where the gap rises at one end and
            // falls at the other, or the middle is above both ends, and a
            // least value likewise; each is found by golden section. The
            // greatest, if its clearance exceeds the depth tolerance, is the line
            // leaving the surface; the least is tested where it is, after
            // the greatest if that came first. A point just in front of a
            // fold is the case: its line leaves the surface, passes under it
            // and comes out again before the cut.
            if let (true, true, Some(ra), Some(rb), Some(rm)) =
                (magnitude.is_some(), k + 1 < ts.len(), g, values.get(k + 1).copied().flatten(), middle)
            {
                let ta = t;
                let tb = ts[k + 1];
                let tm = 0.5 * (ta + tb);
                let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
                let gap_over_f = |x: f64| -> f64 { gaps(x).map_or(f64::INFINITY, |g| g.over_triangle) };
                // The t of the least (or, negated, the greatest) gap on a span.
                let extremum = |lo0: f64, hi0: f64, sign: f64| -> f64 {
                    let (mut lo, mut hi) = (lo0, hi0);
                    let mut c = hi - phi * (hi - lo);
                    let mut e = lo + phi * (hi - lo);
                    let mut fc = sign * gap_over_f(c);
                    let mut fe = sign * gap_over_f(e);
                    for _ in 0..60 {
                        if fc < fe {
                            hi = e;
                            e = c;
                            fe = fc;
                            c = hi - phi * (hi - lo);
                            fc = sign * gap_over_f(c);
                        } else {
                            lo = c;
                            c = e;
                            fc = fe;
                            e = lo + phi * (hi - lo);
                            fe = sign * gap_over_f(e);
                        }
                        if hi - lo < t_eps {
                            break;
                        }
                    }
                    0.5 * (lo + hi)
                };
                let peak = if rm.over_triangle > swift::max(ra.over_triangle, rb.over_triangle) {
                    Some(extremum(ta, tb, -1.0))
                } else if ra.rate > 0.0 && rm.rate < 0.0 {
                    Some(extremum(ta, tm, -1.0))
                } else if rm.rate > 0.0 && rb.rate < 0.0 {
                    Some(extremum(tm, tb, -1.0))
                } else {
                    None
                };
                let dip = if rm.over_triangle < swift::min(ra.over_triangle, rb.over_triangle) {
                    Some(extremum(ta, tb, 1.0))
                } else if ra.rate < 0.0 && rm.rate > 0.0 {
                    Some(extremum(ta, tm, 1.0))
                } else if rm.rate < 0.0 && rb.rate > 0.0 {
                    Some(extremum(tm, tb, 1.0))
                } else if ra.rate < 0.0 && rb.rate > 0.0 {
                    Some(extremum(ta, tb, 1.0))
                } else {
                    None
                };
                let mut extrema: Vec<f64> = [peak, dip].into_iter().flatten().collect();
                extrema.sort_by(|a, b| swift_order(*a, *b));
                for tx in extrema {
                    let Some(gx) = gaps(tx) else { continue };
                    if let Some(trace) = trace.as_mut() {
                        trace(format!(
                            "extremum at {} cells between {} and {}: gap {}",
                            g4(tx / cell),
                            g4(ta / cell),
                            g4(tb / cell),
                            e3(gx.over_triangle, true)
                        ));
                    }
                    if walk.hidden(tx, Some(gx), gx.cosine, &mut trace) {
                        return false;
                    }
                }
            }
            // The truncation edge between this crossing and the next, where the triangle
            // meets the truncation: a crease, tested under both.
            if let (true, Some(ga), Some(gb)) = (k + 1 < ts.len(), g, values.get(k + 1).copied().flatten()) {
                let ea = ga.over_triangle - ga.over_truncation;
                let eb = gb.over_triangle - gb.over_truncation;
                if (ea > 0.0) != (eb > 0.0) && ea != eb {
                    let mut truncation_edge = t + (ts[k + 1] - t) * ea / (ea - eb);
                    if magnitude.is_some() {
                        // On f the excess is not linear between crossings:
                        // bisect the sign change to the last bit.
                        let (mut lo, mut hi, mut elo) = (t, ts[k + 1], ea);
                        for _ in 0..60 {
                            let m = 0.5 * (lo + hi);
                            let Some(gm) = gaps(m) else { break };
                            let em = gm.over_triangle - gm.over_truncation;
                            if (em > 0.0) == (elo > 0.0) {
                                lo = m;
                                elo = em;
                            } else {
                                hi = m;
                            }
                        }
                        truncation_edge = 0.5 * (lo + hi);
                    }
                    if truncation_edge > t_eps
                        && let Some(gr) = gaps(truncation_edge)
                    {
                        if walk.hidden(truncation_edge, Some(gr), middle.map_or(gr.cosine, |m| m.cosine), &mut trace) {
                            return false;
                        }
                        if walk.hidden(truncation_edge, Some(gr), 1.0, &mut trace) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

/// The walk's state, and its one question: Swift's nested `hidden(_:_:cosine:)`.
struct Walk {
    outside_yet: bool,
    start_forgiven: bool,
    forgiven: bool,
    elsewhere: bool,
    strictly_inside: bool,
    start_inside: bool,
    on_f: bool,
    t_eps: f64,
    cell: f64,
    depth_tolerance: f64,
    reach: f64,
}

impl Walk {
    /// Hidden at `t`, tested under a triangle of the given cosine; `g` is
    /// the gap there, None beyond the region (outside, then).
    fn hidden(&mut self, t: f64, g: Option<Sample>, cosine: f64, trace: &mut Trace<'_>) -> bool {
        let (t_eps, cell) = (self.t_eps, self.cell);
        let mut visible = t > t_eps || (self.elsewhere && self.start_inside);
        if !visible && self.forgiven && self.strictly_inside && !self.start_forgiven {
            visible = true;
            self.start_forgiven = true;
        }
        // At the start, a point on the surface to rounding is in its own
        // surface's neighbourhood, not outside it; a point of another
        // solid there is on this one's boundary, and not inside it.
        // Further along, a line under the surface by rounding alone is not
        // inside it: the last sample lies at the highest height by
        // construction, and over a truncation at that height the sign of
        // its gap is a rounding error.
        let z_eps = if t > t_eps || self.elsewhere { -1e-9 * cell } else { 1e-9 * cell };
        let inside = match g {
            Some(g) if g.over_triangle < z_eps && g.over_truncation < z_eps => Some(g),
            _ => None,
        };
        let Some(g_in) = inside else {
            // Outside here, the start included: a point on the boundary
            // that stands above the surface is outside from the start,
            // and what its line then enters is an occluder. Against f,
            // outside means outside by more than the depth tolerance: a line that
            // grazes out of the surface by less than its skin and back
            // in has not left it.
            if let (Some(g), true) = (g, self.on_f) {
                let clearance =
                    if g.over_triangle > g.over_truncation { g.over_triangle * cosine } else { g.over_truncation };
                if clearance <= self.depth_tolerance {
                    if let Some(trace) = trace.as_mut() {
                        trace(format!(
                            "t {} cells: within the skin, {} over f",
                            g4(t / cell),
                            e3(g.over_triangle, false)
                        ));
                    }
                    return false;
                }
            }
            self.outside_yet = true;
            if let Some(trace) = trace.as_mut() {
                let gap = match g {
                    Some(g) => {
                        format!("{} over f, {} over truncation", e3(g.over_triangle, true), e3(g.over_truncation, true))
                    }
                    None => "off the region".to_string(),
                };
                trace(format!("t {} cells: outside (gap {gap})", g4(t / cell)));
            }
            return false;
        };
        let g = g_in;
        if !visible {
            if let Some(trace) = trace.as_mut() {
                trace(format!(
                    "t {} cells: inside by {}, not tested at the start",
                    g4(t / cell),
                    e3(-g.over_triangle, false)
                ));
            }
            return false;
        }
        if self.outside_yet {
            let occluded = !self.forgiven || self.elsewhere || t + t_eps >= self.reach;
            if let Some(trace) = trace.as_mut() {
                trace(format!(
                    "t {} cells: inside by {} (below the truncation by {}) after being outside, reach {} cells: {}",
                    g4(t / cell),
                    e3(-g.over_triangle, false),
                    e3(-g.over_truncation, false),
                    g4(self.reach / cell),
                    if occluded { "hidden" } else { "forgiven" }
                ));
            }
            return occluded;
        }
        let depth = if g.over_triangle > g.over_truncation { g.over_triangle * cosine } else { g.over_truncation };
        if let Some(trace) = trace.as_mut() {
            trace(format!(
                "t {} cells: inside by {} in its own surface's neighbourhood, depth {}, depth tolerance {}: {}",
                g4(t / cell),
                e3(-g.over_triangle, false),
                e3(depth, false),
                e3(self.depth_tolerance, false),
                if depth + self.depth_tolerance < 0.0 { "hidden" } else { "forgiven" }
            ));
        }
        depth + self.depth_tolerance < 0.0
    }
}

/// The order Swift's `sort` puts doubles in with `<`: stable, and a pair
/// neither of which is less than the other left as it was.
pub(crate) fn swift_order(a: f64, b: f64) -> std::cmp::Ordering {
    if a < b {
        std::cmp::Ordering::Less
    } else if b < a {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

/// `%.4g`, for the trace: four significant digits, positional where the
/// exponent is from -4 to 3, trailing zeros dropped.
fn g4(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let scientific = format!("{x:.3e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let trim = |s: String| -> String {
        if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
    };
    if (-4..4).contains(&exponent) {
        let decimals = (3 - exponent).max(0) as usize;
        trim(format!("{x:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa.to_string()), exponent.abs())
    }
}

/// `%.3e`, or `%+.3e` with `plus`: the exponent signed and of two digits at
/// least, as C writes it.
fn e3(x: f64, plus: bool) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let scientific = if plus { format!("{x:+.3e}") } else { format!("{x:.3e}") };
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trace_writes_numbers_as_c_does() {
        assert_eq!(g4(0.5), "0.5");
        assert_eq!(g4(12.34567), "12.35");
        assert_eq!(g4(1.0e-5), "1e-05");
        assert_eq!(g4(123456.0), "1.235e+05");
        assert_eq!(e3(-0.00123, false), "-1.230e-03");
        assert_eq!(e3(2.0, true), "+2.000e+00");
    }
}
