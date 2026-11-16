//! Light on a curved face: its isophotes, and hatching along a coordinate
//! whose widths half-Lambert darkness sets. The port of
//! `Sources/KurvenCore/Isophotes.swift`.
//!
//! An isophote is a level set of n · l, the outward unit normal against the
//! unit direction toward a light (Koenderink and van Doorn 1980). On a flat
//! face n is one vector, so n · l is constant and has no level set: a flat
//! face has no isophote field, by its kind, not by what its samples happen
//! to round to. A curved face (the graph, a parametric surface) is
//! contoured by the plain routine, as its coordinates and height are.
//!
//! A curved face is hatched along a coordinate of its parameter rectangle
//! (D8): lines at a spacing are evenly spaced only where the rectangle is an
//! isometry, and a coordinate is the one direction every curved face has.
//! The tone sets each line's width, read along it, and the width changes
//! where the tone crosses a step, so on a half-Lambert tone a line's width
//! changes exactly on the isophotes n · l = 1 − 2k / widthSteps.

use std::collections::BTreeMap;
use std::sync::Arc;

use kurven_math::swift;

use crate::face::{Face, FaceCopy};
use crate::hatch::HatchContext;
use crate::hatching::{FaceKind, Hatching, Tone};
use crate::manifest::{Affine2, DomainAxis};
use crate::parametric::{ParamAxis, ParametricSurface};
use crate::polyline::PolylineSet;
use crate::scalar_field::{Contoured, ScalarField};
use crate::simd::Vec3;
use crate::solid::Solid;
use crate::spaces::{P2, P3, ParamSpace, WorldSpace};
use crate::surface::{ContourRefine, Surface};

/// A face's pieces of hatching by width step, in increasing order, each a
/// polyline of the parameter plane.
pub type ByWidthStep = Vec<(i64, Vec<Vec<P2<ParamSpace>>>)>;

impl Face {
    /// Whether the face is flat: a plateau, a riser, a wall or a polyhedron
    /// face, whose normal is one vector.
    pub fn is_flat(&self) -> bool {
        match self.id.kind() {
            FaceKind::Plateau | FaceKind::Riser | FaceKind::Wall | FaceKind::Solid => true,
            FaceKind::Graph | FaceKind::Surface => false,
        }
    }

    /// The outward unit normal of `copy` at a point of the parameter plane:
    /// `du × dv` of the copy's jet (`FaceCopy::place`), normalized and turned
    /// by its outward sign. `None` where the face bounds no solid, so has no
    /// outward side, or where the normal vanishes.
    pub fn outward_normal(&self, p: P2<ParamSpace>, copy: &FaceCopy) -> Option<Vec3> {
        let outward = copy.outward?;
        let n = copy.place((self.map)(p)).normal();
        let length = n.length();
        if !(length > 0.0 && length.is_finite()) {
            return None;
        }
        Some(outward * (n / length))
    }

    /// n · l on `copy`, for a light toward `light`, normalized: the field
    /// whose level sets are the isophotes. `None` on a flat face, on a face
    /// with no outward side, and for a light with no direction. Not a number
    /// where the normal vanishes.
    pub fn isophote(&self, light: Vec3, copy: &FaceCopy) -> Option<ScalarField> {
        if self.is_flat() || copy.outward.is_none() || !Tone::has_direction(light) {
            return None;
        }
        let l = light.normalized();
        let (face, copy) = (self.clone(), copy.clone());
        Some(ScalarField::new(None, move |p| face.outward_normal(p, &copy).map_or(f64::NAN, |n| n.dot(l))))
    }

    /// The isophotes of `copy` at `levels`, clipped to its trimmed domain
    /// (`contours`), each vertex its point of the parameter plane. Nothing
    /// on a flat face or one with no outward side. A level at ±1 or beyond
    /// draws nothing: n · l reaches ±1 only where n is ±l, where it touches
    /// its extreme value instead of crossing it, and whether marching
    /// squares found a curve there would be decided by rounding.
    pub fn isophotes(&self, light: Vec3, levels: &[f64], copy: &FaceCopy) -> Contoured {
        let Some(field) = self.isophote(light, copy) else { return Vec::new() };
        let inside: Vec<f64> = levels.iter().copied().filter(|c| c.abs() < 1.0).collect();
        self.contours(&field, &inside, copy)
    }

    /// The levels of the coordinate across hatching at `spacing`, along
    /// `axis`'s other coordinate: the spacing asks for a count, and the lines
    /// are spread to fit it. On a periodic axis a whole number of gaps round
    /// the period, the first line on the axis's start, as coordinate curves
    /// stand; on a bounded one as a plateau's hatching divides the domain
    /// (`Surface::plateau_hatching`), the two lines on its ends left out, so
    /// on a landscape the graph's lines continue the plateaus' across the
    /// truncation edge. (Where Swift's `Int` traps, on a count past Int, no
    /// line.)
    pub fn hatch_levels(axis: &ParamAxis, spacing: f64) -> Vec<f64> {
        if !(spacing.is_finite() && spacing > 0.0) {
            return Vec::new();
        }
        let (lo, length) = (axis.range.lo, axis.range.length());
        if axis.periodic {
            let Some(n) = swift::int((length.abs() / spacing).round()) else { return Vec::new() };
            let n = n.max(1);
            return (0..n).map(|k| lo + length * k as f64 / n as f64).collect();
        }
        let n = Surface::stroke_count(axis.range.hi - lo, spacing);
        (1..(n - 1).max(1)).map(|k| lo + (axis.range.hi - lo) * k as f64 / (n - 1) as f64).collect()
    }

    /// Hatching along a coordinate of a curved face, on `copy`, by width
    /// step in increasing order, each piece a polyline of the parameter
    /// plane. Lines along u (angle 0) at levels of v, along v (angle π/2) at
    /// levels of u (`hatch_levels`, the spacing in the coordinate's own
    /// units), each line the contour of the coordinate across it
    /// (`contours`): clipped to the trimmed domain exactly, with a vertex
    /// wherever it crosses a sample of the parameter rectangle. Each line is
    /// then cut where its width step changes: the step of the tone
    /// (`Hatching::width_step`) read at every vertex, and a change between
    /// two vertices bisected on it to the last bit of the chord between them
    /// (`Solid::graded`), so a piece has one width and ends where the next
    /// begins. A change that reverses between two vertices is not seen: the
    /// vertices are the readings. A piece of step 0 is not drawn.
    ///
    /// The tone: uniform, or half-Lambert darkness `(1 − n · l) / 2` at each
    /// point (`Tone::darkness`), n the outward unit normal there
    /// (`outward_normal`). Nothing on a flat face, at an angle along neither
    /// coordinate, under Turing's tone, or, for half-Lambert, on a face with
    /// no outward side.
    pub fn coordinate_hatching(&self, hatching: &Hatching, copy: &FaceCopy) -> ByWidthStep {
        if self.is_flat() {
            return Vec::new();
        }
        let Some(axis) = DomainAxis::from_angle(hatching.angle) else { return Vec::new() };
        let tone: Box<dyn Fn(P2<ParamSpace>) -> f64> = match hatching.tone {
            Tone::Uniform => Box::new(|_| 1.0),
            Tone::HalfLambert { light } => {
                if copy.outward.is_none() || !Tone::has_direction(light) {
                    return Vec::new();
                }
                Box::new(move |p| self.outward_normal(p, copy).map_or(0.0, |n| Tone::darkness(n, light)))
            }
            Tone::Turing { .. } => return Vec::new(),
        };
        // Along u, the lines are levels of v; along v, of u.
        let along_u = axis == DomainAxis::Real;
        let across = if along_u { self.domain.v } else { self.domain.u };
        let coordinate = ScalarField::new(across.period(), move |p| if along_u { p.y() } else { p.x() });
        let lines = self.contours(&coordinate, &Face::hatch_levels(&across, hatching.spacing), copy);
        let mut by_step: BTreeMap<i64, Vec<Vec<P2<ParamSpace>>>> = BTreeMap::new();
        for line in lines.iter().flat_map(|(_, paths)| paths) {
            for (step, piece) in Face::graded(line, &|p| hatching.width_step(tone(p))) {
                if step > 0 {
                    by_step.entry(step).or_default().push(piece);
                }
            }
        }
        by_step.into_iter().collect()
    }

    /// A polyline cut into pieces of one width step each: the step read at
    /// every vertex, and each change between two bisected along the chord
    /// between them, the piece before ending and the next beginning there.
    pub fn graded(
        path: &[P2<ParamSpace>],
        width_step: &dyn Fn(P2<ParamSpace>) -> i64,
    ) -> Vec<(i64, Vec<P2<ParamSpace>>)> {
        if path.len() < 2 {
            return Vec::new();
        }
        let first = path[0];
        let mut out: Vec<(i64, Vec<P2<ParamSpace>>)> = Vec::new();
        let mut current = width_step(first);
        let mut piece = vec![first];
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            let at = |t: f64| -> P2<ParamSpace> {
                if t <= 0.0 {
                    a
                } else if t >= 1.0 {
                    b
                } else {
                    P2::from_vec(a.v + (b.v - a.v) * t)
                }
            };
            for (start, _, step) in Solid::graded(0.0, 1.0, 1.0, &|t| width_step(at(t))) {
                if step == current {
                    continue;
                }
                // A change at `start`: the piece so far ends there, and the
                // next begins there.
                let p = at(start);
                piece.push(p);
                if piece.len() >= 2 {
                    out.push((current, std::mem::take(&mut piece)));
                }
                current = step;
                piece = vec![p];
            }
            piece.push(b);
        }
        if piece.len() >= 2 {
            out.push((current, piece));
        }
        out
    }
}

impl ParametricSurface {
    /// Hatching along a coordinate of the surface's one face, by width step
    /// in increasing order (`Face::coordinate_hatching`), each piece carried
    /// into space with its surface coordinate beside every vertex.
    pub fn coordinate_hatching(&self, hatching: &Hatching) -> Vec<(i64, PolylineSet<WorldSpace>)> {
        let face = self.face();
        let map = self.map.clone();
        face.coordinate_hatching(hatching, &face.copies[0])
            .into_iter()
            .map(|(step, coords)| {
                let paths: Vec<Vec<P3<WorldSpace>>> =
                    coords.iter().map(|c| c.iter().map(|&p| P3::from_vec(map(p).position)).collect()).collect();
                (step, PolylineSet::from_paths_with_coords(&paths, &coords))
            })
            .collect()
    }

    /// The isophotes of the surface's one face at `levels`, carried into
    /// space with their surface coordinates (`Face::isophotes`).
    pub fn isophotes(&self, light: Vec3, levels: &[f64]) -> PolylineSet<WorldSpace> {
        let face = self.face();
        let map = self.map.clone();
        let coords: Vec<Vec<P2<ParamSpace>>> =
            face.isophotes(light, levels, &face.copies[0]).into_iter().flat_map(|(_, paths)| paths).collect();
        let paths: Vec<Vec<P3<WorldSpace>>> =
            coords.iter().map(|c| c.iter().map(|&p| P3::from_vec(map(p).position)).collect()).collect();
        PolylineSet::from_paths_with_coords(&paths, &coords)
    }
}

impl Surface {
    /// Hatching along a coordinate of the graph, by width step in increasing
    /// order, in world space: each copy's (`Face::coordinate_hatching`),
    /// lifted to |f| and carried to its tile, on every tile where `tiled`.
    /// Lines along Re z at angle 0, along Im z at π/2, at the levels the
    /// plateaus' hatching takes.
    pub fn graph_hatching(
        &self,
        hatching: &Hatching,
        tiled: bool,
        context: &HatchContext,
        refine: Option<&ContourRefine>,
    ) -> Vec<(i64, Vec<Vec<P3<WorldSpace>>>)> {
        let tiles = if tiled { context.tiles.clone() } else { vec![Affine2::IDENTITY] };
        let Some(face) = self.graph(&tiles, &context.region, refine) else { return Vec::new() };
        let magnitude: Arc<dyn Fn(P2<crate::spaces::DomainSpace>) -> f64 + Send + Sync> = self.face_magnitude(refine);
        let mut by_step: BTreeMap<i64, Vec<Vec<P3<WorldSpace>>>> = BTreeMap::new();
        for copy in &face.copies {
            for (step, pieces) in face.coordinate_hatching(hatching, copy) {
                by_step.entry(step).or_default().extend(pieces.iter().map(|piece| {
                    piece
                        .iter()
                        .map(|p| copy.tile.apply(P3::new(p.x(), p.y(), magnitude(P2::new(p.x(), p.y())))))
                        .collect::<Vec<_>>()
                }));
            }
        }
        by_step.into_iter().collect()
    }
}
