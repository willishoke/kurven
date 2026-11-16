//! Contours of a face's coordinates and height on a landscape's faces: the
//! port of `Sources/KurvenCore/FaceContour.swift`.
//!
//! |f| and arg f have the refiner and their own rules (`Surface::derive`);
//! u, v and the height are smooth on every face, with no singular points to
//! seed, and are contoured by the plain routine: marching squares on the
//! face's parameter rectangle, each vertex bisected onto the level set
//! (`scalar_field::levels`), then cut where it leaves the face.

use kurven_math::swift;

use crate::hatch::HatchContext;
use crate::hatching::{FaceKind, FaceSelector};
use crate::manifest::{Affine2, BoundaryPerimeter, ContourField, DomainAxis, Interval, Region, Restriction};
use crate::parametric::{ParamAxis, ParamRectangle};
use crate::scalar_field::{self, ScalarField, Trim};
use crate::simd::{Vec2, Vec3};
use crate::spaces::{P2, P3, ParamSpace, WorldSpace};
use crate::surface::{ContourRefine, Surface};

impl Surface {
    /// The contours of `field` at `levels` on `faces`, in world space: on
    /// the graph cut at its trim loops, the truncation edge among them, and
    /// lifted to |f|; on the plateaus cut where each ends
    /// (`plateau_contours`); on a wall or a riser cut at its top and bottom
    /// edges as drawn, a riser's a hair onto its lower side, as its hatching
    /// stands. Copied to every tile where the faces are `tiled`. Kept where
    /// `restriction` holds at each vertex, a run cut where it does not, read
    /// as membership (`holds`) at the point of the domain the vertex came
    /// from, before a copy moves it, as `derive` reads a band. Nothing for
    /// |f| and arg f, which `derive` draws, nor on a parametric surface's or
    /// a solid's faces, which a landscape has none of. An isophote, n · l for
    /// a light toward `light`, lies on the graph alone: on a plateau, a wall
    /// or a riser it is constant (`Face.isFlat`), and has no level set.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn contours(
        &self,
        field: ContourField,
        levels: &[f64],
        faces: &FaceSelector,
        restriction: &Restriction,
        context: &HatchContext,
        refine: Option<&ContourRefine>,
        light: Option<Vec3>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        if field.is_field_of_f() || levels.is_empty() {
            return Vec::new();
        }
        if field == ContourField::Isophote && (faces.kind() != FaceKind::Graph || light.is_none()) {
            return Vec::new();
        }
        match faces {
            FaceSelector::Graph { tiled } => {
                let identity = [Affine2::IDENTITY];
                let tiles: &[Affine2] = if *tiled { &context.tiles } else { &identity };
                self.graph_contours(field, levels, tiles, &context.region, restriction, refine, light)
            }
            FaceSelector::Plateaus { tiled } => {
                let placed =
                    Surface::restricted(self.plateau_contours(field, levels, refine), restriction, false, &|v| {
                        v.xy().v
                    });
                let placed = if *tiled { Surface::replicate(&placed, &context.tiles) } else { placed };
                Surface::clip(placed, &context.region)
            }
            FaceSelector::Walls { edges, .. } => {
                let Some(perimeter) = &context.perimeter else { return Vec::new() };
                self.wall_contours(field, levels, perimeter, edges, context.base, &context.tiles, restriction, refine)
            }
            FaceSelector::Risers { tiled, .. } => {
                self.riser_contours(field, levels, &context.tiles, *tiled, &context.region, restriction, refine)
            }
            FaceSelector::Surface | FaceSelector::Solid => Vec::new(),
        }
    }

    /// Whether a contour on a face holds a point under `restriction`, read
    /// at `p`, the point of the domain it came from, as membership (D6). A
    /// face is cut to the footprint where it stands already, so `Footprint`
    /// holds all of it; `BelowTruncation` is membership in the graph, so it
    /// holds all of the graph and nothing of another face; a band holds what
    /// it bounds, open.
    pub fn holds(restriction: &Restriction, p: Vec2, on_graph: bool) -> bool {
        match restriction {
            Restriction::All | Restriction::Footprint => true,
            Restriction::BelowTruncation => on_graph,
            Restriction::Band { axis, lo, hi } => {
                if *axis == DomainAxis::Real {
                    p.x > *lo && p.x < *hi
                } else {
                    p.y > *lo && p.y < *hi
                }
            }
            Restriction::Intersection(all) => all.iter().all(|r| Surface::holds(r, p, on_graph)),
        }
    }

    /// The runs of `paths` at whose every vertex `restriction` holds, read at
    /// the point of the domain `domain` carries it to.
    pub fn restricted(
        paths: Vec<Vec<P3<WorldSpace>>>,
        restriction: &Restriction,
        on_graph: bool,
        domain: &dyn Fn(P3<WorldSpace>) -> Vec2,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        if *restriction == Restriction::All {
            return paths;
        }
        let mut out = Vec::new();
        for path in paths {
            let mut run: Vec<P3<WorldSpace>> = Vec::new();
            for v in path {
                if Surface::holds(restriction, domain(v), on_graph) {
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

    /// The contours of u = Re z, v = Im z, the height |f| or an isophote
    /// on the graph, each copy's clipped to its trimmed domain
    /// (`Face::contours`): cut where they cross the truncation edge, the
    /// domain's edge or the footprint where the copy stands. Lifted to |f|,
    /// f's own with a refiner, restricted where they lie in the domain, and
    /// carried to the copy's tile. An isophote is each copy's own, n · l with
    /// the copy's normal in place (`Face.isophotes(light:_:on:)`), which is
    /// Phase 5's: it draws nothing until then.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn graph_contours(
        &self,
        field: ContourField,
        levels: &[f64],
        tiles: &[Affine2],
        region: &Region,
        restriction: &Restriction,
        refine: Option<&ContourRefine>,
        light: Option<Vec3>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let Some(face) = self.graph(tiles, region, refine) else { return Vec::new() };
        let magnitude = self.face_magnitude(refine);
        let m = magnitude.clone();
        let scalar = ScalarField::new(None, move |p| match field {
            ContourField::U => p.x(),
            ContourField::V => p.y(),
            ContourField::Height | ContourField::Magnitude | ContourField::Phase | ContourField::Isophote => {
                m(P2::new(p.x(), p.y()))
            }
        });
        let mut out = Vec::new();
        for copy in &face.copies {
            let contoured = if field == ContourField::Isophote {
                // PORT: phase 5 -- an isophote is each copy's own n · l with the copy's normal in place (`Face.isophotes(light:_:on:)`, Isophotes.swift).
                let _ = light;
                Vec::new()
            } else {
                face.contours(&scalar, levels, copy)
            };
            let lifted: Vec<Vec<P3<WorldSpace>>> = contoured
                .into_iter()
                .flat_map(|(_, paths)| paths)
                .map(|path| path.iter().map(|p| P3::new(p.x(), p.y(), magnitude(P2::new(p.x(), p.y())))).collect())
                .collect();
            out.extend(
                Surface::restricted(lifted, restriction, true, &|v| v.xy().v)
                    .into_iter()
                    .map(|p| p.into_iter().map(|v| copy.tile.apply(v)).collect()),
            );
        }
        out
    }

    /// The contours of u, the distance along the edge, and of v or the
    /// height, both z, on the walls along `edges`: over each wall's parameter
    /// rectangle, from `base` up, cut at its trim loop, its bottom edge at
    /// `base`, its ends, and its top edge as `wallEdges` draws it
    /// (`top_edge`). A line of u is vertical, from the base to the top edge;
    /// one of the height horizontal, where the top edge stands over it. A
    /// restriction is read at the foot's point of the domain, through the
    /// tile under it.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn wall_contours(
        &self,
        field: ContourField,
        levels: &[f64],
        perimeter: &BoundaryPerimeter,
        edges: &[i64],
        base: f64,
        tiles: &[Affine2],
        restriction: &Restriction,
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let ceiling =
            if self.truncation.highest().is_finite() { self.truncation.highest() } else { self.highest_sample() };
        let mut out = Vec::new();
        for &index in edges {
            if !(index >= 0 && (index as usize) < perimeter.edges().len()) {
                continue;
            }
            let index = index as usize;
            let edge = perimeter.edges()[index];
            let along = edge.end.v - edge.start.v;
            let length = along.length();
            if !(length > 0.0) {
                continue;
            }
            let inward = perimeter.inward(index);
            let top: Vec<P2<ParamSpace>> = self
                .top_edge(&edge, inward, tiles, refine)
                .iter()
                .map(|v| {
                    P2::new(
                        (v.xy().v - edge.start.v).dot(along) / length,
                        if v.z().is_finite() { swift::min(v.z(), ceiling) } else { ceiling },
                    )
                })
                .collect();
            let placed = self.vertical_face_contours(
                field,
                levels,
                length / (edge.density.max(2) - 1) as f64,
                &[P2::new(0.0, base), P2::new(length, base)],
                &top,
                &|s, z| {
                    let p = edge.start.v + along * (s / length);
                    P3::new(p.x, p.y, z)
                },
            );
            out.extend(Surface::restricted(placed, restriction, false, &|v| {
                let t = (v.xy().v - edge.start.v).dot(along) / (length * length);
                self.wall_tile(P2::new(v.x(), v.y()), t, along, inward, tiles).0.v
            }));
        }
        out
    }

    /// The contours of u, the distance along the line a riser stands on,
    /// and of v or the height, both z, on the risers where they stand
    /// (`riser_stretches`): over each stretch, cut at its trim loop, its top
    /// and bottom edges as the truncation edge's layer draws them, and its
    /// ends; each vertex a hair onto the lower side, as the riser's hatching
    /// and bottom edge stand. A restriction is read at the point of the
    /// domain under it, through the tile of the copy the riser faces there.
    #[allow(clippy::too_many_arguments)] // Swift's signature, argument for argument
    pub fn riser_contours(
        &self,
        field: ContourField,
        levels: &[f64],
        tiles: &[Affine2],
        tiled: bool,
        region: &Region,
        restriction: &Restriction,
        refine: Option<&ContourRefine>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let side = 1e-9 * self.cell();
        let mut out = Vec::new();
        for riser in self.riser_stretches(tiles, tiled, region, refine) {
            let start = (riser.point)(0.0).v;
            let d = (riser.point)(1.0).v - start;
            let dd = d.dot(d);
            if !(dd > 0.0) {
                continue;
            }
            let on_line = |v: &P3<WorldSpace>| -> P2<ParamSpace> {
                P2::new((v.xy().v - start).dot(d) / dd * riser.length, v.z())
            };
            let bottom: Vec<P2<ParamSpace>> = riser.bottom.iter().map(on_line).collect();
            let top: Vec<P2<ParamSpace>> = riser.top.iter().map(on_line).collect();
            let placed = self.vertical_face_contours(field, levels, self.cell(), &bottom, &top, &|s, z| {
                let w = (riser.point)(s / riser.length);
                let o = side * (riser.lower)(w);
                P3::new(w.x() + o.x, w.y() + o.y, z)
            });
            out.extend(Surface::restricted(placed, restriction, false, &|v| {
                (riser.back)(P2::new(v.x(), v.y())).apply(P3::new(v.x(), v.y(), 0.0)).xy().v
            }));
        }
        out
    }

    /// The contours of u = s or v = height = z on a vertical face between
    /// two curves of (s, z), `bottom` and `top`, each in order along s: over
    /// the rectangle the two span, sampled about every `spacing` along s, cut
    /// where they cross the loop the two close, and placed in space by
    /// `point`.
    pub fn vertical_face_contours(
        &self,
        field: ContourField,
        levels: &[f64],
        spacing: f64,
        bottom: &[P2<ParamSpace>],
        top: &[P2<ParamSpace>],
        point: &dyn Fn(f64, f64) -> P3<WorldSpace>,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let Some(&b0) = bottom.first() else { return Vec::new() };
        if bottom.len() < 2 || top.len() < 2 {
            return Vec::new();
        }
        let mut lp: Vec<P2<ParamSpace>> = bottom.to_vec();
        lp.extend(top.iter().rev().copied());
        lp.push(b0);
        let (Some(s0), Some(s1)) = (sequence_min(lp.iter().map(|p| p.x())), sequence_max(lp.iter().map(|p| p.x())))
        else {
            return Vec::new();
        };
        let (Some(z0), Some(z1)) = (sequence_min(lp.iter().map(|p| p.y())), sequence_max(lp.iter().map(|p| p.y())))
        else {
            return Vec::new();
        };
        if !(s1 > s0 && z1 > z0 && z0.is_finite() && z1.is_finite()) {
            return Vec::new();
        }
        let samples = swift::int(((s1 - s0) / spacing).ceil()).map_or(2, |k| (k + 1).max(2) as usize);
        let rectangle = ParamRectangle::new(
            ParamAxis::new(Interval::new(s0, s1), false, samples),
            ParamAxis::new(Interval::new(z0, z1), false, 2),
        );
        let scalar = ScalarField::new(None, move |p| if field == ContourField::U { p.x() } else { p.y() });
        scalar_field::levels(&scalar, levels, rectangle, &Trim { trim_loops: Some(vec![lp]), ..Trim::default() })
            .into_iter()
            .flat_map(|(_, paths)| paths)
            .map(|path| path.iter().map(|p| point(p.x(), p.y())).collect())
            .collect()
    }
}

/// Swift's `.min()` on a sequence of doubles: the first of the least by `<`.
fn sequence_min(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut best: Option<f64> = None;
    for v in values {
        if best.is_none_or(|b| v < b) {
            best = Some(v);
        }
    }
    best
}

/// Swift's `.max()` on a sequence of doubles: the first of the greatest by `<`.
fn sequence_max(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut best: Option<f64> = None;
    for v in values {
        if best.is_none_or(|b| b < v) {
            best = Some(v);
        }
    }
    best
}
