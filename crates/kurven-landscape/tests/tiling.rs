//! Faces.swift's suites that hold a landscape's faces, walls and risers to
//! what Hatch.swift and Mesh.swift draw: "faces: a landscape's faces tile its
//! boundary, and each face's map returns its points", "walls: a wall on a
//! band boundary reads its footprint's band, and the far edge the band
//! before it", "depth: the depth pass's walls step where they cross a tile
//! boundary, and a riser fills the drop between two copies" and "depth: a
//! bundle's occluder holds the risers at its tile boundaries", with Swift's
//! tolerances, over the face suites' landscapes (`support::face_landscapes`).
//!
//! A solid's faces tile its boundary: a point of it lies in exactly one
//! face, or on a crease two of them share, and that face's map returns it; a
//! point off it lies in none. What the boundary is comes from the drawing,
//! never from the faces, on every tile.

// Swift's `!(d <= tolerance)` keeps a distance that is not a number outside.
#![allow(
    clippy::neg_cmp_op_on_partial_ord,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::manual_range_contains
)]

mod support;

use std::collections::BTreeMap;

use kurven_core::bisection::bisect;
use kurven_core::face::{Face, FaceCopy, FaceID};
use kurven_core::manifest::{Affine2, BoundaryPerimeter, Region, Truncation, Walls, point_in_polygon};
use kurven_core::mesh::Mesh;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, P3, ParamSpace, WorldSpace};
use support::{Landscape, face_landscapes};

fn maxf(a: f64, b: f64) -> f64 {
    a.max(b)
}

fn minf(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// The parameter at which `copy` of `face` passes through `q`, if it does.
/// The graph and the plateaus lie over the domain, so a point's parameter is
/// its (x, y); every other face of a landscape is flat with an affine map,
/// and a point's parameter is its projection onto the face's plane. The map
/// returning the point is what decides.
fn parameter(q: P3<WorldSpace>, face: &Face, copy: &FaceCopy) -> Option<P2<ParamSpace>> {
    let back = copy.tile.inverse()?;
    let local = back.apply(q).v;
    let p: P2<ParamSpace> = match face.id {
        FaceID::Graph | FaceID::Plateau { .. } => P2::new(local.x, local.y),
        _ => {
            let p0 = P2::<ParamSpace>::new(face.domain.u.range.lo, face.domain.v.range.lo);
            let jet = (face.map)(p0);
            let r = local - (jet.position - jet.du * p0.x() - jet.dv * p0.y());
            let (a, b, c) = (jet.du.dot(jet.du), jet.du.dot(jet.dv), jet.dv.dot(jet.dv));
            let det = a * c - b * b;
            if !(det > 0.0) {
                return None;
            }
            let (ru, rv) = (jet.du.dot(r), jet.dv.dot(r));
            P2::new((c * ru - b * rv) / det, (a * rv - b * ru) / det)
        }
    };
    let tolerance = 1e-9 * maxf(1.0, local.length());
    if !((copy.contains)(p) && ((face.map)(p).position - local).length() <= tolerance) {
        return None;
    }
    Some(p)
}

/// Every copy of a face that holds a point, with the point's parameter there.
fn holders(q: P3<WorldSpace>, faces: &[Face]) -> Vec<(&Face, &FaceCopy, P2<ParamSpace>)> {
    faces
        .iter()
        .flat_map(|face| face.copies.iter().filter_map(move |copy| parameter(q, face, copy).map(|p| (face, copy, p))))
        .collect()
}

/// The outward normal of a copy at a parameter, in space: zero where the
/// face bounds no solid.
fn outward_normal(face: &Face, copy: &FaceCopy, p: P2<ParamSpace>) -> Vec3 {
    copy.place((face.map)(p)).normal() * copy.outward.unwrap_or(0.0)
}

/// The tile under a world point as the drawing finds it: the one that
/// carries it back nearest the domain, inside it where any does, and the
/// point carried back (`Surface::tile_under`).
fn carried_back(w: Vec2, l: &Landscape) -> (Vec2, Affine2) {
    if l.tiles.len() <= 1 {
        return (w, Affine2::IDENTITY);
    }
    let (real, imag) = (l.surface.domain().real, l.surface.domain().imag);
    let mut best = (w, Affine2::IDENTITY);
    let mut nearest = f64::INFINITY;
    for tile in &l.tiles {
        let Some(back) = tile.inverse() else { continue };
        let q = back.apply(P3::new(w.x, w.y, 0.0)).xy().v;
        let d = maxf(minf(real.lo, real.hi) - q.x, 0.0)
            + maxf(q.x - maxf(real.lo, real.hi), 0.0)
            + maxf(minf(imag.lo, imag.hi) - q.y, 0.0)
            + maxf(q.y - maxf(imag.lo, imag.hi), 0.0);
        if d < nearest {
            nearest = d;
            best = (q, back);
        }
        if d == 0.0 {
            break;
        }
    }
    best
}

/// Edge `k`'s curtain as `Mesh::walls` builds it: its top's vertices, and
/// all of them, the top's then the bottom's.
fn curtain(
    perimeter: &BoundaryPerimeter,
    k: usize,
    surface: &kurven_core::surface::Surface,
    base: f64,
    tiles: &[Affine2],
) -> (Vec<P3<WorldSpace>>, Vec<P3<WorldSpace>>) {
    let e = perimeter.edges()[k];
    let all = Mesh::wall_curtain(e.start, e.end, perimeter.inward(k), e.density.max(0) as usize, surface, base, tiles)
        .vertices;
    (all[..all.len() / 2].to_vec(), all)
}

/// Where a chord on the ground from `a` to `b` passes from one tile to
/// another, found by bisection on the tile under it a hair `inward`, and
/// whether `top`, read a hair either side of that point along the chord,
/// differs by more than it moves there: a drop between two copies.
fn drop_across(a: Vec2, b: Vec2, inward: Vec2, l: &Landscape, top: &dyn Fn(Vec2) -> f64) -> (Vec2, bool) {
    let size = maxf(l.surface.domain().real.length().abs(), l.surface.domain().imag.length().abs());
    let e = 1e-9 * size;
    let i = 1e-3 * e * inward;
    let back = |t: f64| carried_back(a + (b - a) * t + i, l).1;
    let first = back(1.0 / 32.0);
    let (holds, fails) = bisect(1.0 / 32.0, 31.0 / 32.0, |t| back(t) == first);
    let at = a + (b - a) * (0.5 * (holds + fails));
    let d = (b - a).normalized();
    let (before, after) = (top(at - e * d + i), top(at + e * d + i));
    let slide = maxf((top(at - 2.0 * e * d + i) - before).abs(), (top(at + 2.0 * e * d + i) - after).abs());
    (at, (before - after).abs() > 4.0 * slide + 1e-9 * maxf(1.0, before.abs()))
}

/// Whether a point of a tile boundary, `n` the boundary's normal, lies in
/// the footprint: held a hair either side of the boundary, or on the
/// perimeter, exactly where the edge runs along an axis; where it slants,
/// within ten times that hair.
fn in_footprint(p: Vec2, region: &Region, n: Vec2, size: f64) -> bool {
    let Region::Inside(perimeter) = region else { return true };
    let side = 1e-9 * size;
    if region.contains(P2::from_vec(p - side * n)) && region.contains(P2::from_vec(p + side * n)) {
        return true;
    }
    perimeter.edges().iter().any(|e| {
        let (a, b) = (e.start.v, e.end.v);
        if a.x == b.x {
            return p.x == a.x && p.y >= minf(a.y, b.y) && p.y <= maxf(a.y, b.y);
        }
        if a.y == b.y {
            return p.y == a.y && p.x >= minf(a.x, b.x) && p.x <= maxf(a.x, b.x);
        }
        let d = b - a;
        let t = maxf(0.0, minf(1.0, (p - a).dot(d) / d.length_squared()));
        (p - (a + t * d)).length() <= 10.0 * side
    })
}

/// What the gate counts on one landscape.
#[derive(Default)]
struct Tally {
    sampled: usize,
    wrong: usize,
    inward: usize,
    on_graph: usize,
    normals: usize,
    skewed: usize,
    stations: usize,
    off_top: usize,
    boundary_points: usize,
    drops: usize,
    examples: BTreeMap<String, String>,
}

impl Tally {
    fn fail(&mut self, kind: &str, what: String) {
        self.examples.entry(kind.to_string()).or_insert(what);
    }
}

fn tiling(l: &Landscape) -> Tally {
    let surface = &l.surface;
    let truncation = &surface.truncation;
    let heightfield = l.heightfield();
    let on_f = l.refine.as_ref().is_some_and(|r| r.magnitude.is_some());
    let (real, imag) = (surface.domain().real, surface.domain().imag);
    let (lo, hi) = (minf(real.lo, real.hi), maxf(real.lo, real.hi));
    let size = maxf(real.length().abs(), imag.length().abs());
    let cell = Vec2::new(
        real.length().abs() / (surface.height.width - 1) as f64,
        imag.length().abs() / (surface.height.height - 1) as f64,
    );
    let mut t = Tally::default();

    // `q` lies in `expected` faces, or within `slack` of that. Where it lies
    // in one, `out` says whether that face's normal points out.
    let expect = |t: &mut Tally,
                  q: P3<WorldSpace>,
                  expected: usize,
                  slack: usize,
                  out: Option<&mut dyn FnMut(&mut Tally, &Face, Vec3) -> bool>| {
        t.sampled += 1;
        let found = holders(q, &l.faces);
        if found.len().abs_diff(expected) > slack {
            t.wrong += 1;
            t.fail(
                "wrong",
                format!("{:?} in {:?}, not {expected} face(s)", q.v, found.iter().map(|f| f.0.id).collect::<Vec<_>>()),
            );
            return;
        }
        let Some(out) = out else { return };
        if found.len() != 1 {
            return;
        }
        let (face, copy, p) = found[0];
        if !out(t, face, outward_normal(face, copy, p)) {
            t.inward += 1;
            t.fail("inward", format!("{:?} at {:?} faces in", face.id, q.v));
        }
    };

    // How many tiles' images of the domain hold a point of the ground.
    let cover = |w: P3<WorldSpace>| -> usize {
        l.tiles
            .iter()
            .filter(|tile| {
                let Some(back) = tile.inverse() else { return false };
                let p = back.apply(w);
                (lo..=hi).contains(&p.x()) && (minf(imag.lo, imag.hi)..=maxf(imag.lo, imag.hi)).contains(&p.y())
            })
            .count()
    };

    // |f| at a point of the domain, untruncated.
    let magnitude = |x: f64, y: f64| -> f64 {
        let p = P2::new(x, y);
        let u = l.refine.as_ref().and_then(|r| r.magnitude.as_ref()).map_or_else(|| surface.magnitude_at(p), |m| m(p));
        if u.is_nan() { f64::INFINITY } else { u }
    };

    // The top: over every point of the footprint, at the height the curves
    // are tested against, on every tile; nothing just above it or just
    // under it, nor over the ground the footprint leaves out. Outward is up,
    // and on the graph it is the facing's normal.
    for tile in &l.tiles {
        for j in 0..47 {
            for i in 0..83 {
                let x = real.lo + real.length() * (i as f64 + 0.37) / 83.0;
                let y = imag.lo + imag.length() * (j as f64 + 0.61) / 47.0;
                let ground = tile.apply(P3::new(x, y, 0.0));
                if cover(ground) != 1 {
                    continue;
                }
                let z = heightfield.surface_height(ground.xy());
                if !z.is_finite() {
                    continue;
                }
                let held = l.region.contains(ground.xy());
                let (m, h) = (magnitude(x, y), truncation.height_at(x));
                let on_crease = (m - h).abs() <= 1e-9 * maxf(1.0, h.abs());
                let tile = *tile;
                let mut out = |t: &mut Tally, face: &Face, n: Vec3| -> bool {
                    if !(n.z > 0.0) {
                        return false;
                    }
                    if face.id == FaceID::Graph {
                        t.on_graph += 1;
                    }
                    let reach = [
                        (tile.a * cell.x, tile.c * cell.x),
                        (-tile.a * cell.x, -tile.c * cell.x),
                        (tile.b * cell.y, tile.d * cell.y),
                        (-tile.b * cell.y, -tile.d * cell.y),
                    ];
                    if face.id == FaceID::Graph
                        && reach.iter().all(|r| {
                            let w = P3::<WorldSpace>::new(ground.x() + r.0, ground.y() + r.1, 0.0);
                            cover(w) <= 1 && Some(carried_back(w.xy().v, l).1) == tile.inverse()
                        })
                    {
                        t.normals += 1;
                        let facing =
                            if on_f { heightfield.graph_normal(ground.xy()) } else { heightfield.normal(ground.xy()) };
                        let off = (n.normalized() - facing.normalized()).length();
                        if !(off <= 1e-9) {
                            t.skewed += 1;
                            t.fail(
                                "skewed",
                                format!("the graph's normal at {:?} is {n:?}, the facing's {facing:?}", ground.v),
                            );
                        }
                    }
                    true
                };
                expect(
                    &mut t,
                    P3::new(ground.x(), ground.y(), z),
                    usize::from(held),
                    usize::from(on_crease),
                    Some(&mut out),
                );
                let off = 1e-3 * maxf(1.0, z.abs());
                expect(&mut t, P3::new(ground.x(), ground.y(), z + off), 0, 0, None);
                expect(&mut t, P3::new(ground.x(), ground.y(), z - off), 0, 0, None);
                // The sheet that is not drawn here, over it.
                let over = maxf(m, h);
                if over.is_finite() && !on_crease {
                    expect(&mut t, P3::new(ground.x(), ground.y(), over), 0, 0, None);
                }
            }
        }
    }

    // The band boundaries inside the domain, and one on its far edge.
    let mut boundaries: Vec<f64> = Vec::new();
    let mut far_edge = false;
    if let Truncation::RealBands { bands, .. } = truncation {
        boundaries = bands.iter().map(|b| b.below).filter(|&b| b > lo && b < hi).collect();
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries.dedup();
        far_edge = bands.iter().any(|b| b.below == hi);
    }
    let side = 1e-9 * size;

    // The walls.
    let mut wall_feet: Vec<(P2<WorldSpace>, P2<WorldSpace>)> = Vec::new();
    if let Walls::Perimeter { perimeter, base } = &l.walls {
        let base = *base;
        let corners: Vec<P2<WorldSpace>> = perimeter.edges().iter().map(|e| e.start).collect();
        let eps = 1e-6 * size;
        let area = (0..corners.len()).fold(0.0, |a, k| {
            let (p, q) = (corners[k], corners[(k + 1) % corners.len()]);
            a + p.x() * q.y() - q.x() * p.y()
        });
        let dense = BoundaryPerimeter::new(
            perimeter.edges().iter().map(|e| kurven_core::manifest::PerimeterEdge { density: 41, ..*e }).collect(),
        );
        let tops: Vec<Vec<P3<WorldSpace>>> = (0..perimeter.edges().len())
            .map(|k| {
                if on_f {
                    let context = kurven_core::hatch::HatchContext {
                        perimeter: Some(perimeter.clone()),
                        region: Region::Full,
                        tiles: l.tiles.clone(),
                        base: 0.0,
                    };
                    let drawn = surface.hatch(
                        &kurven_core::manifest::LayerSource::WallEdges { edges: vec![k as i64], base },
                        &context,
                        l.refine.as_ref(),
                    );
                    return drawn.map(|d| d.path(0).to_vec()).unwrap_or_default();
                }
                curtain(&dense, k, surface, base, &l.tiles).0
            })
            .collect();
        // Where two walls meet with different tops, at a corner a riser ends
        // at, the higher stands over the lower.
        let mut corner_top: Vec<(P2<WorldSpace>, f64)> = Vec::new();
        for (edge, stations) in perimeter.edges().iter().zip(&tops) {
            for v in stations.iter().filter(|v| v.xy() == edge.start || v.xy() == edge.end) {
                match corner_top.iter_mut().find(|c| c.0 == v.xy()) {
                    Some(c) => c.1 = maxf(c.1, v.z()),
                    None => corner_top.push((v.xy(), maxf(f64::NEG_INFINITY, v.z()))),
                }
            }
        }
        for (k, edge) in perimeter.edges().iter().enumerate() {
            wall_feet.push((edge.start, edge.end));
            let stations = &tops[k];
            let d = (edge.end.v - edge.start.v).normalized();
            let inward = if area > 0.0 { Vec2::new(-d.y, d.x) } else { Vec2::new(d.y, -d.x) };
            for &v in stations {
                let top = v.z();
                if !(top.is_finite() && top > base) {
                    continue;
                }
                let at_start = (v.xy().v - edge.start.v).length() < 1e-12;
                let at_end = (v.xy().v - edge.end.v).length() < 1e-12;
                let at_foot: Vec<f64> =
                    stations.iter().filter(|s| (s.xy().v - v.xy().v).length() <= 1e-9 * size).map(|s| s.z()).collect();
                let lower = at_foot.iter().copied().reduce(|a, b| if b < a { b } else { a }).unwrap_or(top);
                let crosses_tile_boundary =
                    carried_back(v.xy().v + 1e-6 * size * d, l).1 != carried_back(v.xy().v - 1e-6 * size * d, l).1;
                for j in 0..7 {
                    let z = base + (top - base) * (j as f64 + 0.5) / 7.0;
                    let at_a_copys_top =
                        crosses_tile_boundary && at_foot.iter().any(|&f| (f - z).abs() <= 1e-9 * maxf(1.0, z.abs()));
                    let slack = if at_a_copys_top {
                        2
                    } else if at_start || at_end || z > lower || crosses_tile_boundary {
                        1
                    } else {
                        0
                    };
                    let mut out = |_: &mut Tally, face: &Face, n: Vec3| -> bool {
                        let FaceID::Wall { edge: e } = face.id else { return true };
                        if !(e == k || !(at_start || at_end)) {
                            return true;
                        }
                        let w = v.xy().v
                            + (if at_start {
                                d
                            } else if at_end {
                                -d
                            } else {
                                Vec2::ZERO
                            }) * (10.0 * eps);
                        let h = Vec2::new(n.x, n.y).normalized() * eps;
                        !point_in_polygon(P2::from_vec(w + h), &corners)
                            && point_in_polygon(P2::from_vec(w - h), &corners)
                    };
                    expect(&mut t, P3::new(v.x(), v.y(), z), 1, slack, Some(&mut out));
                }
                // Nothing over the top, nor over the solid inside the
                // footprint within a few times the matching distance.
                let matching = 4e-9 * maxf(1.0, v.v.length());
                let mut dirs = vec![inward];
                if at_start {
                    dirs.push(inward + d);
                } else if at_end {
                    dirs.push(inward - d);
                }
                let near: Vec<f64> = dirs
                    .iter()
                    .map(|&dir| v.xy().v + matching * dir)
                    .filter(|&p| l.region.contains(P2::from_vec(p)))
                    .map(|p| heightfield.surface_height(P2::from_vec(p)))
                    .filter(|z| z.is_finite())
                    .collect();
                let seq_max = |xs: &[f64]| xs.iter().copied().reduce(|a, b| if a < b { b } else { a });
                let over = maxf(
                    maxf(
                        seq_max(&at_foot).unwrap_or(top),
                        corner_top.iter().find(|c| c.0 == v.xy()).map_or(top, |c| c.1),
                    ),
                    seq_max(&near).unwrap_or(top),
                );
                expect(
                    &mut t,
                    P3::new(v.x(), v.y(), over + 1e-3 * maxf(1.0, over.abs()) + 0.1 * (over - base)),
                    0,
                    0,
                    None,
                );
                expect(&mut t, P3::new(v.x(), v.y(), base - 0.1 * (top - base)), 0, 0, None);
                t.stations += 1;
                let mut towards: Vec<Vec2> = if at_foot.len() == 1 {
                    vec![
                        inward
                            + if at_start {
                                d
                            } else if at_end {
                                -d
                            } else {
                                Vec2::ZERO
                            },
                    ]
                } else {
                    vec![d, -d]
                };
                let mut reach = side;
                if at_foot.len() == 1 {
                    let back = carried_back(v.xy().v + side * towards[0], l).1;
                    let q = back.apply(P3::new(v.x(), v.y(), 0.0)).xy().v;
                    let along = Vec2::new(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y);
                    let nearest =
                        boundaries.iter().copied().reduce(|a, b| if (q.x - b).abs() < (q.x - a).abs() { b } else { a });
                    if let Some(b) = nearest {
                        let apart = (q.x - b).abs();
                        if apart > 1e-12 * size && apart < side {
                            reach = 0.5 * apart / towards[0].length();
                        } else if apart <= 1e-12 * size
                            && (at_start || at_end)
                            && along.x.abs() > 1e-12 * along.length()
                            && let Some(forward) = back.inverse()
                        {
                            let own = if (if at_start { along.x } else { -along.x }) > 0.0 { 1.0 } else { -1.0 };
                            towards = vec![Vec2::new(forward.a * own, forward.c * own)];
                        }
                    }
                }
                let reads: Vec<(f64, f64)> = towards
                    .iter()
                    .map(|&toward| {
                        let solid = heightfield.surface_height(P2::from_vec(v.xy().v + reach * toward));
                        (
                            solid,
                            (heightfield.surface_height(P2::from_vec(v.xy().v + 2.0 * reach * toward)) - solid).abs(),
                        )
                    })
                    .collect();
                if reads.iter().all(|r| r.0.is_finite())
                    && !crosses_tile_boundary
                    && !reads.iter().any(|r| (top - r.0).abs() <= 4.0 * r.1 + 1e-9 * maxf(1.0, r.0.abs()))
                {
                    t.off_top += 1;
                    t.fail(
                        "offTop",
                        format!(
                            "wall {k}'s top at {:?} is {top}, the solid's beside it {:?}",
                            v.xy().v,
                            reads.iter().map(|r| r.0).collect::<Vec<_>>()
                        ),
                    );
                }
            }
        }
    }
    let wall_stands = |w: P2<WorldSpace>| -> bool {
        wall_feet.iter().any(|(a, b)| {
            let d = b.v - a.v;
            let s = maxf(0.0, minf(1.0, (w.v - a.v).dot(d) / d.length_squared()));
            (w.v - (a.v + s * d)).length() <= 4e-9 * maxf(1.0, w.v.length())
        })
    };

    // The risers at the band boundaries.
    for &b in &boundaries {
        for tile in &l.tiles {
            for j in 0..47 {
                let y = imag.lo + imag.length() * (j as f64 + 0.53) / 47.0;
                let foot = tile.apply(P3::new(b, y, 0.0));
                let top = |dx: f64, held: bool| -> Option<f64> {
                    let x = b + dx;
                    if !(x >= lo && x <= hi) {
                        return None;
                    }
                    let w = tile.apply(P3::new(x, y, 0.0));
                    if !(cover(w) == 1 && (!held || l.region.contains(w.xy()))) {
                        return None;
                    }
                    let z = heightfield.surface_height(w.xy());
                    if z.is_finite() { Some(z) } else { None }
                };
                let (left, right) = (top(-side, true), top(side, true));
                let slide = (magnitude(b + side, y) - magnitude(b - side, y)).abs();
                match (left, right) {
                    (Some(lz), Some(rz)) => {
                        let (span_lo, span_hi) = (minf(lz, rz), maxf(lz, rz));
                        let lower_side = if lz > rz { 1.0 } else { -1.0 };
                        let outward = Vec2::new(tile.a * lower_side, tile.c * lower_side);
                        if span_hi - span_lo > 4.0 * slide + 1e-9 * maxf(1.0, span_hi.abs()) {
                            for k in 0..7 {
                                let z = span_lo + (span_hi - span_lo) * (k as f64 + 0.5) / 7.0;
                                let mut out = |_: &mut Tally, face: &Face, n: Vec3| -> bool {
                                    if !matches!(face.id, FaceID::Riser { .. }) {
                                        return true;
                                    }
                                    Vec2::new(n.x, n.y).dot(outward) > 0.0
                                };
                                expect(&mut t, P3::new(foot.x(), foot.y(), z), 1, 0, Some(&mut out));
                            }
                        }
                        expect(
                            &mut t,
                            P3::new(foot.x(), foot.y(), span_hi + 1e-3 * maxf(1.0, span_hi.abs())),
                            0,
                            0,
                            None,
                        );
                        if !wall_stands(foot.xy()) {
                            expect(
                                &mut t,
                                P3::new(foot.x(), foot.y(), span_lo - 1e-3 * maxf(1.0, span_lo.abs())),
                                0,
                                0,
                                None,
                            );
                        }
                    }
                    (Some(h), None) | (None, Some(h)) => {
                        if !wall_stands(foot.xy()) {
                            continue;
                        }
                        expect(&mut t, P3::new(foot.x(), foot.y(), h + 1e-3 * maxf(1.0, h.abs())), 0, 0, None);
                        let across = if left.is_none() { top(-side, false) } else { top(side, false) };
                        if let Some(across) = across
                            && across - h > 4.0 * slide + 1e-9 * maxf(1.0, across.abs())
                        {
                            expect(&mut t, P3::new(foot.x(), foot.y(), 0.5 * (h + across)), 0, 0, None);
                        }
                    }
                    (None, None) => continue,
                }
            }
        }
    }

    // The tile boundaries, where two copies meet edge to edge, found on the
    // drawing.
    let (ylo, yhi) = (minf(imag.lo, imag.hi), maxf(imag.lo, imag.hi));
    let domain_corners = [Vec2::new(lo, ylo), Vec2::new(hi, ylo), Vec2::new(hi, yhi), Vec2::new(lo, yhi)];
    for tile in &l.tiles {
        if tile.inverse().is_none() || l.tiles.len() <= 1 {
            continue;
        }
        for e in 0..4 {
            let a = tile.apply(P3::new(domain_corners[e].x, domain_corners[e].y, 0.0)).xy().v;
            let b = tile.apply(P3::new(domain_corners[(e + 1) % 4].x, domain_corners[(e + 1) % 4].y, 0.0)).xy().v;
            let u = (b - a).normalized();
            let n = Vec2::new(-u.y, u.x);
            for j in 0..47 {
                let w = a + (b - a) * ((j as f64 + 0.53) / 47.0);
                let top = |sign: f64, held: bool| -> Option<(f64, Affine2, f64)> {
                    let g = P3::<WorldSpace>::new(w.x + sign * side * n.x, w.y + sign * side * n.y, 0.0);
                    if !(cover(g) == 1 && (!held || l.region.contains(g.xy()))) {
                        return None;
                    }
                    let z = heightfield.surface_height(g.xy());
                    let further = heightfield.surface_height(P2::from_vec(w + 2.0 * sign * side * n));
                    if z.is_finite() { Some((z, carried_back(g.xy().v, l).1, (further - z).abs())) } else { None }
                };
                let (Some(l0), Some(r0)) = (top(-1.0, false), top(1.0, false)) else { continue };
                if l0.1 == r0.1 {
                    continue;
                }
                let foot = P2::<WorldSpace>::from_vec(w);
                match (top(-1.0, true), top(1.0, true)) {
                    (Some(left), Some(right)) => {
                        t.boundary_points += 1;
                        let (span_lo, span_hi) = (minf(left.0, right.0), maxf(left.0, right.0));
                        let toward_lower = if left.0 < right.0 { -n } else { n };
                        if span_hi - span_lo > 4.0 * maxf(left.2, right.2) + 1e-9 * maxf(1.0, span_hi.abs()) {
                            t.drops += 1;
                            for k in 0..7 {
                                let z = span_lo + (span_hi - span_lo) * (k as f64 + 0.5) / 7.0;
                                let mut out = |_: &mut Tally, face: &Face, normal: Vec3| -> bool {
                                    if !matches!(face.id, FaceID::TileBoundaryRiser { .. }) {
                                        return false;
                                    }
                                    Vec2::new(normal.x, normal.y).dot(toward_lower) > 0.0
                                };
                                expect(&mut t, P3::new(w.x, w.y, z), 1, 0, Some(&mut out));
                            }
                            if !wall_stands(foot) {
                                expect(
                                    &mut t,
                                    P3::new(w.x, w.y, span_lo - 1e-3 * maxf(1.0, span_lo.abs())),
                                    0,
                                    0,
                                    None,
                                );
                            }
                        }
                        expect(&mut t, P3::new(w.x, w.y, span_hi + 1e-3 * maxf(1.0, span_hi.abs())), 0, 0, None);
                    }
                    (Some(h), None) | (None, Some(h)) => {
                        if !wall_stands(foot) {
                            continue;
                        }
                        expect(&mut t, P3::new(w.x, w.y, h.0 + 1e-3 * maxf(1.0, h.0.abs())), 0, 0, None);
                        let across = if h.1 == l0.1 { r0.0 } else { l0.0 };
                        if across - h.0 > 4.0 * h.2 + 1e-9 * maxf(1.0, across.abs()) {
                            expect(&mut t, P3::new(w.x, w.y, 0.5 * (h.0 + across)), 0, 0, None);
                        }
                    }
                    (None, None) => continue,
                }
            }
        }
    }
    // A band boundary on the far edge: the landscape ends with the band
    // before it.
    if far_edge {
        for tile in &l.tiles {
            for j in 0..47 {
                let y = imag.lo + imag.length() * (j as f64 + 0.53) / 47.0;
                let foot = tile.apply(P3::new(hi, y, 0.0));
                let inside = tile.apply(P3::new(hi - side, y, 0.0));
                if !(cover(foot) == 1 && !wall_stands(foot.xy()) && l.region.contains(inside.xy())) {
                    continue;
                }
                let top = heightfield.surface_height(inside.xy());
                let beyond = minf(magnitude(hi, y), truncation.height_at(hi));
                if top.is_finite() && top - beyond > 1e-6 * maxf(1.0, top.abs()) {
                    expect(&mut t, P3::new(foot.x(), foot.y(), 0.5 * (top + beyond)), 0, 0, None);
                }
            }
        }
    }
    t
}

#[test]
fn a_landscapes_faces_tile_its_boundary_and_each_faces_map_returns_its_points() {
    let landscapes = face_landscapes();
    let mut failures = Vec::new();
    for l in &landscapes {
        if l.faces.is_empty() {
            if !(l.surface.height.width < 2 || l.surface.height.height < 2) {
                failures.push(format!("{}: no faces, for a grid too small to span its domain", l.name));
            }
            continue;
        }
        let t = tiling(l);
        if !(t.wrong == 0
            && t.inward == 0
            && t.skewed == 0
            && t.sampled > 10_000
            && (t.normals > 300 || t.on_graph == 0))
        {
            failures.push(format!(
                "{}: every point of the boundary in one face, each turned out, the graph's normal the facing's: {} in the wrong number, {} facing in, {} of {} normals off, of {}; {} of {} tile boundary points with a drop: {:?}",
                l.name,
                t.wrong,
                t.inward,
                t.skewed,
                t.normals,
                t.sampled,
                t.drops,
                t.boundary_points,
                t.examples.iter().filter(|e| e.0 != "offTop").collect::<Vec<_>>()
            ));
        }
        if l.drops_at_a_tile_boundary && !(t.drops > 30) {
            failures.push(format!(
                "{}: the solid drops at a tile boundary, and a riser stands there: {} of {}",
                l.name, t.drops, t.boundary_points
            ));
        }
        if matches!(l.walls, Walls::Perimeter { .. }) && !(t.off_top == 0 && t.stations > 50) {
            failures.push(format!(
                "{}: every wall rises to the solid's top beside it: {} of {} stations off: {}",
                l.name,
                t.off_top,
                t.stations,
                t.examples.get("offTop").cloned().unwrap_or_default()
            ));
        }
    }
    // `Surface::faces` lists them in `FaceID`'s order.
    assert!(
        landscapes.iter().all(|l| l.faces.windows(2).all(|w| w[0].id <= w[1].id)),
        "the faces come in FaceID's order"
    );
    assert!(failures.is_empty(), "{} landscapes: {failures:#?}", landscapes.len());
}

use kurven_core::hatch::HatchContext;
use kurven_core::hatching::{FaceSelector, Hatching};
use kurven_core::manifest::{LayerSource, PerimeterEdge};
use kurven_core::surface::{ContourRefine, Surface};
use support::{bands, build, domain, footprint};

fn context(perimeter: &BoundaryPerimeter, tiles: &[Affine2]) -> HatchContext {
    HatchContext { perimeter: Some(perimeter.clone()), region: Region::Full, tiles: tiles.to_vec(), base: 0.0 }
}

fn wall_edges(edges: Vec<i64>, base: f64) -> LayerSource {
    LayerSource::WallEdges { edges, base }
}

fn densified(perimeter: &BoundaryPerimeter, density: i64) -> BoundaryPerimeter {
    BoundaryPerimeter::new(perimeter.edges().iter().map(|e| PerimeterEdge { density, ..*e }).collect())
}

fn seq_max(xs: impl IntoIterator<Item = f64>) -> Option<f64> {
    xs.into_iter().reduce(|a, b| if a < b { b } else { a })
}

fn seq_min(xs: impl IntoIterator<Item = f64>) -> Option<f64> {
    xs.into_iter().reduce(|a, b| if b < a { b } else { a })
}

#[test]
fn walls_a_wall_on_a_band_boundary_reads_its_footprints_band_and_the_far_edge_the_band_before_it() {
    let landscapes = face_landscapes();
    let mut failures: Vec<String> = Vec::new();
    for l in &landscapes {
        let Walls::Perimeter { perimeter, base } = &l.walls else { continue };
        let base = *base;
        let dense = densified(perimeter, 41);
        let ctx = context(&dense, &l.tiles);
        // A vertical wall's feet stand on its own x, exactly.
        let vertical: Vec<usize> = (0..perimeter.edges().len())
            .filter(|&k| perimeter.edges()[k].start.x() == perimeter.edges()[k].end.x())
            .collect();
        if !vertical.is_empty() {
            let (mut feet, mut off) = (0, Vec::new());
            for &k in &vertical {
                let x = perimeter.edges()[k].start.x();
                let mut vertices = curtain(&dense, k, &l.surface, base, &l.tiles).1;
                if let Some(drawn) = l.surface.hatch(&wall_edges(vec![k as i64], base), &ctx, l.refine.as_ref()) {
                    vertices.extend_from_slice(drawn.path(0));
                    vertices.extend_from_slice(drawn.path(1));
                }
                let strokes = Hatching::plain(FaceSelector::WALL_ANGLE, 0.02);
                let source = LayerSource::Hatching {
                    hatching: strokes,
                    faces: FaceSelector::Walls { edges: vec![k as i64], end_lines: true, base, top_offset: 0.0 },
                };
                if let Some(hatch) = l.surface.hatch(&source, &ctx, l.refine.as_ref()) {
                    vertices.extend_from_slice(&hatch.vertices);
                }
                for v in vertices {
                    feet += 1;
                    if v.x() != x {
                        off.push(format!("{} on Re z = {x}", v.x()));
                    }
                }
            }
            if !(feet > 200 && off.is_empty()) {
                failures.push(format!(
                    "{}: every foot of a vertical wall stands on its own x: {} of {feet} off: {:?}",
                    l.name,
                    off.len(),
                    &off[..off.len().min(3)]
                ));
            }
        }
        if l.refine.is_none() {
            // Without f the top edge and the curtain stand on the same
            // stations and agree at every one, to the bit.
            let (mut stations, mut wrong) = (0, Vec::new());
            for k in 0..perimeter.edges().len() {
                let Some(drawn) = l.surface.hatch(&wall_edges(vec![k as i64], base), &ctx, None) else { continue };
                let top = drawn.path(0).to_vec();
                let mut seen: Vec<(P2<WorldSpace>, usize)> = Vec::new();
                for c in curtain(&dense, k, &l.surface, base, &l.tiles).0 {
                    stations += 1;
                    let at: Vec<&P3<WorldSpace>> = top.iter().filter(|v| v.xy() == c.xy()).collect();
                    let index = match seen.iter_mut().find(|s| s.0 == c.xy()) {
                        Some(s) => {
                            s.1 += 1;
                            s.1 - 1
                        }
                        None => {
                            seen.push((c.xy(), 1));
                            0
                        }
                    };
                    if index >= at.len() {
                        wrong.push(format!("no vertex at {:?}", c.xy().v));
                        continue;
                    }
                    if at[index].z() != c.z() {
                        wrong.push(format!("{:?} at {}, the curtain at {}", c.xy().v, at[index].z(), c.z()));
                    }
                }
            }
            if !(stations >= 41 * perimeter.edges().len() && wrong.is_empty()) {
                failures.push(format!(
                    "{}: the top edge is the curtain's top at every station: {} of {stations} off: {:?}",
                    l.name,
                    wrong.len(),
                    &wrong[..wrong.len().min(3)]
                ));
            }
        } else if let Truncation::RealBands { bands, .. } = &l.surface.truncation {
            let walls: Vec<&Face> = l.faces.iter().filter(|f| matches!(f.id, FaceID::Wall { .. })).collect();
            let real = l.surface.domain().real;
            let boundaries: Vec<f64> = bands
                .iter()
                .map(|b| b.below)
                .filter(|&b| b > minf(real.lo, real.hi) && b < maxf(real.lo, real.hi))
                .collect();
            let cell = maxf(
                real.length().abs() / (l.surface.height.width - 1) as f64,
                l.surface.domain().imag.length().abs() / (l.surface.height.height - 1) as f64,
            );
            let (mut vertices, mut crossings, mut wrong) = (0, 0, Vec::new());
            let mut steps = vec![0; l.tiles.len()];
            for (k, edge) in perimeter.edges().iter().enumerate() {
                let Some(drawn) = l.surface.hatch(&wall_edges(vec![k as i64], base), &ctx, l.refine.as_ref()) else {
                    continue;
                };
                let Some(face) = walls.iter().find(|f| f.id == FaceID::Wall { edge: k }) else { continue };
                let top = drawn.path(0).to_vec();
                let apart = |a: &P3<WorldSpace>, b: &P3<WorldSpace>| (a.xy().v - b.xy().v).length() > 1e-9 * cell;
                let d = (edge.end.v - edge.start.v).normalized();
                let size = maxf(real.length().abs(), l.surface.domain().imag.length().abs());
                let on_tile_boundary = |v: &P3<WorldSpace>| {
                    carried_back(v.xy().v + 1e-6 * size * d, l).1 != carried_back(v.xy().v - 1e-6 * size * d, l).1
                };
                for (i, v) in top.iter().enumerate() {
                    if !((i == 0 || apart(&top[i - 1], v)) && (i == top.len() - 1 || apart(v, &top[i + 1]))) {
                        continue;
                    }
                    vertices += 1;
                    let s = (v.xy().v - edge.start.v).length();
                    let (z, dz) = (v.z(), 1e-9 * maxf(1.0, v.z().abs()));
                    let copy = &face.copies[0];
                    if !((copy.contains)(P2::new(s, z - dz)) && !(copy.contains)(P2::new(s, z + dz))) {
                        wrong.push(format!("{:?} at {z}, not the wall's top", v.xy().v));
                    }
                }
                // Under one tile: across a tile boundary between two, the x
                // carried back jumps.
                for w in top.windows(2) {
                    let (a, c) = (w[0], w[1]);
                    let ((qa, back), (qc, back_c)) = (carried_back(a.xy().v, l), carried_back(c.xy().v, l));
                    if back_c != back {
                        continue;
                    }
                    let (xa, xc) = (qa.x, qc.x);
                    for &b in &boundaries {
                        if minf(xa, xc) < b - 1e-9 * cell && maxf(xa, xc) > b + 1e-9 * cell {
                            crossings += 1;
                            wrong.push(format!("a ramp across {b} from {:?} to {:?}", a.v, c.v));
                        }
                    }
                    let Some(tile) = l.tiles.iter().position(|t| t.inverse() == Some(back)) else { continue };
                    if !apart(&a, &c) && boundaries.iter().any(|&b| (xa - b).abs() <= 1e-9 * cell) {
                        steps[tile] += 1;
                    } else if !apart(&a, &c) && a.z() != c.z() && !on_tile_boundary(&a) {
                        wrong.push(format!(
                            "a step at {:?}, on no band boundary in the tile under it, nor on a tile boundary",
                            a.v
                        ));
                    }
                }
            }
            if !(vertices > 100 && wrong.is_empty()) {
                failures.push(format!(
                    "{}: the top edge on f is the wall face's top, and steps where the tile under it crosses a boundary: {} wrong of {vertices} vertices, {crossings} ramps, steps by tile {steps:?}: {:?}",
                    l.name,
                    wrong.len(),
                    &wrong[..wrong.len().min(3)]
                ));
            }
        }
        // A corner's vertical edge rises to the walls' tops there, to the
        // leaving wall's beside a riser. On the grid, read off the walls'
        // own top edges, as the corners are.
        if l.refine.is_some() {
            continue;
        }
        let all: Vec<i64> = (0..perimeter.edges().len() as i64).collect();
        let Some(drawn) = l.surface.hatch(&wall_edges(all, base), &context(perimeter, &l.tiles), None) else {
            continue;
        };
        let (mut corners, mut wrong) = (0, Vec::new());
        let n_edges = perimeter.edges().len();
        let size = maxf(l.surface.domain().real.length().abs(), l.surface.domain().imag.length().abs());
        for (k, edge) in perimeter.edges().iter().enumerate() {
            let corner = edge.start;
            let previous = (k + n_edges - 1) % n_edges;
            let (here, before) = (drawn.path(2 * k), drawn.path(2 * previous));
            let along_x = |e: usize, at_start: bool| -> bool {
                let d = perimeter.edges()[e].end.v - perimeter.edges()[e].start.v;
                let into = d.normalized() * (if at_start { 1.0 } else { -1.0 });
                let back = carried_back(corner.v + 1e-9 * size * (perimeter.inward(e) + into), l).1;
                let carried = Vec2::new(back.a * d.x + back.b * d.y, back.c * d.x + back.d * d.y);
                carried.x.abs() <= 1e-12 * carried.length()
            };
            let walls = [(here[0].z(), along_x(k, true)), (before[before.len() - 1].z(), along_x(previous, false))];
            let leaving: Vec<f64> = walls.iter().filter(|w| !w.1).map(|w| w.0).collect();
            let expected =
                if leaving.len() == 1 { seq_max(leaving) } else { seq_max(walls.iter().map(|w| w.0)) }.unwrap();
            if !(expected > base) {
                continue;
            }
            let Some(vertical) =
                (2 * n_edges..drawn.count()).map(|i| drawn.path(i)).find(|p| p.iter().all(|v| v.xy() == corner))
            else {
                wrong.push(format!("no edge at {:?}", corner.v));
                continue;
            };
            corners += 1;
            let top = vertical[vertical.len() - 1].z();
            if !((top - expected).abs() <= 1e-12 * maxf(1.0, expected.abs())) {
                wrong.push(format!("{:?} to {top}, not {expected}", corner.v));
            }
        }
        if !(corners >= n_edges / 2 && wrong.is_empty()) {
            failures.push(format!(
                "{}: each corner's edge rises to the walls' tops there: {} of {corners}: {:?}",
                l.name,
                wrong.len(),
                &wrong[..wrong.len().min(3)]
            ));
        }
    }
    // On the tile boundary where the turned tile's riser ends, the corner's
    // edge reaches 2, the band below the riser in that tile.
    if let Some(l) = landscapes.iter().find(|l| l.name.ends_with("a corner on the tile boundary where its riser ends"))
        && let Walls::Perimeter { perimeter, base } = &l.walls
    {
        let all: Vec<i64> = (0..perimeter.edges().len() as i64).collect();
        let drawn = l.surface.hatch(&wall_edges(all, *base), &context(perimeter, &l.tiles), None).unwrap();
        let corner = (2 * perimeter.edges().len()..drawn.count())
            .map(|i| drawn.path(i))
            .find(|p| p.iter().all(|v| v.xy() == P2::new(1.0, 0.5)));
        let top = corner.map(|c| c[c.len() - 1].z());
        if !top.is_some_and(|t| (t - 2.0).abs() <= 1e-12) {
            failures.push(format!("{}: the corner's edge at (1, 1/2) rises to 2: to {top:?}", l.name));
        }
    }
    // A wall leaving the boundary at a slant of 1e-8 runs in the band left
    // of it, its end on the boundary too: its curtain reads 1 all along.
    if let Some(l) = landscapes.iter().find(|l| l.name.ends_with("a wall leaving the boundary at a slope of 1e-8"))
        && let Walls::Perimeter { perimeter, base } = &l.walls
    {
        let tops: Vec<f64> =
            curtain(&densified(perimeter, 41), 0, &l.surface, *base, &l.tiles).0.iter().map(|v| v.z()).collect();
        if !tops.iter().all(|&z| z == 1.0) {
            failures.push(format!("{}: its curtain reads 1 all along: {:?}", l.name, tops));
        }
    }

    // A wall that crosses a band boundary steps there, on f.
    let w = domain(-0.5, 1.5, -1.0, 1.0);
    for (left, right) in [(0.5, 1.0), (1.0, 0.5)] {
        let tower = build("1/(z - 0.5)", w, 32, bands(&[(0.5, left)], right));
        let magnitude = tower.magnitude().clone();
        for (slope, front) in [
            ("rising", [(-0.5, -0.9), (1.5, -0.6)]),
            ("falling", [(-0.5, -0.6), (1.5, -0.9)]),
            ("level", [(-0.5, -0.75), (1.5, -0.75)]),
        ] {
            let corners = [front[0], front[1], (1.5, 1.0), (-0.5, 1.0)];
            let perimeter = footprint(&corners, 17);
            let drawn = tower.surface.hatch(
                &wall_edges(vec![0], 0.0),
                &HatchContext::with_perimeter(Some(perimeter)),
                Some(&tower.refine),
            );
            let top = drawn.map(|d| d.path(0).to_vec()).unwrap_or_default();
            let mut wrong = Vec::new();
            for v in &top {
                if !((v.z() - minf(magnitude(P2::new(v.x(), v.y())), if v.x() < 0.5 { left } else { right })).abs()
                    <= 1e-9)
                {
                    wrong.push(format!("({}, {})", v.x(), v.z()));
                }
            }
            let mut steps = 0;
            for p in top.windows(2) {
                let (a, c) = (p[0], p[1]);
                if (a.x() < 0.5) == (c.x() < 0.5) {
                    continue;
                }
                steps += 1;
                if !((a.xy().v - c.xy().v).length() <= 1e-12) {
                    wrong.push(format!("a ramp from ({}, {}) to ({}, {})", a.x(), a.z(), c.x(), c.z()));
                }
            }
            if !(!top.is_empty() && steps == 1 && wrong.is_empty()) {
                failures.push(format!(
                    "a wall {slope} across a boundary at {left} | {right}, on f: {steps} crossings, {} wrong: {:?}",
                    wrong.len(),
                    &wrong[..wrong.len().min(3)]
                ));
            }
        }
    }

    // A wall whose stations either side of a tile boundary lie under
    // different tiles, with a band boundary between them in each.
    let straddling_tower = build("1/(z - 1.4)", w, 32, bands(&[(1.45, 0.5)], 1.0));
    let straddled = footprint(&[(0.2, -0.5), (2.8, -0.5), (2.8, 0.5), (0.2, 0.5)], 4);
    let straddling = Landscape::new(
        "a tower at 0.5 | 1 in two tiles, the second reflected, a wall across the tile boundary at density 4",
        straddling_tower.surface.clone(),
        vec![Affine2::IDENTITY, Affine2 { a: -1.0, b: 0.0, tx: 3.0, c: 0.0, d: 1.0, ty: 0.0 }],
        Region::Inside(straddled.clone()),
        Walls::Perimeter { perimeter: straddled.clone(), base: -0.5 },
        Some(straddling_tower.refine.clone()),
    );
    if let Some(drawn) = straddling.surface.hatch(
        &wall_edges(vec![0], -0.5),
        &context(&straddled, &straddling.tiles),
        straddling.refine.as_ref(),
    ) {
        let tolerance = straddling_tower.refine.tolerance;
        let top = drawn.path(0).to_vec();
        let h = straddling.heightfield();
        let cell = 2.0 / 31.0;
        let steps: Vec<f64> = top
            .windows(2)
            .filter(|p| (p[0].xy().v - p[1].xy().v).length() <= 1e-12 && p[0].z() != p[1].z())
            .map(|p| (p[0].x() * 1e4).round() / 1e4)
            .collect();
        let (mut samples, mut worst, mut off) = (0, 0.0f64, Vec::new());
        for p in top.windows(2) {
            let (a, c) = (p[0], p[1]);
            if !((a.xy().v - c.xy().v).length() > 1e-12) {
                continue;
            }
            for k in 1..32 {
                let t = k as f64 / 32.0;
                let wv = a.xy().v + (c.xy().v - a.xy().v) * t;
                let solid = h.surface_height(P2::new(wv.x, wv.y + 1e-9));
                if !solid.is_finite() {
                    continue;
                }
                samples += 1;
                let gap = (a.z() + (c.z() - a.z()) * t - solid).abs();
                worst = maxf(worst, gap);
                if gap > 2.0 * tolerance * cell {
                    off.push(format!("x {:.4}: {gap:.4} off", wv.x));
                }
            }
        }
        if !(steps == [1.45, 1.55] && samples > 100 && off.is_empty()) {
            failures.push(format!(
                "{}, on f: steps at {steps:?}, {} of {samples} points off, worst {worst}: {:?}",
                straddling.name,
                off.len(),
                &off[..off.len().min(3)]
            ));
        }
    }

    // Where a wall crosses a tile boundary and the copies' tops differ, its
    // top edge on f steps; between its vertices it is the solid's top.
    for l in landscapes.iter().filter(|l| l.refine.as_ref().is_some_and(|r| r.magnitude.is_some()) && l.tiles.len() > 1)
    {
        let Walls::Perimeter { perimeter, base } = &l.walls else { continue };
        let tolerance = l.refine.as_ref().unwrap().tolerance;
        let h = l.heightfield();
        let (real, imag) = (l.surface.domain().real, l.surface.domain().imag);
        let size = maxf(real.length().abs(), imag.length().abs());
        let cell = maxf(
            real.length().abs() / (l.surface.height.width - 1) as f64,
            imag.length().abs() / (l.surface.height.height - 1) as f64,
        );
        let ctx = context(perimeter, &l.tiles);
        let (mut samples, mut steps, mut drops, mut worst, mut wrong) = (0, 0, 0, 0.0f64, Vec::new());
        for (k, edge) in perimeter.edges().iter().enumerate() {
            let Some(drawn) = l.surface.hatch(&wall_edges(vec![k as i64], *base), &ctx, l.refine.as_ref()) else {
                continue;
            };
            let top = drawn.path(0).to_vec();
            let d = (edge.end.v - edge.start.v).normalized();
            let inward = perimeter.inward(k);
            let solid = |wv: Vec2| h.surface_height(P2::from_vec(wv + 1e-9 * size * inward));
            let top_at = |wv: Vec2| h.surface_height(P2::from_vec(wv));
            for p in top.windows(2) {
                let (a, c) = (p[0], p[1]);
                if (a.xy().v - c.xy().v).length() > 1e-12 * size {
                    let near = a.xy().v + (c.xy().v - a.xy().v) / 32.0;
                    let far = a.xy().v + (c.xy().v - a.xy().v) * 31.0 / 32.0;
                    if carried_back(near + 1e-9 * size * inward, l).1 != carried_back(far + 1e-9 * size * inward, l).1
                        && drop_across(a.xy().v, c.xy().v, inward, l, &top_at).1
                    {
                        wrong.push(format!("a chord from {:?} to {:?} across a drop between two tiles", a.v, c.v));
                    }
                    for j in 1..32 {
                        let t = j as f64 / 32.0;
                        let wv = a.xy().v + (c.xy().v - a.xy().v) * t;
                        let z = solid(wv);
                        if !z.is_finite() {
                            continue;
                        }
                        samples += 1;
                        let gap = (a.z() + (c.z() - a.z()) * t - z).abs();
                        worst = maxf(worst, gap);
                        if gap > 4.0 * tolerance * cell {
                            wrong.push(format!("({:.4}, {:.4}) {gap:.4} off the solid", wv.x, wv.y));
                        }
                    }
                } else if carried_back(a.xy().v + 1e-6 * size * d, l).1 != carried_back(a.xy().v - 1e-6 * size * d, l).1
                {
                    steps += 1;
                    let e = 1e-9 * size;
                    let along = |t: f64| h.surface_height(P2::from_vec(a.xy().v + t * e * d + 1e-3 * e * inward));
                    let (before, after) = (along(-1.0), along(1.0));
                    let slide = maxf((along(-2.0) - before).abs(), (along(2.0) - after).abs());
                    if (before - after).abs() > 4.0 * slide + 1e-9 * maxf(1.0, before.abs()) {
                        drops += 1;
                    } else {
                        wrong.push(format!("a step at {:?}, where the copies' tops agree", a.xy().v));
                    }
                    for (v, z) in [(a, before), (c, after)] {
                        if !((v.z() - z).abs() <= 4.0 * slide + 1e-9 * maxf(1.0, z.abs())) {
                            wrong.push(format!(
                                "at the tile boundary {:?} {}, the solid on its side {z}",
                                v.xy().v,
                                v.z()
                            ));
                        }
                    }
                }
            }
        }
        if !(samples > 300 && wrong.is_empty() && (!l.drops_at_a_tile_boundary || drops > 0)) {
            failures.push(format!(
                "{}: the top edge steps where a wall crosses a tile boundary, and follows the solid between: {steps} steps, {drops} down a drop, {} wrong of {samples}, worst {worst}: {:?}",
                l.name,
                wrong.len(),
                &wrong[..wrong.len().min(3)]
            ));
        }
    }

    // Where a wall crosses a tile boundary and the copies' tops agree, it
    // draws as on one copy: cn(z, 0.64) on f.
    let (k_, kp) = (1.9953027776647292, 1.7507538029157523);
    let cn = build("cn(z, 0.64)", domain(-k_, k_, 0.0, kp), 100, Truncation::Uniform(4.0));
    let across_cn =
        footprint(&[(-k_ + 0.3, 0.3), (3.0 * k_ - 0.3, 0.3), (3.0 * k_ - 0.3, kp - 0.3), (-k_ + 0.3, kp - 0.3)], 17);
    for (name, tile, reflected) in [
        ("reflected", Affine2 { a: -1.0, b: 0.0, tx: 2.0 * k_, c: 0.0, d: 1.0, ty: 0.0 }, true),
        ("moved 2K along", Affine2 { a: 1.0, b: 0.0, tx: 2.0 * k_, c: 0.0, d: 1.0, ty: 0.0 }, false),
    ] {
        let tiles = vec![Affine2::IDENTITY, tile];
        let ctx = context(&across_cn, &tiles);
        let (mut vertices, mut doubled, mut extra) = (0, Vec::new(), 0i64);
        for k in 0..across_cn.edges().len() {
            let Some(drawn) = cn.surface.hatch(&wall_edges(vec![k as i64], 0.0), &ctx, Some(&cn.refine)) else {
                continue;
            };
            let top = drawn.path(0);
            vertices += top.len();
            for p in top.windows(2) {
                if p[0].xy() == p[1].xy() {
                    doubled.push(format!("{:?}", p[0].xy().v));
                }
            }
            if reflected {
                extra += curtain(&across_cn, k, &cn.surface, 0.0, &tiles).0.len() as i64 - 17;
            }
        }
        if !(vertices > 100 && doubled.is_empty() && extra == 0) {
            failures.push(format!(
                "cn(z, 0.64), a second tile {name}, walls across both, on f: {} points with two vertices, {extra} curtain samples past the density: {:?}",
                doubled.len(),
                &doubled[..doubled.len().min(2)]
            ));
        }
    }

    // A riser's top is drawn only where the riser stands, on f.
    let flat = build("10*exp(-2*z)", w, 32, bands(&[(0.2, 2.0)], 4.0));
    let flat_tops: Vec<Landscape> = [
        (
            "both sides only above Im z = 0",
            vec![(0.2, -1.2), (1.7, -1.2), (1.7, 1.2), (-0.7, 1.2), (-0.7, 0.0), (0.2, 0.0)],
        ),
        (
            "notched across the boundary over Im z in (-0.3, 0.3)",
            vec![
                (-0.7, -1.2),
                (1.7, -1.2),
                (1.7, 1.2),
                (-0.7, 1.2),
                (-0.7, 0.3),
                (0.2, 0.3),
                (0.2, -0.3),
                (-0.7, -0.3),
            ],
        ),
    ]
    .into_iter()
    .map(|(name, corners)| {
        Landscape::new(
            &format!("a flat riser top, 10 exp(-2z) at 2 | 4, on a footprint {name}, on f"),
            flat.surface.clone(),
            vec![Affine2::IDENTITY],
            Region::Inside(footprint(&corners, 9)),
            Walls::None,
            Some(flat.refine.clone()),
        )
    })
    .collect();
    for l in landscapes.iter().chain(&flat_tops) {
        let Some(magnitude) = l.refine.as_ref().and_then(|r| r.magnitude.as_ref()) else { continue };
        if l.tiles.len() != 1 {
            continue;
        }
        let Truncation::RealBands { bands, .. } = &l.surface.truncation else { continue };
        let ctx = HatchContext { perimeter: None, region: l.region.clone(), tiles: l.tiles.clone(), base: 0.0 };
        let Some(drawn) = l.surface.hatch(&LayerSource::TruncationEdge { tiled: true }, &ctx, l.refine.as_ref()) else {
            continue;
        };
        let (real, imag) = (l.surface.domain().real, l.surface.domain().imag);
        let (lo, hi) = (minf(real.lo, real.hi), maxf(real.lo, real.hi));
        let mut boundaries: Vec<f64> = bands.iter().map(|b| b.below).filter(|&b| b > lo && b < hi).collect();
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries.dedup();
        let risers: Vec<&Face> = l.faces.iter().filter(|f| matches!(f.id, FaceID::Riser { .. })).collect();
        let side = 1e-9 * maxf(real.length().abs(), imag.length().abs());
        let (mut stray, mut missing, mut asked) = (Vec::new(), Vec::new(), 0);
        let tops: Vec<&[P3<WorldSpace>]> = drawn
            .paths()
            .filter(|p| p.len() >= 2 && boundaries.contains(&p[0].x()) && p.iter().all(|v| v.x() == p[0].x()))
            .collect();
        for p in &tops {
            for s in p.windows(2) {
                let (a, c) = (s[0], s[1]);
                let y = 0.5 * (a.y() + c.y());
                let held = risers.iter().any(|face| {
                    let q = P2::<ParamSpace>::new(y, 0.5 * (face.domain.v.range.lo + minf(a.z(), c.z())));
                    (face.map)(q).position.x == a.x() && face.copies.iter().any(|copy| (copy.contains)(q))
                });
                if !held {
                    stray.push(format!("({}, {y})", a.x()));
                }
            }
        }
        for &b in &boundaries {
            let lesser = minf(l.surface.truncation.height_at(b.next_down()), l.surface.truncation.height_at(b));
            if !lesser.is_finite() {
                continue;
            }
            for j in 0..97 {
                let y = imag.lo + imag.length() * (j as f64 + 0.5) / 97.0;
                if !(l.region.contains(P2::new(b - side, y))
                    && l.region.contains(P2::new(b + side, y))
                    && magnitude(P2::new(b, y)) > lesser * (1.0 + 1e-3))
                {
                    continue;
                }
                asked += 1;
                if !tops.iter().any(|p| {
                    p[0].x() == b
                        && y >= seq_min(p.iter().map(|v| v.y())).unwrap()
                        && y <= seq_max(p.iter().map(|v| v.y())).unwrap()
                }) {
                    missing.push(format!("({b}, {y})"));
                }
            }
        }
        if boundaries.is_empty() {
            continue;
        }
        if !(stray.is_empty() && missing.is_empty()) {
            failures.push(format!(
                "{}: the truncation edge draws a riser's top only where the riser stands, and all of it there: {} on no riser, {} of {asked} with no top: {:?} {:?}",
                l.name,
                stray.len(),
                missing.len(),
                &stray[..stray.len().min(2)],
                &missing[..missing.len().min(2)]
            ));
        }
    }

    // A riser's top at a tile boundary is drawn only where the riser
    // stands, at the higher top, and all of it there; its bottom likewise.
    for l in landscapes.iter().filter(|l| l.tiles.len() > 1) {
        let ctx = HatchContext { perimeter: None, region: l.region.clone(), tiles: l.tiles.clone(), base: 0.0 };
        let Some(drawn) = l.surface.hatch(&LayerSource::TruncationEdge { tiled: true }, &ctx, l.refine.as_ref()) else {
            continue;
        };
        let (real, imag) = (l.surface.domain().real, l.surface.domain().imag);
        let size = maxf(real.length().abs(), imag.length().abs());
        let side = 1e-9 * size;
        let cell = maxf(
            real.length().abs() / (l.surface.height.width - 1) as f64,
            imag.length().abs() / (l.surface.height.height - 1) as f64,
        );
        let bound = l.refine.as_ref().map_or(1e-9 * size, |r| 4.0 * r.tolerance * cell);
        let h = l.heightfield();
        let (lo, hi) = (minf(real.lo, real.hi), maxf(real.lo, real.hi));
        let (ylo, yhi) = (minf(imag.lo, imag.hi), maxf(imag.lo, imag.hi));
        let corners = [Vec2::new(lo, ylo), Vec2::new(hi, ylo), Vec2::new(hi, yhi), Vec2::new(lo, yhi)];
        let sides: Vec<(Vec2, Vec2)> = l
            .tiles
            .iter()
            .flat_map(|tile| {
                (0..4).map(move |e| {
                    (
                        tile.apply(P3::new(corners[e].x, corners[e].y, 0.0)).xy().v,
                        tile.apply(P3::new(corners[(e + 1) % 4].x, corners[(e + 1) % 4].y, 0.0)).xy().v,
                    )
                })
            })
            .collect();
        let drop = |wv: Vec2, n: Vec2| -> Option<(f64, f64)> {
            let reads: Vec<Option<(f64, Affine2, f64)>> = [-1.0, 1.0]
                .iter()
                .map(|&sign| {
                    let g = wv + sign * side * n;
                    let covering = l
                        .tiles
                        .iter()
                        .filter(|tile| {
                            let Some(back) = tile.inverse() else { return false };
                            let q = back.apply(P3::new(g.x, g.y, 0.0));
                            q.x() >= lo && q.x() <= hi && q.y() >= ylo && q.y() <= yhi
                        })
                        .count();
                    if !(covering == 1 && l.region.contains(P2::from_vec(g))) {
                        return None;
                    }
                    let z = h.surface_height(P2::from_vec(g));
                    if !z.is_finite() {
                        return None;
                    }
                    Some((
                        z,
                        carried_back(g, l).1,
                        (h.surface_height(P2::from_vec(wv + 2.0 * sign * side * n)) - z).abs(),
                    ))
                })
                .collect();
            let (Some(a), Some(b)) = (reads[0], reads[1]) else { return None };
            if !(a.1 != b.1 && (a.0 - b.0).abs() > 4.0 * maxf(a.2, b.2) + 1e-9 * maxf(1.0, maxf(a.0, b.0))) {
                return None;
            }
            Some((minf(a.0, b.0), maxf(a.0, b.0)))
        };
        let on_side = |v: Vec2, s: (Vec2, Vec2)| -> bool {
            let d = s.1 - s.0;
            let t = maxf(0.0, minf(1.0, (v - s.0).dot(d) / d.length_squared()));
            (v - (s.0 + t * d)).length() <= 1e-9 * size
        };
        let (mut segments, mut stray, mut asked, mut missing) = (0, Vec::new(), 0, Vec::new());
        let (mut along, mut under): (Vec<(P3<WorldSpace>, P3<WorldSpace>)>, Vec<(P3<WorldSpace>, P3<WorldSpace>)>) =
            (Vec::new(), Vec::new());
        for path in drawn.paths() {
            let Some(&s) = sides.iter().find(|&&s| path.iter().all(|v| on_side(v.xy().v, s))) else { continue };
            let u = (s.1 - s.0).normalized();
            let n = Vec2::new(-u.y, u.x);
            let bottom = path.iter().all(|v| (v.xy().v - s.0).dot(n).abs() > 1e-12 * size);
            for e in [path[0], path[path.len() - 1]] {
                if !in_footprint(e.xy().v, &l.region, n, size) {
                    stray.push(format!("an end at {:?} off the footprint", e.xy().v));
                }
            }
            for p in path.windows(2) {
                let (a, c) = (p[0], p[1]);
                if !((a.xy().v - c.xy().v).length() > 1e-12 * size) {
                    continue;
                }
                let m = 0.5 * (a.xy().v + c.xy().v);
                if !(drop(m, n).is_some()
                    || (l.region.contains(P2::from_vec(m - side * n)) && l.region.contains(P2::from_vec(m + side * n))))
                {
                    stray.push(format!("{m:?} off the footprint"));
                    continue;
                }
                segments += 1;
                let Some(span) = drop(m, n) else {
                    stray.push(format!("{m:?} over no drop"));
                    continue;
                };
                if bottom {
                    under.push((a, c));
                    if !((0.5 * (a.z() + c.z()) - span.0).abs() <= bound) {
                        stray.push(format!("a bottom at {m:?} at {}, the lower top {}", 0.5 * (a.z() + c.z()), span.0));
                    }
                } else {
                    along.push((a, c));
                    if !((0.5 * (a.z() + c.z()) - span.1).abs() <= bound) {
                        stray.push(format!("{m:?} at {}, the higher top {}", 0.5 * (a.z() + c.z()), span.1));
                    }
                }
            }
        }
        for &s in &sides {
            let u = (s.1 - s.0).normalized();
            let n = Vec2::new(-u.y, u.x);
            for j in 0..47 {
                let wv = s.0 + (s.1 - s.0) * ((j as f64 + 0.53) / 47.0);
                let Some(span) = drop(wv, n) else { continue };
                asked += 1;
                let passes = |edge: &[(P3<WorldSpace>, P3<WorldSpace>)], z: f64| -> bool {
                    edge.iter().any(|(a, c)| {
                        let d = c.xy().v - a.xy().v;
                        let t = (wv - a.xy().v).dot(d) / d.length_squared();
                        t >= 0.0
                            && t <= 1.0
                            && (wv - (a.xy().v + t * d)).length() <= 1e-9 * size
                            && (a.z() + (c.z() - a.z()) * t - z).abs() <= bound
                    })
                };
                if !passes(&along, span.1) {
                    missing.push(format!("{wv:?}, to {}", span.1));
                }
                if !passes(&under, span.0) {
                    missing.push(format!("{wv:?}, its bottom at {}", span.0));
                }
            }
        }
        if !(stray.is_empty() && missing.is_empty() && (!l.drops_at_a_tile_boundary || (asked > 30 && segments > 10))) {
            failures.push(format!(
                "{}: the truncation edge draws a riser's top and bottom at a tile boundary where the riser stands: {segments} segments, {} stray; {} of {asked} with no top or bottom: {:?} {:?}",
                l.name,
                stray.len(),
                missing.len(),
                &stray[..stray.len().min(2)],
                &missing[..missing.len().min(2)]
            ));
        }
    }

    // A band boundary on the domain's far edge draws as one a cell beyond
    // it, on the grid and on f.
    let moved = |truncation: &Truncation, from: f64, to: f64| -> Truncation {
        let Truncation::RealBands { bands, beyond } = truncation else { return truncation.clone() };
        Truncation::RealBands {
            bands: bands
                .iter()
                .map(|b| kurven_core::manifest::RealBand {
                    below: if b.below == from { to } else { b.below },
                    height: b.height,
                })
                .collect(),
            beyond: *beyond,
        }
    };
    let mut pairs: Vec<(String, Surface, Surface, Walls, Option<ContourRefine>, Option<ContourRefine>)> = Vec::new();
    if let Some(edge) = landscapes.iter().find(|l| l.name == "the hatch fixture, a band boundary on the far edge") {
        let s = &edge.surface;
        let hi = maxf(s.domain().real.lo, s.domain().real.hi);
        let cell = s.domain().real.length().abs() / (s.height.width - 1) as f64;
        let t = Surface { truncation: moved(&s.truncation, hi, hi + cell), ..s.clone() };
        pairs.push((edge.name.clone(), s.clone(), t, edge.walls.clone(), None, None));
    }
    let tower_walls = footprint(&[(-0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-0.5, 1.0)], 33);
    let on_edge = bands(&[(0.5, 2.0), (1.5, 4.0)], 1.0);
    let tower = build("1/(z - 0.5)", w, 32, on_edge.clone());
    let past = build("1/(z - 0.5)", w, 32, moved(&on_edge, 1.5, 1.5 + 2.0 / 31.0));
    pairs.push((
        "a tower banded at 2 | 4 to the far edge and 1 beyond, on f".into(),
        tower.surface.clone(),
        past.surface.clone(),
        Walls::Perimeter { perimeter: tower_walls, base: 0.0 },
        Some(tower.refine.clone()),
        Some(past.refine.clone()),
    ));
    let window = domain(-1.3, 1.6, -1.2, 1.5);
    let window_walls = footprint(&[(-1.3, -1.2), (1.6, -1.2), (1.6, 1.5), (-1.3, 1.5)], 33);
    let cubic_bands = bands(&[(0.2, 2.5), (1.6, 5.0)], 0.3);
    let cubic = build("1/(z^3 - 1)", window, 64, cubic_bands.clone());
    let cubic_cell = 2.9 / (cubic.surface.height.width - 1) as f64;
    let cubic_past = build("1/(z^3 - 1)", window, 64, moved(&cubic_bands, 1.6, 1.6 + cubic_cell));
    let last_column = cubic.surface.height.position(cubic.surface.height.width - 1, 0).x();
    if !(last_column > 1.6) {
        failures.push(format!("the cubic window's last column lies a rounding past Re z = 1.6: {last_column}"));
    }
    pairs.push((
        "1/(z³ - 1) at 2.5 | 5 to the far edge at 1.6 and 0.3 beyond, its last column past it, on f".into(),
        cubic.surface.clone(),
        cubic_past.surface.clone(),
        Walls::Perimeter { perimeter: window_walls, base: 0.0 },
        Some(cubic.refine.clone()),
        Some(cubic_past.refine.clone()),
    ));
    for (name, s, t, walls, refine, refine_past) in &pairs {
        let Walls::Perimeter { perimeter, base } = walls else { continue };
        let ctx = context(perimeter, &[Affine2::IDENTITY]);
        let edges: Vec<i64> = (0..perimeter.edges().len() as i64).collect();
        let layers: Vec<(&str, LayerSource)> = vec![
            ("the wall edges", wall_edges(edges.clone(), *base)),
            (
                "the wall hatching",
                LayerSource::Hatching {
                    hatching: Hatching::plain(FaceSelector::WALL_ANGLE, 0.05),
                    faces: FaceSelector::Walls { edges: edges.clone(), end_lines: true, base: *base, top_offset: 0.0 },
                },
            ),
            ("the truncation edge", LayerSource::TruncationEdge { tiled: false }),
            (
                "the plateaus' hatching along Re z",
                LayerSource::Hatching {
                    hatching: Hatching::plain(0.0, 0.05),
                    faces: FaceSelector::Plateaus { tiled: false },
                },
            ),
            (
                "the plateaus' hatching along Im z",
                LayerSource::Hatching {
                    hatching: Hatching::plain(std::f64::consts::FRAC_PI_2, 0.05),
                    faces: FaceSelector::Plateaus { tiled: false },
                },
            ),
        ];
        let mut differ: Vec<&str> = Vec::new();
        if s.clamped().values != t.clamped().values {
            differ.push("the depth pass's grid");
        }
        if Mesh::truncation_edge_mesh(s, 1, &Region::Full, &[Affine2::IDENTITY])
            != Mesh::truncation_edge_mesh(t, 1, &Region::Full, &[Affine2::IDENTITY])
        {
            differ.push("the truncation-edge mesh");
        }
        if Mesh::walls(perimeter, s, *base, &[Affine2::IDENTITY])
            != Mesh::walls(perimeter, t, *base, &[Affine2::IDENTITY])
        {
            differ.push("the curtain");
        }
        for (what, layer) in &layers {
            if s.hatch(layer, &ctx, refine.as_ref()) != t.hatch(layer, &ctx, refine_past.as_ref()) {
                differ.push(what);
            }
        }
        let (loops, loops_past) = (s.graph_trim_loops(refine.as_ref()), t.graph_trim_loops(refine_past.as_ref()));
        if loops.loops != loops_past.loops || loops.open != 0 {
            differ.push("the graph's trim loops");
        }
        if !differ.is_empty() {
            failures
                .push(format!("{name}: drawn as with the band boundary a cell past the far edge; differ: {differ:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn depth_the_depth_passs_walls_step_where_they_cross_a_tile_boundary_and_a_riser_fills_the_drop_between_two_copies() {
    let mut failures: Vec<String> = Vec::new();
    for l in face_landscapes().iter().filter(|l| l.tiles.len() > 1 && l.refine.is_none()) {
        let (real, imag) = (l.surface.domain().real, l.surface.domain().imag);
        let size = maxf(real.length().abs(), imag.length().abs());
        let side = 1e-9 * size;
        let h = l.heightfield();
        let top_at = |wv: Vec2| h.surface_height(P2::from_vec(wv));
        let (mut chords, mut ramps) = (0, Vec::new());
        if let Walls::Perimeter { perimeter, base } = &l.walls {
            for k in 0..perimeter.edges().len() {
                let top = curtain(perimeter, k, &l.surface, *base, &l.tiles).0;
                let inward = perimeter.inward(k);
                let d = (perimeter.edges()[k].end.v - perimeter.edges()[k].start.v).normalized();
                for p in top.windows(2) {
                    let (a, c) = (p[0], p[1]);
                    if (a.xy().v - c.xy().v).length() <= 1e-12 * size {
                        // Read along the wall either side.
                        if !drop_across(a.xy().v - 1e-6 * size * d, a.xy().v + 1e-6 * size * d, inward, l, &top_at).1 {
                            ramps.push(format!("a step at {:?} where the copies' tops agree", a.xy().v));
                        }
                    }
                }
                for p in top.windows(2) {
                    let (a, c) = (p[0], p[1]);
                    if !((a.xy().v - c.xy().v).length() > 1e-12 * size) {
                        continue;
                    }
                    chords += 1;
                    let near = a.xy().v + (c.xy().v - a.xy().v) / 32.0;
                    let far = a.xy().v + (c.xy().v - a.xy().v) * 31.0 / 32.0;
                    if carried_back(near + side * inward, l).1 != carried_back(far + side * inward, l).1
                        && drop_across(a.xy().v, c.xy().v, inward, l, &top_at).1
                    {
                        ramps.push(format!("{:?} to {:?}", a.v, c.v));
                    }
                }
            }
        }
        let mesh = Mesh::tile_boundary_risers(&l.surface, 1, &l.region, &l.tiles);
        let triangles: Vec<[Vec3; 3]> =
            mesh.triangles.iter().map(|t| [0, 1, 2].map(|i| mesh.vertices[t[i] as usize].v)).collect();
        // Whether a point lies on a triangle of the strip, which is vertical:
        // in its plane and inside it.
        let covered = |p: Vec3| -> bool {
            triangles.iter().any(|t| {
                let n = (t[1] - t[0]).cross(t[2] - t[0]);
                if !(n.length() > 0.0 && (p - t[0]).dot(n).abs() <= 1e-9 * n.length() * maxf(1.0, p.length())) {
                    return false;
                }
                let edges: Vec<f64> = (0..3).map(|i| (t[(i + 1) % 3] - t[i]).cross(p - t[i]).dot(n)).collect();
                edges.iter().all(|&e| e >= -1e-9 * n.length_squared())
                    || edges.iter().all(|&e| e <= 1e-9 * n.length_squared())
            })
        };
        let (lo, hi) = (minf(real.lo, real.hi), maxf(real.lo, real.hi));
        let (ylo, yhi) = (minf(imag.lo, imag.hi), maxf(imag.lo, imag.hi));
        let corners = [Vec2::new(lo, ylo), Vec2::new(hi, ylo), Vec2::new(hi, yhi), Vec2::new(lo, yhi)];
        let cover = |g: Vec2| -> usize {
            l.tiles
                .iter()
                .filter(|tile| {
                    let Some(back) = tile.inverse() else { return false };
                    let q = back.apply(P3::new(g.x, g.y, 0.0));
                    q.x() >= lo && q.x() <= hi && q.y() >= ylo && q.y() <= yhi
                })
                .count()
        };
        // The drawn drop at a point of a tile's side: the depth pass's top a
        // hair either side.
        let drop = |wv: Vec2, n: Vec2| -> Option<(f64, f64)> {
            let reads: Vec<Option<(f64, Affine2, f64)>> = [-1.0, 1.0]
                .iter()
                .map(|&sign| {
                    let g = wv + sign * side * n;
                    if !(cover(g) == 1 && l.region.contains(P2::from_vec(g))) {
                        return None;
                    }
                    let z = h.drawn_height(P2::from_vec(g));
                    if !z.is_finite() {
                        return None;
                    }
                    Some((
                        z,
                        carried_back(g, l).1,
                        (h.drawn_height(P2::from_vec(wv + 2.0 * sign * side * n)) - z).abs(),
                    ))
                })
                .collect();
            let (Some(a), Some(b)) = (reads[0], reads[1]) else { return None };
            if !(a.1 != b.1 && (a.0 - b.0).abs() > 4.0 * maxf(a.2, b.2) + 1e-9 * maxf(1.0, maxf(a.0, b.0))) {
                return None;
            }
            Some((minf(a.0, b.0), maxf(a.0, b.0)))
        };
        let (mut asked, mut uncovered) = (0, Vec::new());
        for tile in &l.tiles {
            for e in 0..4 {
                let a = tile.apply(P3::new(corners[e].x, corners[e].y, 0.0)).xy().v;
                let b = tile.apply(P3::new(corners[(e + 1) % 4].x, corners[(e + 1) % 4].y, 0.0)).xy().v;
                let u = (b - a).normalized();
                let n = Vec2::new(-u.y, u.x);
                for j in 0..47 {
                    let wv = a + (b - a) * ((j as f64 + 0.53) / 47.0);
                    let Some(span) = drop(wv, n) else { continue };
                    for k in 0..7 {
                        asked += 1;
                        let z = span.0 + (span.1 - span.0) * (k as f64 + 0.5) / 7.0;
                        if !covered(Vec3::new(wv.x, wv.y, z)) {
                            uncovered.push(format!("({}, {}, {z})", wv.x, wv.y));
                        }
                    }
                }
            }
        }
        let mut stray = Vec::new();
        for t in &triangles {
            let c = (t[0] + t[1] + t[2]) / 3.0;
            let u = (Vec2::new(t[1].x - t[0].x, t[1].y - t[0].y) + Vec2::new(t[2].x - t[0].x, t[2].y - t[0].y))
                .normalized();
            let n = Vec2::new(-u.y, u.x);
            // Every corner in the footprint, on its perimeter at the most.
            for v in t {
                if !in_footprint(Vec2::new(v.x, v.y), &l.region, n, size) {
                    stray.push(format!("a corner at ({}, {}) off the footprint", v.x, v.y));
                }
            }
            match drop(Vec2::new(c.x, c.y), n) {
                Some(span) if c.z >= span.0 - 1e-9 && c.z <= span.1 + 1e-9 => {}
                _ => stray.push(format!("({}, {}, {})", c.x, c.y, c.z)),
            }
        }
        if !(chords > 0 || asked > 0 || !triangles.is_empty()) {
            continue;
        }
        if !(ramps.is_empty()
            && uncovered.is_empty()
            && stray.is_empty()
            && (!l.drops_at_a_tile_boundary || (asked > 200 && triangles.len() > 10)))
        {
            failures.push(format!(
                "{}: the curtain steps at a tile boundary, and the riser strip covers each drop and nothing else: {} of {chords} curtain chords under two tiles, {} of {asked} points uncovered, {} of {} triangles over no drop: {:?} {:?} {:?}",
                l.name,
                ramps.len(),
                uncovered.len(),
                stray.len(),
                triangles.len(),
                &ramps[..ramps.len().min(1)],
                &uncovered[..uncovered.len().min(2)],
                &stray[..stray.len().min(2)]
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The reviewer's tower, a bundle on f, its manifest given a second tile
/// turned beside the first: the occluder the depth pass draws
/// (`KurvenBundle::occluder`) has every triangle of the riser strip. Swift
/// builds the bundle with `NativeLandscape.build`; here the manifest is a
/// contract landscape's with the build's domain, truncation and occluder
/// (walls on the domain's rectangle at the grid's density, step 1), which is
/// all the occluder reads.
#[test]
fn depth_a_bundles_occluder_holds_the_risers_at_its_tile_boundaries() {
    use kurven_core::bundle::KurvenBundle;
    use kurven_core::manifest::Occluder;
    use kurven_landscape::NativeLandscape;
    let d = domain(-0.5, 1.5, -1.0, 1.0);
    let tower = build("1/(z - 1.4)", d, 32, bands(&[(1.45, 0.5)], 1.0));
    let template = KurvenBundle::read(kurven_core::fixtures::shared("contract/landscape.kurven")).unwrap();
    let mut manifest = template.manifest.clone();
    manifest.domain = d;
    manifest.truncation = tower.surface.truncation.clone();
    let density = NativeLandscape::wall_density(d, tower.shape).unwrap();
    let step = 1.max(tower.shape.n_real.max(tower.shape.n_imag) as i64 / NativeLandscape::OCCLUDER_RESOLUTION);
    manifest.occluder = Occluder {
        step,
        tiles: vec![Affine2::IDENTITY, Affine2 { a: 0.0, b: 1.0, tx: 2.5, c: -1.0, d: 0.0, ty: 0.5 }],
        walls: Walls::Perimeter { perimeter: NativeLandscape::rectangle(d, &density), base: 0.0 },
        region: Region::Full,
        base: 0.0,
    };
    let tiled = KurvenBundle::new(
        "/native/landscape.kurven".into(),
        manifest.clone(),
        tower.surface.clone(),
        Vec::new(),
        None,
        Some(tower.refine.clone()),
    );
    let o = &manifest.occluder;
    let strip = Mesh::tile_boundary_risers(&tiled.surface, o.step as usize, &o.region, &o.tiles);
    let occluder = tiled.occluder();
    let triangles = |m: &Mesh<WorldSpace>| -> Vec<[[u64; 3]; 3]> {
        m.triangles
            .iter()
            .map(|t| [0, 1, 2].map(|i| m.vertices[t[i] as usize].v.to_array().map(|x| (x + 0.0).to_bits())))
            .collect()
    };
    let held = triangles(&occluder);
    let missing = triangles(&strip).into_iter().filter(|t| !held.contains(t)).count();
    assert!(
        strip.triangles.len() > 10 && missing == 0,
        "the tower's occluder, a tile turned beside it, holds the riser strip at the tile boundary: {missing} of {} strip triangles missing",
        strip.triangles.len()
    );
}
