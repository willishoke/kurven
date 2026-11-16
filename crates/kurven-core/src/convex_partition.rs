//! The convex partition of a flat face with holes: the port of
//! `Sources/KurvenCore/ConvexPartition.swift`. What a `Solid` computes
//! from each face's loops, so that visibility and hatching only ever clip
//! against convex polygons.
//!
//! Three steps, each the textbook one. The holes are joined one at a time to
//! the outer loop, or to a hole joined before, by bridges, which makes the
//! face one polygon that runs along each bridge twice. That polygon is cut
//! into triangles by ear clipping. Then Hertel and Mehlhorn (1983): every
//! diagonal whose removal leaves both its ends convex is removed, which
//! merges its two pieces. For a face without holes the result has at most
//! four times as many pieces as the fewest possible. No vertex is
//! added: every piece's corners are the face's own vertices, so the
//! neighbouring faces' sides still meet the pieces' sides end to end.
//!
//! A loop passes a vertex twice where the face touches itself there: a hole
//! touching the outer loop is part of the outer loop, and two holes touching
//! each other are one hole. Corners are taken by their place in a loop, not
//! by vertex, so such a vertex is two corners, each with its own two sides.
//!
//! Every choice is made in index order, so a face gives the same pieces in
//! every process.

use std::collections::{HashMap, HashSet};

use crate::simd::Vec2;
use crate::solid::SolidError;

type Point<'a> = &'a dyn Fn(usize) -> Vec2;

/// The convex pieces of the face bounded by `loops` -- the outer loop
/// counter-clockwise, then each hole clockwise, as vertex indices --
/// with `point` placing each vertex in the face's plane. Each piece is
/// counter-clockwise. `tolerance` is the area below which a corner
/// counts as straight, a small fraction of the face's own: every test
/// here is a cross product held to it. Loops that bound no region the
/// partition can cut -- a hole outside its face, loops that cross -- are
/// refused, naming the face as `face` (`SolidError`).
pub fn pieces(loops: &[Vec<usize>], point: Point, tolerance: f64, face: usize) -> Result<Vec<Vec<usize>>, SolidError> {
    let outer = &loops[0];
    // A loop that passes a vertex twice is never one piece.
    if loops.len() == 1
        && outer.iter().collect::<HashSet<_>>().len() == outer.len()
        && is_convex(outer, point, tolerance)
    {
        return Ok(vec![outer.clone()]);
    }
    let ring = bridged(loops, point, tolerance, face)?;
    Ok(merged(triangles(ring, point, tolerance, face)?, point, tolerance))
}

fn cross(a: Vec2, b: Vec2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// Every corner turns left or runs straight.
fn is_convex(polygon: &[usize], point: Point, tolerance: f64) -> bool {
    let n = polygon.len();
    (0..n).all(|k| {
        let (a, b, c) = (point(polygon[k]), point(polygon[(k + 1) % n]), point(polygon[(k + 2) % n]));
        cross(b - a, c - b) >= -tolerance
    })
}

// MARK: - bridges

/// The outer loop with every hole spliced in: one ring of vertex
/// indices in which each bridge's two ends appear twice.
///
/// The holes are taken from the one reaching furthest along `u` (Eberly's
/// order), so a hole is never walled off by one bridged later. Each is
/// joined by the shortest segment from one of its vertices to a vertex
/// of the ring that leaves both ends into the face and meets nothing
/// else on the way.
fn bridged(loops: &[Vec<usize>], point: Point, tolerance: f64, face: usize) -> Result<Vec<usize>, SolidError> {
    let mut ring = loops[0].clone();
    let mut holes: Vec<Vec<usize>> = loops[1..].to_vec();
    let reach = |h: &[usize]| h.iter().map(|&k| point(k).x).fold(f64::NEG_INFINITY, f64::max);
    holes.sort_by(|a, b| {
        let (xa, xb) = (reach(a), reach(b));
        if xa != xb { kurven_math::real::total_order(xb, xa) } else { a.iter().min().cmp(&b.iter().min()) }
    });
    for h in 0..holes.len() {
        let hole = &holes[h];
        // Everything a bridge must not cross: the ring as it stands,
        // this hole, and the holes still to come.
        let mut walls: Vec<&[usize]> = vec![&ring, hole];
        walls.extend(holes[h + 1..].iter().map(Vec::as_slice));
        let mut best: Option<(f64, usize, usize)> = None;
        for r in 0..ring.len() {
            for j in 0..hole.len() {
                let (p, q) = (point(ring[r]), point(hole[j]));
                let length = (q - p).length_squared();
                if let Some(b) = best
                    && length >= b.0
                {
                    continue;
                }
                if !(opens_into(&ring, r, q, point, tolerance)
                    && opens_into(hole, j, p, point, tolerance)
                    && is_clear(ring[r], hole[j], &walls, point, tolerance))
                {
                    continue;
                }
                best = Some((length, r, j));
            }
        }
        let Some((_, r, j)) = best else {
            return Err(SolidError(format!("face {face}: a hole cannot be joined to the face: it is not inside it")));
        };
        let around: Vec<usize> = hole[j..].iter().chain(&hole[..j]).copied().collect();
        let mut next: Vec<usize> = ring[..=r].to_vec();
        next.extend(around);
        next.extend([hole[j], ring[r]]);
        next.extend_from_slice(&ring[r + 1..]);
        ring = next;
    }
    Ok(ring)
}

/// Whether a segment leaving corner `k` of `lp` toward `q` starts into
/// the face, which lies to the left of every loop, outer or hole.
fn opens_into(lp: &[usize], k: usize, q: Vec2, point: Point, tolerance: f64) -> bool {
    let n = lp.len();
    let (a, c, b) = (point(lp[(k + n - 1) % n]), point(lp[k]), point(lp[(k + 1) % n]));
    let d = q - c;
    if cross(c - a, b - c) > tolerance {
        // A convex corner: strictly between the two sides.
        return cross(b - c, d) > tolerance && cross(d, a - c) > tolerance;
    }
    // A reflex or straight corner: anywhere but on or between the two
    // sides on the outside.
    !(cross(a - c, d) >= -tolerance && cross(d, b - c) >= -tolerance)
}

/// Whether the segment `i -> j` crosses no side of any wall and passes
/// through no vertex other than its own two.
fn is_clear(i: usize, j: usize, walls: &[&[usize]], point: Point, tolerance: f64) -> bool {
    let (p, q) = (point(i), point(j));
    let d = q - p;
    for wall in walls {
        let n = wall.len();
        for k in 0..n {
            let (a, b) = (wall[k], wall[(k + 1) % n]);
            let (pa, pb) = (point(a), point(b));
            if a != i && a != j {
                // A vertex lying on the segment.
                let t = (pa - p).dot(d) / d.length_squared();
                if t > 0.0 && t < 1.0 && cross(d, pa - p).abs() <= tolerance {
                    return false;
                }
            }
            if a == i || a == j || b == i || b == j {
                continue;
            }
            // A proper crossing: each segment's ends on opposite sides of
            // the other.
            let (s1, s2) = (cross(d, pa - p), cross(d, pb - p));
            let e = pb - pa;
            let (s3, s4) = (cross(e, p - pa), cross(e, q - pa));
            if ((s1 > tolerance && s2 < -tolerance) || (s1 < -tolerance && s2 > tolerance))
                && ((s3 > tolerance && s4 < -tolerance) || (s3 < -tolerance && s4 > tolerance))
            {
                return false;
            }
        }
    }
    true
}

// MARK: - ears

/// The ring cut into counter-clockwise triangles by clipping ears: a
/// convex corner whose triangle holds no other vertex, closed, is cut
/// off, the first such in ring order from where the last ear was.
///
/// Clipping an ear changes only its two neighbours' corners, so the
/// scan resumes at the one before it rather than at the start: the
/// corners before that were not ears and, short of the clipped vertex
/// having been all that blocked one, are still not. That makes the cut
/// quadratic in the ring's length, not cubic, and it is the cut a scan
/// from the start would make whenever no such corner is freed.
fn triangles(ring: Vec<usize>, point: Point, tolerance: f64, face: usize) -> Result<Vec<Vec<usize>>, SolidError> {
    let mut ring = ring;
    let mut out = Vec::new();
    let is_ear = |ring: &[usize], k: usize| -> bool {
        let n = ring.len();
        let (a, c, b) = (ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]);
        let (pa, pc, pb) = (point(a), point(c), point(b));
        if !(cross(pc - pa, pb - pc) > tolerance) {
            return false;
        }
        !ring.iter().any(|&v| {
            if v == a || v == b || v == c {
                return false;
            }
            let p = point(v);
            cross(pc - pa, p - pa) >= -tolerance
                && cross(pb - pc, p - pc) >= -tolerance
                && cross(pa - pb, p - pb) >= -tolerance
        })
    };
    let mut start = 0usize;
    while ring.len() > 3 {
        let n = ring.len();
        let Some(k) = (0..n).map(|i| (start + i) % n).find(|&k| is_ear(&ring, k)) else {
            return Err(SolidError(format!(
                "face {face} has no ear to cut: its loops cross, or a hole is not inside it"
            )));
        };
        out.push(vec![ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]]);
        ring.remove(k);
        start = k.saturating_sub(1);
    }
    let (pa, pb, pc) = (point(ring[0]), point(ring[1]), point(ring[2]));
    if !(cross(pb - pa, pc - pb) > tolerance) {
        return Err(SolidError(format!("face {face} leaves a triangle of no area: its loops overlap")));
    }
    out.push(ring);
    Ok(out)
}

// MARK: - Hertel and Mehlhorn

/// The triangles joined across every diagonal whose removal keeps both
/// of its ends convex, each diagonal tried in turn, until none can go.
///
/// Every corner of a join but the diagonal's two ends is a corner of a
/// convex piece, so a diagonal is first tried at its ends alone, the
/// same two cross products the whole join's test would compute. Only
/// one that passes is joined and tested whole, so the many diagonals
/// tried again after each merge cost little.
fn merged(triangles: Vec<Vec<usize>>, point: Point, tolerance: f64) -> Vec<Vec<usize>> {
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (f, t) in triangles.iter().enumerate() {
        for k in 0..3 {
            owner.insert((t[k], t[(k + 1) % 3]), f);
        }
    }
    let mut pieces: Vec<Option<Vec<usize>>> = triangles.into_iter().map(Some).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for f in 0..pieces.len() {
            let Some(mut piece) = pieces[f].clone() else { continue };
            let mut k = 0;
            while k < piece.len() {
                let (a, b) = (piece[k], piece[(k + 1) % piece.len()]);
                let joined = owner.get(&(b, a)).copied().filter(|&g| g != f).and_then(|g| {
                    let other = pieces[g].as_ref()?;
                    if !ends_stay_convex(&piece, k, other, point, tolerance) {
                        return None;
                    }
                    let joined = join(&piece, other, a, b)?;
                    is_convex(&joined, point, tolerance).then_some((g, joined))
                });
                let Some((g, joined)) = joined else {
                    k += 1;
                    continue;
                };
                piece = joined.clone();
                pieces[f] = Some(joined.clone());
                pieces[g] = None;
                owner.remove(&(a, b));
                owner.remove(&(b, a));
                for j in 0..joined.len() {
                    owner.insert((joined[j], joined[(j + 1) % joined.len()]), f);
                }
                changed = true;
                k = 0;
            }
        }
    }
    pieces.into_iter().flatten().collect()
}

/// Whether `p` and `q` joined across the side `k` of `p`, `a -> b`,
/// which `q` runs `b -> a`, would turn left or run straight at `a` and
/// at `b`: from the vertex before `a` in `p` to the one after it in
/// `q`, and from the vertex before `b` in `q` to the one after it in `p`.
fn ends_stay_convex(p: &[usize], k: usize, q: &[usize], point: Point, tolerance: f64) -> bool {
    let (n, m) = (p.len(), q.len());
    let (a, b) = (p[k], p[(k + 1) % n]);
    let j = q.iter().position(|&v| v == b).expect("q runs b -> a");
    let turns_left = |x: usize, y: usize, z: usize| {
        let (px, py, pz) = (point(x), point(y), point(z));
        cross(py - px, pz - py) >= -tolerance
    };
    turns_left(p[(k + n - 1) % n], a, q[(j + 2) % m]) && turns_left(q[(j + m - 1) % m], b, p[(k + 2) % n])
}

/// `p` and `q` joined across the side `a -> b` of `p`, which `q` runs
/// `b -> a`; `None` when that would visit a vertex twice.
fn join(p: &[usize], q: &[usize], a: usize, b: usize) -> Option<Vec<usize>> {
    let i = p.iter().position(|&v| v == b).expect("p has b");
    let j = q.iter().position(|&v| v == a).expect("q has a");
    // p from b round to a, then q from just after a round to just before b.
    let from_b: Vec<usize> = p[i..].iter().chain(&p[..i]).copied().collect();
    let from_a: Vec<usize> = q[j..].iter().chain(&q[..j]).copied().collect();
    let mut joined = from_b;
    if from_a.len() >= 2 {
        joined.extend_from_slice(&from_a[1..from_a.len() - 1]);
    }
    (joined.iter().collect::<HashSet<_>>().len() == joined.len()).then_some(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square with a square hole: every piece convex and counter-clockwise,
    /// and their areas sum to the face's.
    #[test]
    fn a_square_with_a_hole() {
        let pts = [
            Vec2::new(-2.0, -2.0),
            Vec2::new(2.0, -2.0),
            Vec2::new(2.0, 2.0),
            Vec2::new(-2.0, 2.0),
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let point = |k: usize| pts[k];
        let loops = vec![vec![0, 1, 2, 3], vec![4, 7, 6, 5]];
        let got = pieces(&loops, &point, 1e-9, 0).unwrap();
        let area = |poly: &[usize]| {
            let n = poly.len();
            (0..n).map(|k| cross(pts[poly[k]], pts[poly[(k + 1) % n]])).sum::<f64>() / 2.0
        };
        assert!(got.iter().all(|p| is_convex(p, &point, 1e-9) && area(p) > 0.0));
        assert!((got.iter().map(|p| area(p)).sum::<f64>() - 12.0).abs() < 1e-12);
    }
}
