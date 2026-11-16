//! A polyhedron, built and checked: the construction half of
//! `Sources/KurvenCore/Solid.swift`.
//!
//! The bundle schema decodes a polyhedron's manifest into a `Solid`, checking
//! its loops as it builds it, so the solid's construction comes with the
//! schema (Phase 2). Its visibility (`Solid.Occlusion`, `visibleEdges`,
//! `visibleCurves`), its hatching, its mesh, its rigid motions and the named
//! solids (`convexHull`, `platonic`, `frame`) come with the polyhedra
//! (Phase 5).
//!
//! A polyhedron: a solid bounded by flat faces. Its curves are its edges
//! and whatever is hatched on its faces.
//!
//! The third kind of geometry kurven draws, beside a heightfield and a
//! parametric surface, and the one whose visibility is exact in the plainest
//! sense. A point of an edge is seen when one of the two faces at that edge
//! faces the eye and nothing stands between the point and the eye; a point
//! of a curve on a face is seen when that face faces the eye and nothing
//! stands in front. Both halves are decided in closed form: the first by
//! signs, the second by occlusion against every piece (`Occlusion`) -- the
//! curve and the piece are both flat, so where one passes behind the other
//! is an interval, solved, not searched. No pixel and no grid: the few
//! tolerances are fractions of the solid's own size, so the drawing is the
//! same at any resolution.
//!
//! The solid is given as its faces, in boundary representation: each face
//! as its *loops*, the outer loop counter-clockwise seen from outside, so the
//! face's normal by the right-hand rule points out of the solid, and then one
//! loop per hole, clockwise. A loop is one component of the face's
//! boundary: where a face touches itself at a vertex, its boundary there is
//! one loop that passes the vertex twice. A hole touching the outer loop is
//! part of the outer loop, and two holes touching each other are one hole,
//! so Euler–Poincaré's count of holes stays right. A face need not be
//! convex -- the front of a picture frame is a square with a square hole --
//! so the solid computes each face's *convex partition* (`ConvexPartition`)
//! and keeps the *pieces*: visibility and hatching only ever clip against a
//! convex polygon. A convex face with no hole is one piece.
//!
//! A side shared by two pieces of one face is a *diagonal* of the
//! partition, not an edge of the solid: it is not drawn, and a piece hides
//! nothing of another in its own plane. Hatching is laid over a whole face,
//! so a hatch line runs unbroken across the diagonals and breaks only at the
//! hole.
//!
//! The input is checked, not trusted: a solid whose faces do not close
//! consistently, or whose face is not flat, is refused, and the Platonic
//! tables are built from vertex coordinates alone by a convex hull, so no
//! face is typed in by hand.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;
use std::sync::Arc;

use kurven_math::{real, swift};

use crate::convex_partition;
use crate::mesh::Mesh;
use crate::occlusion::SolidOcclusion;
use crate::polyline::PolylineSet;
use crate::runs::Runs;
use crate::simd::{Mat3, Vec2, Vec3, Vec4};
use crate::spaces::{P3, Transform, ViewSpace, WorldSpace};
use crate::text;

/// Why vertices and loops bound no solid (`Solid::checking`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolidError(pub String);

impl fmt::Display for SolidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SolidError {}

fn refuse<T>(why: String) -> Result<T, SolidError> {
    Err(SolidError(why))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    /// The piece in which the edge runs `a -> b`, and the one in which it
    /// runs `b -> a`.
    pub left: usize,
    pub right: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Solid {
    pub vertices: Vec<P3<WorldSpace>>,
    /// Each face as its loops of vertex indices: the outer loop
    /// counter-clockwise from outside, then each hole clockwise.
    pub loops: Vec<Vec<Vec<usize>>>,
    /// The convex pieces of every face, each counter-clockwise from outside.
    /// Shared, so a camera's occlusion (`SolidOcclusion`) holds them without
    /// a copy.
    pub pieces: Arc<Vec<Vec<usize>>>,
    /// The faces, each as the indices of its pieces.
    pub faces: Vec<Vec<usize>>,
    /// Per face, the frame its hatching is laid out in.
    pub frames: Vec<FaceFrame>,
    /// Every side of every piece once, with the two pieces it bounds: the
    /// solid's edges and the partition's diagonals.
    pub edges: Vec<Edge>,
    /// Unit outward normals, one per piece.
    pub normals: Vec<Vec3>,
    /// Per entry of `edges`, whether it is a diagonal: its two pieces are
    /// pieces of one face.
    pub is_diagonal: Vec<bool>,
    /// Per piece, the face it is a piece of.
    face_of_piece: Vec<usize>,
}

impl Solid {
    /// - Precondition: the loops bound a solid (`checking`).
    pub fn new(vertices: Vec<P3<WorldSpace>>, loops: Vec<Vec<Vec<i64>>>) -> Solid {
        Solid::checking(vertices, loops).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The solid `loops` bound, or why they bound none (`SolidError`): every
    /// index names a vertex and every vertex is a finite point; a face has an
    /// outer loop and every loop at least three vertices, every vertex lies
    /// on a face, no side of no
    /// length, and no side that crosses another of its face or passes
    /// through a vertex of it; no two loops of a face share a vertex; every
    /// face is flat and encloses some area, and its holes run against its
    /// outer loop and lie inside it; every edge lies in exactly two faces,
    /// traversed in opposite directions -- the surface is closed and
    /// consistently oriented -- and the orientation is outward (the signed
    /// volume is positive); no two faces that share an edge lie in one plane,
    /// since then they are one face. What it does not check is that two
    /// faces apart from each other do not pass through each other.
    pub fn checking(vertices: Vec<P3<WorldSpace>>, loops: Vec<Vec<Vec<i64>>>) -> Result<Solid, SolidError> {
        if loops.is_empty() {
            return refuse("a solid has faces, and this has none".to_string());
        }
        for (i, face) in loops.iter().enumerate() {
            if face.is_empty() {
                return refuse(format!("face {i} has no loop"));
            }
            for (l, lp) in face.iter().enumerate() {
                if lp.len() < 3 {
                    return refuse(format!(
                        "face {i}: loop {l} has {} vertices, and a loop has at least three",
                        lp.len()
                    ));
                }
                if let Some(k) = lp.iter().find(|&&k| k < 0 || k as usize >= vertices.len()) {
                    return refuse(format!("face {i}: loop {l} names vertex {k}, and there are {}", vertices.len()));
                }
            }
        }
        if let Some(k) = vertices.iter().position(|p| !(p.x().is_finite() && p.y().is_finite() && p.z().is_finite())) {
            return refuse(format!("vertex {k} is not a finite point: {}", text::doubles(&vertices[k].v.to_array())));
        }
        let loops: Vec<Vec<Vec<usize>>> =
            loops.iter().map(|f| f.iter().map(|l| l.iter().map(|&k| k as usize).collect()).collect()).collect();
        // Each face cut into convex pieces in its own frame.
        let mut pieces: Vec<Vec<usize>> = Vec::new();
        let mut faces: Vec<Vec<usize>> = Vec::new();
        let mut face_of_piece: Vec<usize> = Vec::new();
        let mut frames: Vec<FaceFrame> = Vec::new();
        for (i, face) in loops.iter().enumerate() {
            for (l, lp) in face.iter().enumerate() {
                for k in 0..lp.len() {
                    let next = lp[(k + 1) % lp.len()];
                    if vertices[lp[k]] == vertices[next] {
                        return refuse(format!(
                            "face {i}: loop {l} has a side of no length, from vertex {} to vertex {next}",
                            lp[k]
                        ));
                    }
                }
            }
            // A loop is one component of the face's boundary. Where the face
            // touches itself at a vertex, that component passes the vertex
            // twice, so it is one loop, not two that share the vertex.
            for l in 0..face.len() {
                for m in (l + 1)..face.len() {
                    let a: BTreeSet<usize> = face[l].iter().copied().collect();
                    if let Some(k) = face[m].iter().filter(|k| a.contains(k)).min() {
                        return refuse(format!(
                            "face {i}: loops {l} and {m} share vertex {k}. Where a face touches itself at a vertex, \
                             write its boundary there as one loop that passes the vertex twice"
                        ));
                    }
                }
            }
            let frame = FaceFrame::new(face, &vertices, i)?;
            // The tolerances scale with the face itself, the diagonal of its
            // bounding box in its plane, so a face is cut the same way
            // wherever the solid stands, down to what its coordinates hold.
            for lp in face {
                for &k in lp {
                    let off = (vertices[k].v - frame.origin).dot(frame.normal).abs();
                    if !(off <= frame.flatness_tolerance) {
                        return refuse(format!("face {i} is not flat at vertex {k}"));
                    }
                }
            }
            // The frame's normal is the loops' own, so the outer loop runs
            // counter-clockwise in it whichever way it was given; a face
            // turned inside out is caught where its edges meet its
            // neighbours'. A hole has to run the other way.
            for (l, hole) in face.iter().enumerate().skip(1) {
                let polygon: Vec<Vec3> = hole.iter().map(|&k| vertices[k].v).collect();
                if !(frame.area(&polygon) < 0.0) {
                    return refuse(format!("face {i}: hole {l} runs the same way as the outer loop"));
                }
            }
            if let Some(why) = Solid::crossing(face, &frame, &vertices) {
                return refuse(format!("face {i}: {why}"));
            }
            let cut = convex_partition::pieces(
                face,
                &|k| frame.coordinates(vertices[k].v),
                1e-9 * frame.extent * frame.extent,
                i,
            )?;
            faces.push((pieces.len()..pieces.len() + cut.len()).collect());
            face_of_piece.extend(std::iter::repeat_n(i, cut.len()));
            pieces.extend(cut);
            frames.push(frame);
        }
        // A vertex on no face is no part of the boundary, and would still
        // stretch the solid's bounds and the tolerances measured from them.
        let used: HashSet<usize> = loops.iter().flatten().flatten().copied().collect();
        if let Some(k) = (0..vertices.len()).find(|k| !used.contains(k)) {
            return refuse(format!("vertex {k} lies on no face"));
        }
        // Ordered by the edge, so the refusals below name the same edge in
        // every process (Swift walks a dictionary here, in no fixed order).
        let mut directed: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (f, piece) in pieces.iter().enumerate() {
            for k in 0..piece.len() {
                let e = (piece[k], piece[(k + 1) % piece.len()]);
                if directed.contains_key(&e) {
                    return refuse(format!(
                        "edge {}-{} is traversed twice the same way: the faces are not consistently oriented, or \
                         more than two meet there",
                        e.0, e.1
                    ));
                }
                directed.insert(e, f);
            }
        }
        let mut edges: Vec<Edge> = Vec::new();
        for (&(a, b), &left) in directed.iter().filter(|((a, b), _)| a < b) {
            let Some(&right) = directed.get(&(b, a)) else {
                return refuse(format!("edge {a}-{b} bounds only one face: the surface is not closed"));
            };
            edges.push(Edge { a, b, left, right });
        }
        if directed.len() != 2 * edges.len() {
            let open = directed.keys().filter(|(a, b)| !directed.contains_key(&(*b, *a))).min().expect("an open edge");
            return refuse(format!("edge {}-{} bounds only one face: the surface is not closed", open.1, open.0));
        }
        edges.sort_by_key(|e| (e.a, e.b));
        // Newell's normal: exact for a planar polygon however many vertices
        // it has, and the sum it is is the one the divergence theorem wants.
        let mut normals: Vec<Vec3> = Vec::new();
        let mut volume = 0.0;
        for (f, piece) in pieces.iter().enumerate() {
            let mut n = Solid::newell(std::slice::from_ref(piece), &vertices);
            let len = n.length();
            if !(len > 0.0) {
                return refuse(format!("face {} has a piece of no area", face_of_piece[f]));
            }
            n /= len;
            normals.push(n);
            // (n is twice the area vector; p · n / 6 is the cone's volume.)
            volume += vertices[piece[0]].v.dot(n * len) / 6.0;
        }
        if !(volume > 0.0) {
            return refuse(format!(
                "the faces are oriented inward, or enclose nothing (signed volume {})",
                text::double(volume)
            ));
        }
        let is_diagonal: Vec<bool> = edges.iter().map(|e| face_of_piece[e.left] == face_of_piece[e.right]).collect();
        for (k, e) in edges.iter().enumerate() {
            if is_diagonal[k] {
                continue;
            }
            if !(normals[e.left].dot(normals[e.right]) < 1.0 - 1e-9) {
                return refuse(format!(
                    "faces {} and {} meet at edge {}-{} in one plane: they are one face",
                    face_of_piece[e.left], face_of_piece[e.right], e.a, e.b
                ));
            }
        }
        Ok(Solid {
            vertices,
            loops,
            pieces: Arc::new(pieces),
            faces,
            frames,
            edges,
            normals,
            is_diagonal,
            face_of_piece,
        })
    }

    /// Why the loops of a face cross, or `None`: two of its sides cross, or a
    /// vertex lies on a side it is not an end of. Sides that share a vertex
    /// meet there, as neighbours or where a loop passes the vertex twice.
    /// Measured in the face's plane, to a billionth of its extent, and swept
    /// along `u` so that only sides whose spans overlap are compared.
    fn crossing(loops: &[Vec<usize>], frame: &FaceFrame, vertices: &[P3<WorldSpace>]) -> Option<String> {
        #[derive(Clone, Copy)]
        struct Side {
            a: usize,
            b: usize,
            p: Vec2,
            q: Vec2,
        }
        let mut sides: Vec<Side> = Vec::new();
        for lp in loops {
            for k in 0..lp.len() {
                let (a, b) = (lp[k], lp[(k + 1) % lp.len()]);
                sides.push(Side { a, b, p: frame.coordinates(vertices[a].v), q: frame.coordinates(vertices[b].v) });
            }
        }
        // Stable, as Swift's sort is in practice.
        sides.sort_by(|s, t| real::total_order(s.p.x.min(s.q.x), t.p.x.min(t.q.x)));
        let tolerance = 1e-9 * frame.extent;
        let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
        // The side of the line through `s` that `r` lies on, beyond the
        // tolerance: 1 to the left, -1 to the right, 0 on it.
        let side = |r: Vec2, s: &Side| -> i32 {
            let d = s.q - s.p;
            let c = cross(d, r - s.p);
            let reach = tolerance * d.length();
            if c > reach {
                1
            } else if c < -reach {
                -1
            } else {
                0
            }
        };
        // Whether `r` lies on `s` strictly between its ends.
        let within = |r: Vec2, s: &Side| -> bool {
            let d = s.q - s.p;
            let t = (r - s.p).dot(d) / d.length_squared();
            side(r, s) == 0 && t > 0.0 && t < 1.0
        };
        for i in 0..sides.len() {
            let s = sides[i];
            let reach = s.p.x.max(s.q.x) + tolerance;
            for t in &sides[i + 1..] {
                if !(t.p.x.min(t.q.x) <= reach) {
                    break;
                }
                for (v, r, other) in [(t.a, t.p, &s), (t.b, t.q, &s), (s.a, s.p, t), (s.b, s.q, t)] {
                    if v != other.a && v != other.b && within(r, other) {
                        return Some(format!("vertex {v} lies on side {}-{}", other.a, other.b));
                    }
                }
                if s.a == t.a || s.a == t.b || s.b == t.a || s.b == t.b {
                    continue;
                }
                if side(t.p, &s) * side(t.q, &s) < 0 && side(s.p, t) * side(s.q, t) < 0 {
                    return Some(format!("sides {}-{} and {}-{} cross", s.a, s.b, t.a, t.b));
                }
            }
        }
        None
    }

    /// The boundary's Euler characteristic, 2 − 2g for genus g, by
    /// Euler–Poincaré: V − E + F − H, with a hole of a face counted in H.
    /// 16 − 24 + 10 − 2 = 0 for the picture frame. Over the partition, where
    /// every piece is a disc, V − E + F counts the same.
    pub fn euler_characteristic(&self) -> i64 {
        let v = self.vertices.len() as i64;
        let e = self.is_diagonal.iter().filter(|d| !**d).count() as i64;
        let f = self.loops.len() as i64;
        let h: i64 = self.loops.iter().map(|l| l.len() as i64 - 1).sum();
        v - e + f - h
    }

    /// How many shells the boundary has: its faces in sets that edges join.
    /// One for a solid in one piece with no cavity; the checks accept more,
    /// such as two cubes apart, or a cube with a cavity inside.
    pub fn shell_count(&self) -> usize {
        let mut parent: Vec<usize> = (0..self.loops.len()).collect();
        fn root(parent: &mut [usize], f: usize) -> usize {
            let mut f = f;
            while parent[f] != f {
                parent[f] = parent[parent[f]];
                f = parent[f];
            }
            f
        }
        for e in &self.edges {
            let a = root(&mut parent, self.face_of_piece[e.left]);
            let b = root(&mut parent, self.face_of_piece[e.right]);
            parent[a] = b;
        }
        (0..self.loops.len()).map(|f| root(&mut parent, f)).collect::<HashSet<_>>().len()
    }

    /// The genus of the boundary, `1 − χ / 2`, when it is one shell; `None`
    /// otherwise, where the Euler characteristic sums over the shells and
    /// says nothing of any one of them.
    pub fn genus(&self) -> Option<i64> {
        (self.shell_count() == 1).then(|| 1 - self.euler_characteristic() / 2)
    }

    /// Whether every vertex lies on or behind every face's plane.
    ///
    /// A vertex counts as in front when it is further in front than the
    /// plane tolerance (`plane_tolerance`) for the solid's own size: a
    /// billionth of its bounding box's diagonal, not of its distance from
    /// the origin, with the same floor for the coordinates' rounding as the
    /// flatness check. The solid is at least as large as any face, so a
    /// vertex the flatness check let through is never in front of its own
    /// face.
    pub fn is_convex(&self) -> bool {
        let points: Vec<Vec3> = self.vertices.iter().map(|p| p.v).collect();
        let Some(&first) = points.first() else { return true };
        let hi = points.iter().fold(first, |a, &p| a.max(p));
        let lo = points.iter().fold(first, |a, &p| a.min(p));
        let size = (hi - lo).length();
        let magnitude = Solid::magnitude(&points);
        for (i, frame) in self.frames.iter().enumerate() {
            let count = self.loops[i].iter().map(Vec::len).sum();
            let tolerance = Solid::plane_tolerance(size, count, magnitude);
            if points.iter().any(|&p| (p - frame.origin).dot(frame.normal) > tolerance) {
                return false;
            }
        }
        true
    }

    /// Every edge as a two-vertex path, before visibility: the curves a
    /// solid carries. Diagonals are not edges of the solid and are left out.
    pub fn edge_paths(&self) -> PolylineSet<WorldSpace> {
        let paths: Vec<[P3<WorldSpace>; 2]> = self
            .edges
            .iter()
            .zip(&self.is_diagonal)
            .filter(|(_, d)| !**d)
            .map(|(e, _)| [self.vertices[e.a], self.vertices[e.b]])
            .collect();
        PolylineSet::from_paths(&paths)
    }

    /// The face piece `p` belongs to.
    pub fn face_index_of_piece(&self, p: usize) -> usize {
        self.face_of_piece[p]
    }

    /// How far from a plane fitted to `count` points a point may lie and
    /// still count as on it, when the region measured is `size` across and
    /// no coordinate exceeds `magnitude`: a billionth of the size, and never
    /// less than the rounding in the coordinates. A coordinate is held to
    /// half a unit in the last place of its magnitude, and the plane's
    /// normal and origin are sums over the points, each adding about as
    /// much, so the floor is `4 count ε magnitude`: a few units in the last
    /// place of the largest coordinate per point. At the origin's scale the
    /// floor is a millionth of the billionth and changes nothing; a
    /// millimetre face a million out is held to its coordinates' 1e-10, not
    /// to 1e-12.
    pub fn plane_tolerance(size: f64, count: usize, magnitude: f64) -> f64 {
        1e-9 * size + 4.0 * count as f64 * f64::EPSILON * magnitude
    }

    /// The largest magnitude of any coordinate of `points`.
    pub fn magnitude(points: &[Vec3]) -> f64 {
        points
            .iter()
            .map(|p| p.abs().reduce_max())
            .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))))
            .unwrap_or(0.0)
    }

    /// Newell's sum over closed loops: twice the area vector of the region
    /// they bound, whose direction is the plane's normal by the right-hand
    /// rule.
    pub fn newell(loops: &[Vec<usize>], vertices: &[P3<WorldSpace>]) -> Vec3 {
        let mut n = Vec3::ZERO;
        for lp in loops {
            for k in 0..lp.len() {
                let p = vertices[lp[k]].v;
                let q = vertices[lp[(k + 1) % lp.len()]].v;
                n += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
            }
        }
        n
    }
}

/// A frame in a face's plane: an origin and two unit axes `u`, `v` with
/// `u × v` the outward normal, so a loop counter-clockwise from outside
/// is counter-clockwise in `(u, v)`. Hatching on the face is laid out in
/// these coordinates.
///
/// The origin is the centroid of the face's vertices and `u` runs along
/// the first side of its outer loop. Both are read off the loops, so the
/// frame, and the hatching laid out in it, does not depend on the
/// partition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceFrame {
    pub normal: Vec3,
    pub origin: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    /// The diagonal of the face's bounding box in `(u, v)`: its size,
    /// which its tolerances are fractions of.
    pub extent: f64,
    /// The flatness tolerance: how far a vertex may lie from the face's
    /// plane and the face still count as flat. A billionth of the
    /// extent, and never less than the rounding the face's coordinates
    /// carry into the measurement (`Solid::plane_tolerance`), so a small
    /// face far from the origin is not refused for the last bits of its
    /// coordinates.
    pub flatness_tolerance: f64,
}

impl FaceFrame {
    fn new(loops: &[Vec<usize>], vertices: &[P3<WorldSpace>], face: usize) -> Result<FaceFrame, SolidError> {
        // Newell's normal over every loop: the holes, running the other
        // way, take their area off the outer loop's.
        let n = Solid::newell(loops, vertices);
        if !(n.length() > 0.0) {
            return refuse(format!("face {face} encloses no area"));
        }
        let normal = n.normalized();
        // Summed in index order, not a set's: the frame, and every hatch
        // line laid out in it, is then the same in every process.
        let ids: BTreeSet<usize> = loops.iter().flatten().copied().collect();
        let origin = ids.iter().fold(Vec3::ZERO, |a, &k| a + vertices[k].v) / ids.len() as f64;
        let along = vertices[loops[0][1]].v - vertices[loops[0][0]].v;
        let u = (along - along.dot(normal) * normal).normalized();
        let v = normal.cross(u);
        let points: Vec<Vec3> = loops.iter().flatten().map(|&k| vertices[k].v).collect();
        let planar: Vec<Vec2> = points.iter().map(|&p| Vec2::new((p - origin).dot(u), (p - origin).dot(v))).collect();
        let hi = planar.iter().fold(planar[0], |a, &p| a.max(p));
        let lo = planar.iter().fold(planar[0], |a, &p| a.min(p));
        let extent = (hi - lo).length();
        let flatness_tolerance = Solid::plane_tolerance(extent, points.len(), Solid::magnitude(&points));
        Ok(FaceFrame { normal, origin, u, v, extent, flatness_tolerance })
    }

    pub fn coordinates(&self, p: Vec3) -> Vec2 {
        Vec2::new((p - self.origin).dot(self.u), (p - self.origin).dot(self.v))
    }

    pub fn point(&self, c: Vec2) -> Vec3 {
        self.origin + c.x * self.u + c.y * self.v
    }

    /// The signed area of a polygon in the face's plane, by the shoelace
    /// formula: positive when it runs counter-clockwise from outside.
    pub fn area(&self, polygon: &[Vec3]) -> f64 {
        let c: Vec<Vec2> = polygon.iter().map(|&p| self.coordinates(p)).collect();
        let n = c.len();
        (0..n).fold(0.0, |a, k| {
            let (p, q) = (c[k], c[(k + 1) % n]);
            a + p.x * q.y - q.x * p.y
        }) / 2.0
    }
}

// MARK: - the mesh and rigid motions

impl Solid {
    /// The pieces as triangles, fanned from each piece's first vertex: what
    /// the depth pass rasterizes. Exact for convex pieces.
    pub fn mesh(&self) -> Mesh<WorldSpace> {
        let mut tris: Vec<[i32; 3]> = Vec::new();
        for piece in self.pieces.iter() {
            for k in 1..piece.len() - 1 {
                tris.push([piece[0] as i32, piece[k] as i32, piece[k + 1] as i32]);
            }
        }
        Mesh::new(self.vertices.clone(), tris)
    }

    /// The solid moved rigidly: rotated about the origin, then translated.
    pub fn transformed(&self, rotation: Mat3, translation: Vec3) -> Solid {
        Solid::new(
            self.vertices.iter().map(|p| P3::from_vec(rotation * p.v + translation)).collect(),
            self.index_loops(),
        )
    }

    /// The loops as the constructor takes them.
    fn index_loops(&self) -> Vec<Vec<Vec<i64>>> {
        self.loops.iter().map(|f| f.iter().map(|l| l.iter().map(|&k| k as i64).collect()).collect()).collect()
    }

    /// The solid set down on a face: rotated so that face's outward normal
    /// points straight down, and lifted so the face lies in the plane
    /// `z = 0`, centred over the origin.
    pub fn resting(&self, i: usize) -> Solid {
        let n = self.normals[self.faces[i][0]];
        let down = Vec3::new(0.0, 0.0, -1.0);
        let axis = n.cross(down);
        let s = axis.length();
        let c = n.dot(down);
        let r = if s < 1e-12 {
            if c > 0.0 {
                Mat3::IDENTITY
            } else {
                Mat3::from_rows(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 0.0, -1.0))
            }
        } else {
            quaternion_matrix(real::atan2(s, c), axis / s)
        };
        let turned: Vec<Vec3> = self.vertices.iter().map(|p| r * p.v).collect();
        let low = turned.iter().map(|p| p.z).reduce(swift::min).unwrap_or(0.0);
        let centre = turned.iter().fold(Vec3::ZERO, |a, &p| a + p) / turned.len().max(1) as f64;
        self.transformed(r, Vec3::new(-centre.x, -centre.y, -low))
    }
}

/// `simd_double3x3(simd_quatd(angle:axis:))`: the rotation by `angle` about
/// the unit `axis`, through the quaternion `(sin(angle/2) axis, cos(angle/2))`
/// and simd's quaternion-to-matrix formula, term for term.
fn quaternion_matrix(angle: f64, axis: Vec3) -> Mat3 {
    let s = real::sin(angle / 2.0);
    let q = Vec4::new(s * axis.x, s * axis.y, s * axis.z, real::cos(angle / 2.0));
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    Mat3::from_columns(
        Vec3::new(x * x - y * y - z * z + w * w, 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)),
        Vec3::new(2.0 * (x * y - z * w), y * y - z * z + w * w - x * x, 2.0 * (y * z + x * w)),
        Vec3::new(2.0 * (z * x + y * w), 2.0 * (y * z - x * w), z * z + w * w - x * x - y * y),
    )
}

// MARK: - visibility

impl Solid {
    /// The solid's closed-form occlusion along `view` (`occlusion(view:)`).
    pub fn occlusion(&self, view: &Transform<WorldSpace, ViewSpace>) -> SolidOcclusion {
        SolidOcclusion::new(self, *view)
    }

    /// The edges as seen along `view`'s projection direction, each cut to
    /// the spans that can be seen.
    ///
    /// Per edge: with `d` the direction back toward the eye and `n₁`, `n₂`
    /// the outward normals of its two pieces, nothing of the edge shows unless
    /// `n₁ · d > 0 ∨ n₂ · d > 0` -- both pieces turned away means the edge is
    /// on the back of the solid, behind the solid itself. On a convex solid
    /// that is the whole test, and an edge is whole or absent. On any other
    /// solid an edge that passes the sign test can still run behind a piece
    /// elsewhere on the body -- the far end of a torus's tunnel seen through
    /// the near one -- so it is tested against every other piece (`Occlusion`).
    pub fn visible_edges(&self, view: &Transform<WorldSpace, ViewSpace>) -> PolylineSet<WorldSpace> {
        let occlusion = self.occlusion(view);
        let d = occlusion.to_eye;
        let mut paths: Vec<[P3<WorldSpace>; 2]> = Vec::new();
        for (k, e) in self.edges.iter().enumerate() {
            if self.is_diagonal[k] {
                continue;
            }
            if !(self.normals[e.left].dot(d) > 0.0 || self.normals[e.right].dot(d) > 0.0) {
                continue;
            }
            let pa = occlusion.projected[e.a];
            let pb = occlusion.projected[e.b];
            let dir = Vec2::new(pb.x() - pa.x(), pb.y() - pa.y());
            if !(dir.length() > occlusion.eps) {
                continue; // seen end-on: a point
            }
            let (wa, wb) = (self.vertices[e.a].v, self.vertices[e.b].v);
            for (a, b) in occlusion.visible(pa, pb, &|f| f == e.left || f == e.right) {
                if b - a > 1e-12 {
                    paths.push([Solid::along(wa, wb, a), Solid::along(wa, wb, b)]);
                }
            }
        }
        PolylineSet::from_paths(&paths)
    }

    /// Curves lying on face `face`, cut to the spans that can be seen.
    ///
    /// Nothing of them shows unless the face faces the eye: a curve on a
    /// back face is behind the solid itself, and one on a face seen edge-on
    /// (`Occlusion.isEdgeOn`) lies on the outline, which is the edges'. What
    /// remains is tested against every piece outside the face -- a piece in
    /// the face's own plane stands in front of nothing on it -- and a path is
    /// kept as one run wherever its consecutive spans meet, so a curve breaks
    /// only where something hides it.
    pub fn visible_curves(
        &self,
        paths: &[Vec<P3<WorldSpace>>],
        face: usize,
        occlusion: &SolidOcclusion,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let members: BTreeSet<usize> = self.faces[face].iter().copied().collect();
        if !(self.normals[self.faces[face][0]].dot(occlusion.to_eye) > 0.0)
            || self.faces[face].iter().all(|&f| occlusion.is_edge_on(f))
        {
            return Vec::new();
        }
        let mut runs = Runs::new();
        for path in paths {
            for w in path.windows(2) {
                let (wa, wb) = (w[0], w[1]);
                let (pa, pb) = (occlusion.view.apply(wa), occlusion.view.apply(wb));
                let seen = occlusion.visible(pa, pb, &|f| members.contains(&f));
                runs.segment(&seen, |t| Solid::along(wa.v, wb.v, t));
            }
            runs.end_path();
        }
        runs.finish()
    }

    /// Curves stored on the solid's surface, along its edges or across its
    /// faces, cut to the spans that can be seen: a segment is seen where a
    /// face whose plane holds both its ends faces the eye, and no piece of
    /// another face stands in front of it. A picked face's boundary is
    /// tested so (`SceneGeometry.boundary(of:faces:)`). A segment in no
    /// face's plane is tested against every piece.
    pub fn visible_curves_on_surface(
        &self,
        paths: &[Vec<P3<WorldSpace>>],
        occlusion: &SolidOcclusion,
    ) -> Vec<Vec<P3<WorldSpace>>> {
        let mut runs = Runs::new();
        for path in paths {
            for w in path.windows(2) {
                let (wa, wb) = (w[0], w[1]);
                let holding: Vec<usize> = (0..self.frames.len())
                    .filter(|&f| {
                        let frame = &self.frames[f];
                        let tolerance = 10.0 * frame.flatness_tolerance;
                        (wa.v - frame.origin).dot(frame.normal).abs() <= tolerance
                            && (wb.v - frame.origin).dot(frame.normal).abs() <= tolerance
                    })
                    .collect();
                let facing = holding.is_empty()
                    || holding.iter().any(|&f| {
                        self.normals[self.faces[f][0]].dot(occlusion.to_eye) > 0.0
                            && !self.faces[f].iter().all(|&p| occlusion.is_edge_on(p))
                    });
                let members: BTreeSet<usize> = holding.iter().flat_map(|&f| self.faces[f].iter().copied()).collect();
                let seen = if facing {
                    let (pa, pb) = (occlusion.view.apply(wa), occlusion.view.apply(wb));
                    occlusion.visible(pa, pb, &|f| members.contains(&f))
                } else {
                    Vec::new()
                };
                runs.segment(&seen, |t| Solid::along(wa.v, wb.v, t));
            }
            runs.end_path();
        }
        runs.finish()
    }

    /// The point `t` of the way from `a` to `b`: the ends themselves at 0
    /// and 1, to the bit, so a segment seen whole is the segment it was.
    pub fn along(a: Vec3, b: Vec3, t: f64) -> P3<WorldSpace> {
        if t == 0.0 {
            P3::from_vec(a)
        } else if t == 1.0 {
            P3::from_vec(b)
        } else {
            P3::from_vec(a + t * (b - a))
        }
    }
}

// MARK: - construction

/// The five Platonic solids (`Solid.Platonic`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platonic {
    Tetrahedron,
    Cube,
    Octahedron,
    Dodecahedron,
    Icosahedron,
}

impl Platonic {
    pub const ALL: [Platonic; 5] =
        [Platonic::Tetrahedron, Platonic::Cube, Platonic::Octahedron, Platonic::Dodecahedron, Platonic::Icosahedron];

    /// Its raw value, the name the CLI takes.
    pub fn name(self) -> &'static str {
        match self {
            Platonic::Tetrahedron => "tetrahedron",
            Platonic::Cube => "cube",
            Platonic::Octahedron => "octahedron",
            Platonic::Dodecahedron => "dodecahedron",
            Platonic::Icosahedron => "icosahedron",
        }
    }

    pub fn from_name(name: &str) -> Option<Platonic> {
        Platonic::ALL.into_iter().find(|p| p.name() == name)
    }
}

impl Solid {
    /// The convex hull of a few points, as a solid: every supporting plane
    /// through three of them, with every point on it as one face, ordered
    /// round the face's centroid and turned away from the rest. Every face
    /// of a convex solid is convex and has no hole, so each is one loop and
    /// one piece.
    ///
    /// Cubic in the number of points and meant for the dozen or two a
    /// polyhedron table has, where it is the difference between faces that
    /// are computed and faces that are typed.
    pub fn convex_hull(points: &[P3<WorldSpace>]) -> Solid {
        assert!(points.len() >= 4, "a solid needs four points");
        let scale = points.iter().map(|p| p.v.length()).reduce(swift::max).unwrap_or(1.0);
        let tol = 1e-9 * swift::max(scale, 1.0);
        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        let mut faces: Vec<Vec<Vec<i64>>> = Vec::new();
        let n = points.len();
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n {
                    let mut normal = (points[j].v - points[i].v).cross(points[k].v - points[i].v);
                    let len = normal.length();
                    if !(len > tol * tol) {
                        continue;
                    }
                    normal /= len;
                    let heights: Vec<f64> = points.iter().map(|p| (p.v - points[i].v).dot(normal)).collect();
                    let above = heights.iter().any(|&h| h > tol);
                    let below = heights.iter().any(|&h| h < -tol);
                    if above && below {
                        continue;
                    }
                    if above {
                        normal = -normal;
                    }
                    let on: Vec<usize> = (0..n).filter(|&m| heights[m].abs() <= tol).collect();
                    if !seen.insert(on.clone()) {
                        continue;
                    }
                    // Round the centroid, counter-clockwise about the normal.
                    let centroid = on.iter().fold(Vec3::ZERO, |a, &m| a + points[m].v) / on.len() as f64;
                    let u = (points[on[0]].v - centroid).normalized();
                    let v = normal.cross(u);
                    let angle = |m: usize| {
                        let a = points[m].v - centroid;
                        real::atan2(a.dot(v), a.dot(u))
                    };
                    let mut sorted = on.clone();
                    // Stable, as Swift's sort is.
                    sorted.sort_by(|&a, &b| swift::ascending(angle(a), angle(b)));
                    faces.push(vec![sorted.iter().map(|&m| m as i64).collect()]);
                }
            }
        }
        Solid::new(points.to_vec(), faces)
    }

    /// One of the five, by its vertex coordinates alone, scaled to the given
    /// circumradius; the faces are found by the hull.
    pub fn platonic(kind: Platonic, circumradius: f64) -> Solid {
        let phi = (1.0 + real::sqrt(5.0)) / 2.0;
        let signs = |body: &dyn Fn(f64, f64, f64) -> Vec3| -> Vec<Vec3> {
            let mut out = Vec::new();
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        out.push(body(sx, sy, sz));
                    }
                }
            }
            out
        };
        let points: Vec<Vec3> = match kind {
            Platonic::Tetrahedron => vec![
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(-1.0, -1.0, 1.0),
            ],
            Platonic::Cube => signs(&Vec3::new),
            Platonic::Octahedron => vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, -1.0),
            ],
            Platonic::Icosahedron => {
                let mut p = Vec::new();
                for s in [-1.0, 1.0] {
                    for t in [-1.0, 1.0] {
                        p.extend([Vec3::new(0.0, s, t * phi), Vec3::new(s, t * phi, 0.0), Vec3::new(t * phi, 0.0, s)]);
                    }
                }
                p
            }
            Platonic::Dodecahedron => {
                let mut p = signs(&Vec3::new);
                for s in [-1.0, 1.0] {
                    for t in [-1.0, 1.0] {
                        p.extend([
                            Vec3::new(0.0, s / phi, t * phi),
                            Vec3::new(s / phi, t * phi, 0.0),
                            Vec3::new(t * phi, 0.0, s / phi),
                        ]);
                    }
                }
                p
            }
        };
        let r = points.iter().map(|p| p.length()).reduce(swift::max).expect("points");
        let scaled: Vec<Vec3> = points.iter().map(|&p| p / r * circumradius).collect();
        // `Array(Set(scaled))` sorted by x, then y, then z: the points once
        // each, in that order.
        let mut unique: Vec<Vec3> = Vec::new();
        for p in scaled {
            if !unique.contains(&p) {
                unique.push(p);
            }
        }
        unique.sort_by(|a, b| {
            let less = if a.x != b.x {
                a.x < b.x
            } else if a.y != b.y {
                a.y < b.y
            } else {
                a.z < b.z
            };
            if less {
                std::cmp::Ordering::Less
            } else if a == b {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Greater
            }
        });
        Solid::convex_hull(&unique.into_iter().map(P3::from_vec).collect::<Vec<_>>())
    }

    /// The Császár polyhedron: the seven-vertex torus, the fewest vertices a
    /// flat-faced torus can have, and the only polyhedron besides the
    /// tetrahedron with no diagonals -- every pair of its vertices is an
    /// edge, K₇ drawn on the torus.
    ///
    /// Combinatorially unique: vertices mod 7, faces {i, i+1, i+3} and
    /// {i, i+2, i+3}, 21 edges, 14 triangles. The coordinates are Császár's
    /// own (1949), with the faces as Szilassi tabulates them (Bridges 2008);
    /// of the 5040 ways to put that abstract triangulation on these seven
    /// points, exactly the 42 automorphic ones embed, so the face list is
    /// determined by the coordinates. It is a torus, not a ball, and not
    /// convex: the sign test is necessary and not sufficient, and the
    /// occlusion test earns its keep.
    pub fn csaszar() -> Solid {
        let v = [
            (-3.0, 3.0, 0.0),
            (-3.0, -3.0, 1.0),
            (-1.0, -2.0, 3.0),
            (1.0, 2.0, 3.0),
            (3.0, 3.0, 1.0),
            (3.0, -3.0, 0.0),
            (0.0, 0.0, 15.0),
        ];
        // Szilassi's list, one-based and inward; reversed here to face out.
        let listed: [[i64; 3]; 14] = [
            [1, 2, 6],
            [1, 4, 2],
            [5, 3, 2],
            [4, 1, 3],
            [2, 7, 6],
            [3, 7, 2],
            [1, 7, 3],
            [6, 5, 1],
            [6, 3, 5],
            [2, 4, 5],
            [3, 6, 4],
            [5, 7, 1],
            [4, 7, 5],
            [6, 7, 4],
        ];
        Solid::new(
            v.iter().map(|&(x, y, z)| P3::new(x, y, z)).collect(),
            listed.iter().map(|f| vec![f.iter().rev().map(|k| k - 1).collect()]).collect(),
        )
    }

    /// A picture frame: a square `outer` across with a square hole `inner`
    /// across, `depth` thick, lying flat on `z = 0` and centred.
    ///
    /// The other torus here, and the plainer test of everything a convex
    /// solid never asks. Its front and back faces are annuli, each an outer
    /// loop and a hole, which no convex polygon is: the partition cuts each
    /// into convex pieces by diagonals, which must not be drawn and must hide
    /// nothing. Its hole is a window: the far inner wall and the far edges of
    /// the back are seen through it, cut by the near part of the front, so an
    /// edge's visible spans are decided by occlusion and not by the signs. And
    /// every edge lies in the planes of the faces beside it, where the gap
    /// between curve and face is exactly zero.
    pub fn frame(outer: f64, inner: f64, depth: f64) -> Solid {
        assert!(
            inner > 0.0 && inner < outer && depth > 0.0,
            "a frame has a hole smaller than itself and some thickness"
        );
        let (a, b) = (outer / 2.0, inner / 2.0);
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let mut v: Vec<P3<WorldSpace>> = Vec::new();
        for z in [0.0, depth] {
            for r in [a, b] {
                for (cx, cy) in corners {
                    v.push(P3::new(cx * r, cy * r, z));
                }
            }
        }
        // Outer bottom 0-3, inner bottom 4-7, outer top 8-11, inner top 12-15,
        // each ring counter-clockwise seen from above.
        let ob = |k: i64| k % 4;
        let ib = |k: i64| 4 + k % 4;
        let ot = |k: i64| 8 + k % 4;
        let it = |k: i64| 12 + k % 4;
        // The top seen from above, the bottom from below: each outer loop
        // counter-clockwise from outside and its hole clockwise.
        let mut faces: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![ot(0), ot(1), ot(2), ot(3)], vec![it(0), it(3), it(2), it(1)]],
            vec![vec![ob(1), ob(0), ob(3), ob(2)], vec![ib(1), ib(2), ib(3), ib(0)]],
        ];
        for k in 0..4 {
            faces.push(vec![vec![ob(k), ob(k + 1), ot(k + 1), ot(k)]]); // outer wall
            faces.push(vec![vec![ib(k + 1), ib(k), it(k), it(k + 1)]]); // inner wall, facing the hole
        }
        Solid::new(v, faces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> (Vec<P3<WorldSpace>>, Vec<Vec<Vec<i64>>>) {
        let v: Vec<P3<WorldSpace>> =
            (0..8).map(|k| P3::new((k & 1) as f64, ((k >> 1) & 1) as f64, ((k >> 2) & 1) as f64)).collect();
        let faces = vec![
            vec![vec![0, 2, 3, 1]],
            vec![vec![4, 5, 7, 6]],
            vec![vec![0, 1, 5, 4]],
            vec![vec![2, 6, 7, 3]],
            vec![vec![0, 4, 6, 2]],
            vec![vec![1, 3, 7, 5]],
        ];
        (v, faces)
    }

    #[test]
    fn a_cube_is_a_convex_sphere() {
        let (v, f) = cube();
        let s = Solid::checking(v, f).unwrap();
        assert_eq!(s.pieces.len(), 6);
        assert_eq!(s.edges.len(), 12);
        assert!(s.is_diagonal.iter().all(|d| !d));
        assert_eq!(s.euler_characteristic(), 2);
        assert_eq!(s.genus(), Some(0));
        assert!(s.is_convex());
        assert_eq!(s.edge_paths().count(), 12);
    }

    #[test]
    fn what_bounds_no_solid_is_refused() {
        let (v, mut f) = cube();
        f[0][0].reverse();
        assert!(Solid::checking(v.clone(), f).unwrap_err().0.contains("traversed twice"));
        let (_, mut f) = cube();
        f.pop();
        assert!(Solid::checking(v.clone(), f).unwrap_err().0.contains("bounds only one face"));
        let (_, mut f) = cube();
        f[0][0][0] = 8;
        assert_eq!(Solid::checking(v, f).unwrap_err().0, "face 0: loop 0 names vertex 8, and there are 8");
    }
}
