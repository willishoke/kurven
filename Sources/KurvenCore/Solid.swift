import Foundation
import simd

/// A polyhedron bounding a solid: flat faces, sharp edges, and its ink is
/// its edges and whatever is ruled on its faces.
///
/// The third kind of geometry a plate can draw, beside a heightfield and a
/// parametric surface, and the one whose visibility is exact in the plainest
/// sense. A point of an edge is seen when some face at that edge faces the
/// eye and no other face stands between the point and the eye; a point of
/// ink in a face is seen when that face faces the eye and nothing stands in
/// front. Both halves are decided in closed form: the first by signs, the
/// second by casting the ink against every face -- the ink and the face are
/// both flat, so where one passes behind the other is an interval, solved,
/// not searched. No margin, no pixel, no lattice: the drawing is the same at
/// any resolution, which is the property every other visibility rule here is
/// held to and this one has by construction.
///
/// Faces are convex polygons, listed counter-clockwise seen from outside, so
/// a face's normal by the right-hand rule points out of the solid. That is
/// checked, not trusted: a solid whose faces do not close consistently, or
/// whose faces are not convex, is refused, and the Platonic tables are built
/// from vertex coordinates alone by a convex hull, so no face is typed in by
/// hand.
///
/// A flat face that is not convex -- the front of a picture frame, a square
/// with a square hole, which no polygon can be -- is several convex faces in
/// one plane. The edges between them are *soft*: seams of the face list, not
/// creases of the solid. They are not ink, and a coplanar face hides nothing
/// of its neighbour. The faces of one plane joined across soft edges are a
/// *facet*, and ink ruled on a face is ruled over its whole facet, so a
/// hatch runs unbroken across the seams and breaks only at the hole.
public struct Solid: Sendable, Equatable {
    public let vertices: [P3<WorldSpace>]
    /// Each face as its vertex indices in order, counter-clockwise from
    /// outside.
    public let faces: [[Int]]
    /// Each edge once, with the two faces it bounds.
    public let edges: [Edge]
    /// Unit outward normals, one per face.
    public let normals: [SIMD3<Double>]
    /// Per edge, whether it is a crease: its two faces lie in different
    /// planes. The rest are soft, seams between coplanar faces.
    public let crease: [Bool]
    /// The facets: the faces of one plane joined across soft edges, each as
    /// its face indices. A face with no soft edge is a facet of one.
    public let facets: [[Int]]

    public struct Edge: Sendable, Equatable, Hashable {
        public let a: Int, b: Int
        /// The face in which the edge runs `a -> b`, and the one in which it
        /// runs `b -> a`.
        public let left: Int, right: Int
    }

    /// - Precondition: every edge lies in exactly two faces, traversed in
    ///   opposite directions -- the surface is closed and consistently
    ///   oriented -- the orientation is outward (the signed volume is
    ///   positive), and every face is convex.
    public init(vertices: [P3<WorldSpace>], faces: [[Int]]) {
        precondition(faces.allSatisfy { $0.count >= 3 }, "a face has at least three vertices")
        self.vertices = vertices
        self.faces = faces
        var directed: [SIMD2<Int>: Int] = [:]
        for (f, face) in faces.enumerated() {
            for k in 0..<face.count {
                let e = SIMD2(face[k], face[(k + 1) % face.count])
                precondition(directed[e] == nil, "edge \(e) is traversed twice the same way")
                directed[e] = f
            }
        }
        var edges: [Edge] = []
        for (e, left) in directed where e.x < e.y {
            guard let right = directed[SIMD2(e.y, e.x)] else {
                preconditionFailure("edge \(e) bounds only one face: the surface is not closed")
            }
            edges.append(Edge(a: e.x, b: e.y, left: left, right: right))
        }
        precondition(directed.count == 2 * edges.count,
                     "an edge bounds one face only: the surface is not closed")
        self.edges = edges.sorted { ($0.a, $0.b) < ($1.a, $1.b) }
        // Newell's normal: exact for a planar polygon however many vertices
        // it has, and the sum it is is the one the divergence theorem wants.
        let scale = vertices.map { simd_length($0.v) }.max() ?? 1
        var normals: [SIMD3<Double>] = []
        var volume = 0.0
        for face in faces {
            var n = SIMD3<Double>.zero
            for k in 0..<face.count {
                let p = vertices[face[k]].v, q = vertices[face[(k + 1) % face.count]].v
                n += SIMD3((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x),
                           (p.x - q.x) * (p.y + q.y))
            }
            let len = simd_length(n)
            precondition(len > 0, "a face is degenerate")
            n /= len
            normals.append(n)
            // Convex: every corner turns the same way about the normal.
            for k in 0..<face.count {
                let p = vertices[face[k]].v, q = vertices[face[(k + 1) % face.count]].v,
                    r = vertices[face[(k + 2) % face.count]].v
                precondition(simd_dot(simd_cross(q - p, r - q), n) >= -1e-9 * scale * scale,
                             "face \(face) is not convex at vertex \(face[(k + 1) % face.count])")
            }
            // (n is twice the area vector; p · n / 6 is the cone's volume.)
            volume += simd_dot(vertices[face[0]].v, n * len) / 6
        }
        precondition(volume > 0, "the faces are oriented inward (signed volume \(volume))")
        self.normals = normals
        let crease = self.edges.map { simd_dot(normals[$0.left], normals[$0.right]) < 1 - 1e-9 }
        self.crease = crease
        // Facets: the faces, joined across soft edges.
        var parent = Array(0..<faces.count)
        func root(_ i: Int) -> Int {
            var i = i
            while parent[i] != i { parent[i] = parent[parent[i]]; i = parent[i] }
            return i
        }
        for (k, e) in self.edges.enumerated() where !crease[k] {
            parent[root(e.left)] = root(e.right)
        }
        var byRoot: [Int: [Int]] = [:]
        var order: [Int] = []
        for f in 0..<faces.count {
            let r = root(f)
            if byRoot[r] == nil { order.append(r) }
            byRoot[r, default: []].append(f)
        }
        self.facets = order.map { byRoot[$0]! }
    }

    public var eulerCharacteristic: Int { vertices.count - edges.count + faces.count }

    /// Whether every vertex lies on or behind every face's plane.
    public var isConvex: Bool {
        let scale = vertices.map { simd_length($0.v) }.max() ?? 1
        for (f, face) in faces.enumerated() {
            let o = vertices[face[0]].v
            for v in vertices where simd_dot(v.v - o, normals[f]) > 1e-9 * scale { return false }
        }
        return true
    }

    /// The faces as triangles, fanned from each face's first vertex: what
    /// the depth pass rasterizes. Exact for convex faces.
    public var mesh: Mesh<WorldSpace> {
        var tris: [SIMD3<Int32>] = []
        for face in faces {
            for k in 1..<(face.count - 1) {
                tris.append(SIMD3(Int32(face[0]), Int32(face[k]), Int32(face[k + 1])))
            }
        }
        return Mesh(vertices: vertices, triangles: tris)
    }

    /// Every crease as a two-vertex path, unjudged: the edge ink a solid
    /// carries. Soft edges are seams, not ink.
    public var edgePaths: PolylineSet<WorldSpace> {
        PolylineSet(paths: zip(edges, crease).filter { $1 }.map { [vertices[$0.0.a], vertices[$0.0.b]] })
    }

    /// The solid moved rigidly: rotated about the origin, then translated.
    public func transformed(rotation: simd_double3x3, translation: SIMD3<Double> = .zero) -> Solid {
        Solid(vertices: vertices.map { P3(rotation * $0.v + translation) }, faces: faces)
    }

    /// The solid set down on a face: rotated so that face's outward normal
    /// points straight down, and lifted so the face lies in the plane
    /// `z = 0`, centred over the origin.
    public func resting(on face: Int) -> Solid {
        let n = normals[face]
        let down = SIMD3<Double>(0, 0, -1)
        let axis = simd_cross(n, down)
        let s = simd_length(axis), c = simd_dot(n, down)
        let r: simd_double3x3
        if s < 1e-12 {
            r = c > 0 ? matrix_identity_double3x3
                      : simd_double3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, -1, 0), SIMD3(0, 0, -1)])
        } else {
            r = simd_double3x3(simd_quatd(angle: atan2(s, c), axis: axis / s))
        }
        let turned = vertices.map { r * $0.v }
        let low = turned.map(\.z).min() ?? 0
        let centre = turned.reduce(SIMD3<Double>.zero, +) / Double(max(turned.count, 1))
        return transformed(rotation: r, translation: SIMD3(-centre.x, -centre.y, -low))
    }

    // MARK: - facets

    /// A facet with a frame in its plane: an origin and two unit axes `u`,
    /// `v` with `u × v` the outward normal, so a polygon counter-clockwise
    /// from outside is counter-clockwise in `(u, v)`. Ink ruled on the
    /// facet is ruled in these coordinates.
    public struct Facet: Sendable {
        public let faces: [Int]
        public let normal: SIMD3<Double>
        public let origin: SIMD3<Double>
        public let u: SIMD3<Double>, v: SIMD3<Double>

        public func coordinates(_ p: SIMD3<Double>) -> SIMD2<Double> {
            SIMD2(simd_dot(p - origin, u), simd_dot(p - origin, v))
        }
        public func point(_ c: SIMD2<Double>) -> SIMD3<Double> {
            origin + c.x * u + c.y * v
        }
    }

    /// Facet `i`, framed: the origin is the centroid of its vertices and
    /// `u` runs along the first edge of its first face.
    public func facet(_ i: Int) -> Facet {
        let members = facets[i]
        let n = normals[members[0]]
        var ids = Set<Int>()
        for f in members { for k in faces[f] { ids.insert(k) } }
        let centroid = ids.reduce(SIMD3<Double>.zero) { $0 + vertices[$1].v } / Double(ids.count)
        let first = faces[members[0]]
        let along = vertices[first[1]].v - vertices[first[0]].v
        let u = simd_normalize(along - simd_dot(along, n) * n)
        return Facet(faces: members, normal: n, origin: centroid, u: u, v: simd_cross(n, u))
    }

    /// The facet face `f` belongs to.
    public func facetIndex(of f: Int) -> Int {
        facets.firstIndex { $0.contains(f) }!
    }

    // MARK: - visibility

    /// The faces projected to the picture plane, ready to be cast against:
    /// built once per camera, then asked about any flat segment.
    ///
    /// Under an orthographic camera a segment and a face both project to
    /// the picture plane, their overlap along the segment is an interval,
    /// and over it the difference of their view depths is linear: where that
    /// difference says the face is nearer, the segment is hidden. The hidden
    /// intervals are subtracted from the segment, exactly.
    public struct Cast: Sendable {
        let solid: Solid
        public let view: Transform<WorldSpace, ViewSpace>
        let projected: [P3<ViewSpace>]
        /// Twice each face's signed area in projection; zero when seen
        /// edge-on, where it hides nothing.
        let area2: [Double]
        let scale: Double
        let eps: Double
        /// The direction back toward the eye, in world space.
        public let toEye: SIMD3<Double>

        init(_ solid: Solid, view: Transform<WorldSpace, ViewSpace>) {
            self.solid = solid
            self.view = view
            toEye = -view.sightLine
            let projected = solid.vertices.map { view($0) }
            self.projected = projected
            scale = max(projected.map { simd_length($0.v) }.max() ?? 1, 1)
            eps = 1e-9 * scale
            area2 = solid.faces.map { face in
                var a = 0.0
                for k in 0..<face.count {
                    let p = projected[face[k]], q = projected[face[(k + 1) % face.count]]
                    a += p.x * q.y - q.x * p.y
                }
                return a
            }
        }

        /// The sub-intervals of `[0, 1]` along the segment `pa -> pb` (in
        /// view space) over which a face not `excluded` stands in front:
        /// sorted and disjoint.
        public func hidden(from pa: P3<ViewSpace>, to pb: P3<ViewSpace>,
                           excluding: (Int) -> Bool) -> [(Double, Double)] {
            var hidden: [(Double, Double)] = []
            for (f, face) in solid.faces.enumerated() where !excluding(f) {
                // A face seen edge-on projects to nothing and hides nothing.
                guard abs(area2[f]) > 1e-9 * scale * scale else { continue }
                let sign: Double = area2[f] > 0 ? 1 : -1
                // The segment's overlap with the face's projection: clipped
                // against each side's half-plane.
                var lo = 0.0, hi = 1.0
                for k in 0..<face.count {
                    let p = projected[face[k]], q = projected[face[(k + 1) % face.count]]
                    let ex = q.x - p.x, ey = q.y - p.y
                    // Inside is to the left of p -> q when the polygon winds
                    // positively, to the right otherwise.
                    let f0 = sign * (ex * (pa.y - p.y) - ey * (pa.x - p.x))
                    let f1 = sign * (ex * (pb.y - p.y) - ey * (pb.x - p.x))
                    // f(t) = f0 + t (f1 - f0) must be >= 0.
                    if f0 < 0 && f1 < 0 { lo = 1; hi = 0; break }
                    if f0 < 0 { lo = max(lo, f0 / (f0 - f1)) }
                    if f1 < 0 { hi = min(hi, f0 / (f0 - f1)) }
                }
                guard lo < hi else { continue }
                // The face's view depth over the picture plane is affine:
                // z = z0 + (gx, gy) · (x - x0, y - y0), from the plane's
                // normal in view space.
                let o = projected[face[0]].v
                let n = simd_cross(projected[face[1]].v - o, projected[face[2]].v - o)
                guard abs(n.z) > 1e-9 * scale * scale else { continue }
                func gap(_ t: Double) -> Double {
                    let p = pa.v + t * (pb.v - pa.v)
                    let z = o.z - (n.x * (p.x - o.x) + n.y * (p.y - o.y)) / n.z
                    return z - p.z
                }
                // gap > eps means the face is in front, and gap is linear.
                let g0 = gap(lo), g1 = gap(hi)
                var a = lo, b = hi
                if g0 <= eps && g1 <= eps { continue }
                if g0 <= eps { a = lo + (hi - lo) * (eps - g0) / (g1 - g0) }
                if g1 <= eps { b = lo + (hi - lo) * (eps - g0) / (g1 - g0) }
                if a < b { hidden.append((a, b)) }
            }
            hidden.sort { $0.0 < $1.0 }
            var merged: [(Double, Double)] = []
            for (a, b) in hidden {
                if let last = merged.last, a <= last.1 {
                    merged[merged.count - 1].1 = max(last.1, b)
                } else {
                    merged.append((a, b))
                }
            }
            return merged
        }

        /// The complement: the pieces of the segment that can be seen, as
        /// fractions of the way along it.
        public func visible(from pa: P3<ViewSpace>, to pb: P3<ViewSpace>,
                            excluding: (Int) -> Bool) -> [(Double, Double)] {
            var t = 0.0
            var pieces: [(Double, Double)] = []
            for (a, b) in hidden(from: pa, to: pb, excluding: excluding) {
                if a > t { pieces.append((t, a)) }
                t = max(t, b)
            }
            if t < 1 { pieces.append((t, 1)) }
            return pieces
        }
    }

    public func cast(view: Transform<WorldSpace, ViewSpace>) -> Cast { Cast(self, view: view) }

    /// The creases as seen along `view`'s sight line, each cut to the pieces
    /// that can be seen.
    ///
    /// Per edge: with `d` the direction back toward the eye and `n₁`, `n₂`
    /// the outward normals of its two faces, nothing of the edge shows unless
    /// `n₁ · d > 0 ∨ n₂ · d > 0` -- both faces turned away means the edge is
    /// on the back of the solid, behind the solid itself. On a convex solid
    /// that is the whole test, and an edge is whole or absent. On any other
    /// solid an edge that passes the sign test can still run behind a face
    /// elsewhere on the body -- the far rim of a torus's tunnel seen through
    /// the near one -- so every other face is cast against it (`Cast`).
    public func visibleEdges(view: Transform<WorldSpace, ViewSpace>) -> PolylineSet<WorldSpace> {
        let cast = self.cast(view: view)
        let d = cast.toEye
        var paths: [[P3<WorldSpace>]] = []
        for (k, e) in edges.enumerated() where crease[k] {
            guard simd_dot(normals[e.left], d) > 0 || simd_dot(normals[e.right], d) > 0 else { continue }
            let pa = cast.projected[e.a], pb = cast.projected[e.b]
            let dir = SIMD2(pb.x - pa.x, pb.y - pa.y)
            guard simd_length(dir) > cast.eps else { continue }   // seen end-on: a point
            let wa = vertices[e.a].v, wb = vertices[e.b].v
            for (a, b) in cast.visible(from: pa, to: pb, excluding: { $0 == e.left || $0 == e.right })
                where b - a > 1e-12 {
                paths.append([Solid.along(wa, wb, a), Solid.along(wa, wb, b)])
            }
        }
        return PolylineSet(paths: paths)
    }

    /// Ink lying in facet `facet`, cut to the pieces that can be seen.
    ///
    /// Nothing of it shows unless the facet faces the eye: ink in a back
    /// face is behind the solid itself. What remains is cast against every
    /// face outside the facet -- a face in the facet's own plane stands in
    /// front of nothing in it -- and a path is kept as one run wherever its
    /// consecutive pieces meet, so a stroke breaks only where something
    /// hides it.
    public func visibleInk(_ paths: [[P3<WorldSpace>]], inFacet facet: Int,
                           cast: Cast) -> [[P3<WorldSpace>]] {
        let members = Set(facets[facet])
        guard simd_dot(normals[facets[facet][0]], cast.toEye) > 0 else { return [] }
        var out: [[P3<WorldSpace>]] = []
        for path in paths {
            var run: [P3<WorldSpace>] = []
            var openAtEnd = false
            for (wa, wb) in zip(path, path.dropFirst()) {
                let pa = cast.view(wa), pb = cast.view(wb)
                for (a, b) in cast.visible(from: pa, to: pb, excluding: { members.contains($0) })
                    where b - a > 1e-12 {
                    let start = Solid.along(wa.v, wb.v, a), end = Solid.along(wa.v, wb.v, b)
                    if openAtEnd, a == 0, !run.isEmpty {
                        run.append(end)
                    } else {
                        if run.count >= 2 { out.append(run) }
                        run = [start, end]
                    }
                    openAtEnd = b == 1
                }
                if !openAtEnd {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return out
    }

    /// The point `t` of the way from `a` to `b`: the ends themselves at 0
    /// and 1, to the bit, so a segment seen whole is the segment it was.
    static func along(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ t: Double) -> P3<WorldSpace> {
        t == 0 ? P3(a) : (t == 1 ? P3(b) : P3(a + t * (b - a)))
    }

    // MARK: - construction

    /// The convex hull of a few points, as a solid: every supporting plane
    /// through three of them, with every point on it as one face, ordered
    /// round the face's centroid and turned to face away from the rest.
    ///
    /// Cubic in the number of points and meant for the dozen or two a
    /// polyhedron table has, where it is the difference between faces that
    /// are computed and faces that are typed.
    public static func convexHull(of points: [P3<WorldSpace>]) -> Solid {
        precondition(points.count >= 4, "a solid needs four points")
        let scale = points.map { simd_length($0.v) }.max() ?? 1
        let tol = 1e-9 * max(scale, 1)
        var seen = Set<[Int]>()
        var faces: [[Int]] = []
        let n = points.count
        for i in 0..<n {
            for j in (i + 1)..<n {
                for k in (j + 1)..<n {
                    var normal = simd_cross(points[j].v - points[i].v, points[k].v - points[i].v)
                    let len = simd_length(normal)
                    guard len > tol * tol else { continue }
                    normal /= len
                    let heights = points.map { simd_dot($0.v - points[i].v, normal) }
                    let above = heights.contains { $0 > tol }
                    let below = heights.contains { $0 < -tol }
                    guard !(above && below) else { continue }
                    if above { normal = -normal }
                    let on = (0..<n).filter { abs(heights[$0]) <= tol }
                    guard seen.insert(on).inserted else { continue }
                    // Round the centroid, counter-clockwise about the normal.
                    let centroid = on.reduce(SIMD3<Double>.zero) { $0 + points[$1].v } / Double(on.count)
                    let u = simd_normalize(points[on[0]].v - centroid)
                    let v = simd_cross(normal, u)
                    faces.append(on.sorted {
                        let a = points[$0].v - centroid, b = points[$1].v - centroid
                        return atan2(simd_dot(a, v), simd_dot(a, u)) < atan2(simd_dot(b, v), simd_dot(b, u))
                    })
                }
            }
        }
        return Solid(vertices: points, faces: faces)
    }

    public enum Platonic: String, CaseIterable, Sendable {
        case tetrahedron, cube, octahedron, dodecahedron, icosahedron
    }

    /// One of the five, by its vertex coordinates alone, scaled to the given
    /// circumradius; the faces are found by the hull.
    public static func platonic(_ kind: Platonic, circumradius: Double = 1) -> Solid {
        let phi = (1 + 5.0.squareRoot()) / 2
        var points: [SIMD3<Double>]
        func signs(_ body: (Double, Double, Double) -> SIMD3<Double>) -> [SIMD3<Double>] {
            var out: [SIMD3<Double>] = []
            for sx in [-1.0, 1.0] { for sy in [-1.0, 1.0] { for sz in [-1.0, 1.0] {
                out.append(body(sx, sy, sz))
            } } }
            return out
        }
        switch kind {
        case .tetrahedron:
            points = [SIMD3(1, 1, 1), SIMD3(1, -1, -1), SIMD3(-1, 1, -1), SIMD3(-1, -1, 1)]
        case .cube:
            points = signs { SIMD3($0, $1, $2) }
        case .octahedron:
            points = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0),
                      SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
        case .icosahedron:
            points = []
            for s in [-1.0, 1.0] { for t in [-1.0, 1.0] {
                points += [SIMD3(0, s, t * phi), SIMD3(s, t * phi, 0), SIMD3(t * phi, 0, s)]
            } }
        case .dodecahedron:
            points = signs { SIMD3($0, $1, $2) }
            for s in [-1.0, 1.0] { for t in [-1.0, 1.0] {
                points += [SIMD3(0, s / phi, t * phi), SIMD3(s / phi, t * phi, 0), SIMD3(t * phi, 0, s / phi)]
            } }
        }
        let r = points.map { simd_length($0) }.max()!
        let scaled: [SIMD3<Double>] = points.map { $0 / r * circumradius }
        var unique: [SIMD3<Double>] = Array(Set(scaled))
        unique.sort { (a: SIMD3<Double>, b: SIMD3<Double>) -> Bool in
            a.x != b.x ? a.x < b.x : (a.y != b.y ? a.y < b.y : a.z < b.z)
        }
        return convexHull(of: unique.map { P3<WorldSpace>($0) })
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
    /// convex: the sign test is necessary and not sufficient, and the edge
    /// cast earns its keep.
    public static var csaszar: Solid {
        let v: [SIMD3<Double>] = [
            SIMD3(-3, 3, 0), SIMD3(-3, -3, 1), SIMD3(-1, -2, 3), SIMD3(1, 2, 3),
            SIMD3(3, 3, 1), SIMD3(3, -3, 0), SIMD3(0, 0, 15),
        ]
        // Szilassi's list, one-based and inward; reversed here to face out.
        let listed = [[1, 2, 6], [1, 4, 2], [5, 3, 2], [4, 1, 3], [2, 7, 6], [3, 7, 2], [1, 7, 3],
                      [6, 5, 1], [6, 3, 5], [2, 4, 5], [3, 6, 4], [5, 7, 1], [4, 7, 5], [6, 7, 4]]
        return Solid(vertices: v.map { P3($0) },
                     faces: listed.map { $0.reversed().map { $0 - 1 } })
    }

    /// A picture frame: a square `outer` across with a square hole `inner`
    /// across, `depth` thick, lying flat on `z = 0` and centred.
    ///
    /// The other torus on the table, and the plainer trial of everything a
    /// convex solid never asks. Its front and back are annuli, which no
    /// convex polygon is, so each is four trapezoids with four soft edges
    /// between them: seams that must not be drawn and must hide nothing.
    /// Its hole is a window: the far inner wall and the far rim of the back
    /// are seen through it, cut by the near part of the front, so an edge's
    /// visible pieces are decided by the cast and not by the signs. And its
    /// rims lie in the planes of the faces beside them, where the gap
    /// between ink and face is exactly zero.
    public static func frame(outer: Double, inner: Double, depth: Double) -> Solid {
        precondition(inner > 0 && inner < outer && depth > 0, "a frame has a hole smaller than itself and some thickness")
        let a = outer / 2, b = inner / 2
        let corners = [SIMD2(-1.0, -1.0), SIMD2(1.0, -1.0), SIMD2(1.0, 1.0), SIMD2(-1.0, 1.0)]
        var v: [P3<WorldSpace>] = []
        for z in [0.0, depth] {
            for r in [a, b] {
                for c in corners { v.append(P3(c.x * r, c.y * r, z)) }
            }
        }
        // Outer bottom 0-3, inner bottom 4-7, outer top 8-11, inner top 12-15,
        // each ring counter-clockwise seen from above.
        func ob(_ k: Int) -> Int { k % 4 }
        func ib(_ k: Int) -> Int { 4 + k % 4 }
        func ot(_ k: Int) -> Int { 8 + k % 4 }
        func it(_ k: Int) -> Int { 12 + k % 4 }
        var faces: [[Int]] = []
        for k in 0..<4 {
            faces.append([ot(k), ot(k + 1), it(k + 1), it(k)])     // top, outward up
            faces.append([ob(k + 1), ob(k), ib(k), ib(k + 1)])     // bottom, outward down
            faces.append([ob(k), ob(k + 1), ot(k + 1), ot(k)])     // outer wall
            faces.append([ib(k + 1), ib(k), it(k), it(k + 1)])     // inner wall, facing the hole
        }
        return Solid(vertices: v, faces: faces)
    }
}
