import Foundation
import simd

/// A polyhedron: a solid bounded by flat faces. Its curves are its edges
/// and whatever is hatched on its faces.
///
/// The third kind of geometry kurven draws, beside a heightfield and a
/// parametric surface, and the one whose visibility is exact in the plainest
/// sense. A point of an edge is seen when one of the two faces at that edge
/// faces the eye and nothing stands between the point and the eye; a point
/// of a curve on a face is seen when that face faces the eye and nothing
/// stands in front. Both halves are decided in closed form: the first by
/// signs, the second by occlusion against every piece (`Occlusion`) -- the
/// curve and the piece are both flat, so where one passes behind the other
/// is an interval, solved, not searched. No tolerance, no pixel, no grid:
/// the drawing is the same at any resolution.
///
/// The solid is given as a *convex partition* of its boundary: convex
/// polygons, the *pieces*, listed counter-clockwise seen from outside, so a
/// piece's normal by the right-hand rule points out of the solid. That is
/// checked, not trusted: a solid whose pieces do not close consistently, or
/// whose pieces are not convex, is refused, and the Platonic tables are built
/// from vertex coordinates alone by a convex hull, so no piece is typed in by
/// hand.
///
/// A face is the pieces of one plane that share sides. A convex face is one
/// piece; a face that is not convex -- the front of a picture frame, a square
/// with a square hole -- is several. A side shared by two pieces of one face
/// is a *diagonal* of the partition, not an edge of the solid: it is not
/// drawn, and a piece hides nothing of another in its own plane. Hatching is
/// laid over a whole face, so a hatch line runs unbroken across the
/// diagonals and breaks only at the hole.
public struct Solid: Sendable, Equatable {
    public let vertices: [P3<WorldSpace>]
    /// Each piece as its vertex indices in order, counter-clockwise from
    /// outside.
    public let pieces: [[Int]]
    /// Every side of every piece once, with the two pieces it bounds: the
    /// solid's edges and the partition's diagonals.
    public let edges: [Edge]
    /// Unit outward normals, one per piece.
    public let normals: [SIMD3<Double>]
    /// Per entry of `edges`, whether it is a diagonal: its two pieces lie
    /// in one plane.
    public let isDiagonal: [Bool]
    /// The faces, each as its piece indices: the pieces of one plane joined
    /// across diagonals. A convex face is a face of one piece.
    public let faces: [[Int]]

    public struct Edge: Sendable, Equatable, Hashable {
        public let a: Int, b: Int
        /// The piece in which the edge runs `a -> b`, and the one in which it
        /// runs `b -> a`.
        public let left: Int, right: Int
    }

    /// - Precondition: every edge lies in exactly two pieces, traversed in
    ///   opposite directions -- the surface is closed and consistently
    ///   oriented -- the orientation is outward (the signed volume is
    ///   positive), and every piece is convex.
    public init(vertices: [P3<WorldSpace>], pieces: [[Int]]) {
        precondition(pieces.allSatisfy { $0.count >= 3 }, "a piece has at least three vertices")
        self.vertices = vertices
        self.pieces = pieces
        var directed: [SIMD2<Int>: Int] = [:]
        for (f, piece) in pieces.enumerated() {
            for k in 0..<piece.count {
                let e = SIMD2(piece[k], piece[(k + 1) % piece.count])
                precondition(directed[e] == nil, "edge \(e) is traversed twice the same way")
                directed[e] = f
            }
        }
        var edges: [Edge] = []
        for (e, left) in directed where e.x < e.y {
            guard let right = directed[SIMD2(e.y, e.x)] else {
                preconditionFailure("edge \(e) bounds only one piece: the surface is not closed")
            }
            edges.append(Edge(a: e.x, b: e.y, left: left, right: right))
        }
        precondition(directed.count == 2 * edges.count,
                     "an edge bounds one piece only: the surface is not closed")
        self.edges = edges.sorted { ($0.a, $0.b) < ($1.a, $1.b) }
        // Newell's normal: exact for a planar polygon however many vertices
        // it has, and the sum it is is the one the divergence theorem wants.
        let scale = vertices.map { simd_length($0.v) }.max() ?? 1
        var normals: [SIMD3<Double>] = []
        var volume = 0.0
        for piece in pieces {
            var n = SIMD3<Double>.zero
            for k in 0..<piece.count {
                let p = vertices[piece[k]].v, q = vertices[piece[(k + 1) % piece.count]].v
                n += SIMD3((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x),
                           (p.x - q.x) * (p.y + q.y))
            }
            let len = simd_length(n)
            precondition(len > 0, "a piece is degenerate")
            n /= len
            normals.append(n)
            // Convex: every corner turns the same way about the normal.
            for k in 0..<piece.count {
                let p = vertices[piece[k]].v, q = vertices[piece[(k + 1) % piece.count]].v,
                    r = vertices[piece[(k + 2) % piece.count]].v
                precondition(simd_dot(simd_cross(q - p, r - q), n) >= -1e-9 * scale * scale,
                             "piece \(piece) is not convex at vertex \(piece[(k + 1) % piece.count])")
            }
            // (n is twice the area vector; p · n / 6 is the cone's volume.)
            volume += simd_dot(vertices[piece[0]].v, n * len) / 6
        }
        precondition(volume > 0, "the pieces are oriented inward (signed volume \(volume))")
        self.normals = normals
        let isDiagonal = self.edges.map { simd_dot(normals[$0.left], normals[$0.right]) >= 1 - 1e-9 }
        self.isDiagonal = isDiagonal
        // Faces: the pieces, joined across diagonals.
        var parent = Array(0..<pieces.count)
        func root(_ i: Int) -> Int {
            var i = i
            while parent[i] != i { parent[i] = parent[parent[i]]; i = parent[i] }
            return i
        }
        for (k, e) in self.edges.enumerated() where isDiagonal[k] {
            parent[root(e.left)] = root(e.right)
        }
        var byRoot: [Int: [Int]] = [:]
        var order: [Int] = []
        for f in 0..<pieces.count {
            let r = root(f)
            if byRoot[r] == nil { order.append(r) }
            byRoot[r, default: []].append(f)
        }
        self.faces = order.map { byRoot[$0]! }
    }

    /// The boundary's Euler characteristic, 2 − 2g for genus g. Counted over
    /// the partition, where every piece is a disc: V − E + F with the pieces
    /// for faces and all their sides, diagonals included, for edges. (Over
    /// the faces themselves a face with a hole needs the Euler–Poincaré
    /// term for it: 16 − 24 + 10 − 2 = 0 for the picture frame.)
    public var eulerCharacteristic: Int { vertices.count - edges.count + pieces.count }

    /// Whether every vertex lies on or behind every piece's plane.
    public var isConvex: Bool {
        let scale = vertices.map { simd_length($0.v) }.max() ?? 1
        for (f, piece) in pieces.enumerated() {
            let o = vertices[piece[0]].v
            for v in vertices where simd_dot(v.v - o, normals[f]) > 1e-9 * scale { return false }
        }
        return true
    }

    /// The pieces as triangles, fanned from each piece's first vertex: what
    /// the depth pass rasterizes. Exact for convex pieces.
    public var mesh: Mesh<WorldSpace> {
        var tris: [SIMD3<Int32>] = []
        for piece in pieces {
            for k in 1..<(piece.count - 1) {
                tris.append(SIMD3(Int32(piece[0]), Int32(piece[k]), Int32(piece[k + 1])))
            }
        }
        return Mesh(vertices: vertices, triangles: tris)
    }

    /// Every edge as a two-vertex path, before visibility: the curves a
    /// solid carries. Diagonals are not edges of the solid and are left out.
    public var edgePaths: PolylineSet<WorldSpace> {
        PolylineSet(paths: zip(edges, isDiagonal).filter { !$1 }.map { [vertices[$0.0.a], vertices[$0.0.b]] })
    }

    /// The solid moved rigidly: rotated about the origin, then translated.
    public func transformed(rotation: simd_double3x3, translation: SIMD3<Double> = .zero) -> Solid {
        Solid(vertices: vertices.map { P3(rotation * $0.v + translation) }, pieces: pieces)
    }

    /// The solid set down on a face: rotated so that face's outward normal
    /// points straight down, and lifted so the face lies in the plane
    /// `z = 0`, centred over the origin.
    public func resting(onFace i: Int) -> Solid {
        let n = normals[faces[i][0]]
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

    // MARK: - faces

    /// A face with a frame in its plane: an origin and two unit axes `u`,
    /// `v` with `u × v` the outward normal, so a polygon counter-clockwise
    /// from outside is counter-clockwise in `(u, v)`. Hatching on the face
    /// is laid out in these coordinates.
    public struct Face: Sendable {
        public let pieces: [Int]
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

    /// Face `i`, framed: the origin is the centroid of its vertices and
    /// `u` runs along the first edge of its first piece.
    public func face(_ i: Int) -> Face {
        let members = faces[i]
        let n = normals[members[0]]
        // Summed in index order, not a set's: the frame, and every hatch
        // line laid out in it, is then the same in every process.
        let ids = Set(members.flatMap { pieces[$0] }).sorted()
        let centroid = ids.reduce(SIMD3<Double>.zero) { $0 + vertices[$1].v } / Double(ids.count)
        let first = pieces[members[0]]
        let along = vertices[first[1]].v - vertices[first[0]].v
        let u = simd_normalize(along - simd_dot(along, n) * n)
        return Face(pieces: members, normal: n, origin: centroid, u: u, v: simd_cross(n, u))
    }

    /// The face piece `p` belongs to.
    public func faceIndex(ofPiece p: Int) -> Int {
        faces.firstIndex { $0.contains(p) }!
    }

    // MARK: - visibility

    /// The pieces projected to the picture plane, ready to test against:
    /// built once per camera, then asked about any flat segment.
    ///
    /// Under an orthographic camera a segment and a piece both project to
    /// the picture plane, their overlap along the segment is an interval,
    /// and over it the difference of their view depths is linear: where that
    /// difference says the piece is nearer, the segment is hidden. The hidden
    /// intervals are subtracted from the segment, exactly.
    public struct Occlusion: Sendable {
        let solid: Solid
        public let view: Transform<WorldSpace, ViewSpace>
        let projected: [P3<ViewSpace>]
        /// Twice each piece's signed area in projection; zero when seen
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
            area2 = solid.pieces.map { piece in
                var a = 0.0
                for k in 0..<piece.count {
                    let p = projected[piece[k]], q = projected[piece[(k + 1) % piece.count]]
                    a += p.x * q.y - q.x * p.y
                }
                return a
            }
        }

        /// Whether piece `f` is seen edge-on: it projects to a line, of no
        /// area, and hides nothing and shows nothing.
        public func isEdgeOn(_ f: Int) -> Bool { abs(area2[f]) <= 1e-9 * scale * scale }

        /// The sub-intervals of `[0, 1]` along the segment `pa -> pb` (in
        /// view space) over which a piece not `excluded` stands in front:
        /// sorted and disjoint.
        public func hidden(from pa: P3<ViewSpace>, to pb: P3<ViewSpace>,
                           excluding: (Int) -> Bool) -> [(Double, Double)] {
            var hidden: [(Double, Double)] = []
            for (f, piece) in solid.pieces.enumerated() where !excluding(f) {
                guard !isEdgeOn(f) else { continue }
                let sign: Double = area2[f] > 0 ? 1 : -1
                // The segment's overlap with the piece's projection: clipped
                // against each side's half-plane.
                var lo = 0.0, hi = 1.0
                for k in 0..<piece.count {
                    let p = projected[piece[k]], q = projected[piece[(k + 1) % piece.count]]
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
                // The piece's view depth over the picture plane is affine:
                // z = z0 + (gx, gy) · (x - x0, y - y0), from the plane's
                // normal in view space.
                let o = projected[piece[0]].v
                let n = simd_cross(projected[piece[1]].v - o, projected[piece[2]].v - o)
                guard abs(n.z) > 1e-9 * scale * scale else { continue }
                func gap(_ t: Double) -> Double {
                    let p = pa.v + t * (pb.v - pa.v)
                    let z = o.z - (n.x * (p.x - o.x) + n.y * (p.y - o.y)) / n.z
                    return z - p.z
                }
                // gap > eps means the piece is in front, and gap is linear.
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

        /// The complement: the spans of the segment that can be seen, as
        /// fractions of the way along it.
        public func visible(from pa: P3<ViewSpace>, to pb: P3<ViewSpace>,
                            excluding: (Int) -> Bool) -> [(Double, Double)] {
            var t = 0.0
            var spans: [(Double, Double)] = []
            for (a, b) in hidden(from: pa, to: pb, excluding: excluding) {
                if a > t { spans.append((t, a)) }
                t = max(t, b)
            }
            if t < 1 { spans.append((t, 1)) }
            return spans
        }
    }

    public func occlusion(view: Transform<WorldSpace, ViewSpace>) -> Occlusion { Occlusion(self, view: view) }

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
    public func visibleEdges(view: Transform<WorldSpace, ViewSpace>) -> PolylineSet<WorldSpace> {
        let occlusion = self.occlusion(view: view)
        let d = occlusion.toEye
        var paths: [[P3<WorldSpace>]] = []
        for (k, e) in edges.enumerated() where !isDiagonal[k] {
            guard simd_dot(normals[e.left], d) > 0 || simd_dot(normals[e.right], d) > 0 else { continue }
            let pa = occlusion.projected[e.a], pb = occlusion.projected[e.b]
            let dir = SIMD2(pb.x - pa.x, pb.y - pa.y)
            guard simd_length(dir) > occlusion.eps else { continue }   // seen end-on: a point
            let wa = vertices[e.a].v, wb = vertices[e.b].v
            for (a, b) in occlusion.visible(from: pa, to: pb, excluding: { $0 == e.left || $0 == e.right })
                where b - a > 1e-12 {
                paths.append([Solid.along(wa, wb, a), Solid.along(wa, wb, b)])
            }
        }
        return PolylineSet(paths: paths)
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
    public func visibleCurves(_ paths: [[P3<WorldSpace>]], onFace face: Int,
                              occlusion: Occlusion) -> [[P3<WorldSpace>]] {
        let members = Set(faces[face])
        guard simd_dot(normals[faces[face][0]], occlusion.toEye) > 0,
              !faces[face].allSatisfy(occlusion.isEdgeOn) else { return [] }
        var out: [[P3<WorldSpace>]] = []
        for path in paths {
            var run: [P3<WorldSpace>] = []
            var openAtEnd = false
            for (wa, wb) in zip(path, path.dropFirst()) {
                let pa = occlusion.view(wa), pb = occlusion.view(wb)
                // Whether the run so far reaches this segment's start. A
                // segment of which nothing is seen ends the run.
                let continuing = openAtEnd
                openAtEnd = false
                for (a, b) in occlusion.visible(from: pa, to: pb, excluding: { members.contains($0) })
                    where b - a > 1e-12 {
                    let start = Solid.along(wa.v, wb.v, a), end = Solid.along(wa.v, wb.v, b)
                    if continuing, a == 0, !run.isEmpty {
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
    /// round the face's centroid and turned away from the rest. Every face
    /// of a convex solid is convex, so each is a single piece.
    ///
    /// Cubic in the number of points and meant for the dozen or two a
    /// polyhedron table has, where it is the difference between faces that
    /// are computed and faces that are typed.
    public static func convexHull(of points: [P3<WorldSpace>]) -> Solid {
        precondition(points.count >= 4, "a solid needs four points")
        let scale = points.map { simd_length($0.v) }.max() ?? 1
        let tol = 1e-9 * max(scale, 1)
        var seen = Set<[Int]>()
        var pieces: [[Int]] = []
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
                    pieces.append(on.sorted {
                        let a = points[$0].v - centroid, b = points[$1].v - centroid
                        return atan2(simd_dot(a, v), simd_dot(a, u)) < atan2(simd_dot(b, v), simd_dot(b, u))
                    })
                }
            }
        }
        return Solid(vertices: points, pieces: pieces)
    }

    public enum Platonic: String, CaseIterable, Sendable {
        case tetrahedron, cube, octahedron, dodecahedron, icosahedron
    }

    /// One of the five, by its vertex coordinates alone, scaled to the given
    /// circumradius; the pieces are found by the hull.
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
    /// convex: the sign test is necessary and not sufficient, and the
    /// occlusion test earns its keep.
    public static var csaszar: Solid {
        let v: [SIMD3<Double>] = [
            SIMD3(-3, 3, 0), SIMD3(-3, -3, 1), SIMD3(-1, -2, 3), SIMD3(1, 2, 3),
            SIMD3(3, 3, 1), SIMD3(3, -3, 0), SIMD3(0, 0, 15),
        ]
        // Szilassi's list, one-based and inward; reversed here to face out.
        let listed = [[1, 2, 6], [1, 4, 2], [5, 3, 2], [4, 1, 3], [2, 7, 6], [3, 7, 2], [1, 7, 3],
                      [6, 5, 1], [6, 3, 5], [2, 4, 5], [3, 6, 4], [5, 7, 1], [4, 7, 5], [6, 7, 4]]
        return Solid(vertices: v.map { P3($0) },
                     pieces: listed.map { $0.reversed().map { $0 - 1 } })
    }

    /// A picture frame: a square `outer` across with a square hole `inner`
    /// across, `depth` thick, lying flat on `z = 0` and centred.
    ///
    /// The other torus here, and the plainer test of everything a convex
    /// solid never asks. Its front and back faces are annuli, which no
    /// convex polygon is, so each is partitioned into four trapezoids by
    /// four diagonals, which must not be drawn and must hide nothing. Its
    /// hole is a window: the far inner wall and the far edges of the back
    /// are seen through it, cut by the near part of the front, so an edge's
    /// visible spans are decided by occlusion and not by the signs. And
    /// every edge lies in the planes of the pieces beside it, where the gap
    /// between curve and piece is exactly zero.
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
        var pieces: [[Int]] = []
        for k in 0..<4 {
            pieces.append([ot(k), ot(k + 1), it(k + 1), it(k)])     // top, outward up
            pieces.append([ob(k + 1), ob(k), ib(k), ib(k + 1)])     // bottom, outward down
            pieces.append([ob(k), ob(k + 1), ot(k + 1), ot(k)])     // outer wall
            pieces.append([ib(k + 1), ib(k), it(k), it(k + 1)])     // inner wall, facing the hole
        }
        return Solid(vertices: v, pieces: pieces)
    }
}
