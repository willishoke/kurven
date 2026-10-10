import Foundation
import simd

/// A polyhedron bounding a solid: flat faces, sharp edges, and its ink is
/// its edges.
///
/// The third kind of geometry a plate can draw, beside a heightfield and a
/// parametric surface, and the one whose visibility is exact in the plainest
/// sense. A point of an edge is seen when some face at that edge faces the
/// eye and no other face stands between the point and the eye. Both halves
/// are decided in closed form: the first by two signs, the second by casting
/// the edge against every face -- the edge and the face are both flat, so
/// where one passes behind the other is an interval, solved, not searched.
/// No margin, no pixel, no lattice: the drawing is the same at any
/// resolution, which is the property every other visibility rule here is
/// held to and this one has by construction.
///
/// Faces are convex polygons, listed counter-clockwise seen from outside, so
/// a face's normal by the right-hand rule points out of the solid. That is
/// checked, not trusted: a solid whose faces do not close consistently is
/// refused, and the tables below are built from vertex coordinates alone by
/// a convex hull, so no face is ever typed in by hand.
public struct Solid: Sendable, Equatable {
    public let vertices: [P3<WorldSpace>]
    /// Each face as its vertex indices in order, counter-clockwise from
    /// outside.
    public let faces: [[Int]]
    /// Each edge once, with the two faces it bounds.
    public let edges: [Edge]
    /// Unit outward normals, one per face.
    public let normals: [SIMD3<Double>]

    public struct Edge: Sendable, Equatable, Hashable {
        public let a: Int, b: Int
        /// The face in which the edge runs `a -> b`, and the one in which it
        /// runs `b -> a`.
        public let left: Int, right: Int
    }

    /// - Precondition: every edge lies in exactly two faces, traversed in
    ///   opposite directions -- the surface is closed and consistently
    ///   oriented -- and the orientation is outward: the signed volume is
    ///   positive.
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
            normals.append(n / len)
            // (n is twice the area vector; p · n / 6 is the cone's volume.)
            volume += simd_dot(vertices[face[0]].v, n) / 6
        }
        precondition(volume > 0, "the faces are oriented inward (signed volume \(volume))")
        self.normals = normals
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

    /// Every edge as a two-vertex path, unjudged: the ink a solid carries.
    public var edgePaths: PolylineSet<WorldSpace> {
        PolylineSet(paths: edges.map { [vertices[$0.a], vertices[$0.b]] })
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

    // MARK: - visibility

    /// The edges as seen along `view`'s sight line, each cut to the pieces
    /// that can be seen.
    ///
    /// Per edge: with `d` the direction back toward the eye and `n₁`, `n₂`
    /// the outward normals of its two faces, nothing of the edge shows unless
    /// `n₁ · d > 0 ∨ n₂ · d > 0` -- both faces turned away means the edge is
    /// on the back of the solid, behind the solid itself. On a convex solid
    /// that is the whole test, and an edge is whole or absent. On any other
    /// solid an edge that passes the sign test can still run behind a face
    /// elsewhere on the body -- the far rim of a torus's tunnel seen through
    /// the near one -- so every other face is cast against it. Under an
    /// orthographic camera the edge and a face both project to the picture
    /// plane, their overlap along the edge is an interval, and over it the
    /// difference of their view depths is linear: where that difference says
    /// the face is nearer, the edge is hidden. The hidden intervals are
    /// subtracted from the edge, exactly.
    public func visibleEdges(view: Transform<WorldSpace, ViewSpace>) -> PolylineSet<WorldSpace> {
        let d = -view.sightLine
        let projected = vertices.map { view($0) }
        let scale = projected.map { simd_length($0.v) }.max() ?? 1
        let eps = 1e-9 * max(scale, 1)
        // Faces that cannot hide anything: edge-on in projection.
        let area2: [Double] = faces.map { face in
            var a = 0.0
            for k in 0..<face.count {
                let p = projected[face[k]], q = projected[face[(k + 1) % face.count]]
                a += p.x * q.y - q.x * p.y
            }
            return a
        }
        var paths: [[P3<WorldSpace>]] = []
        for e in edges {
            guard simd_dot(normals[e.left], d) > 0 || simd_dot(normals[e.right], d) > 0 else { continue }
            let pa = projected[e.a], pb = projected[e.b]
            let dir = SIMD2(pb.x - pa.x, pb.y - pa.y)
            guard simd_length(dir) > eps else { continue }   // seen end-on: a point
            var hidden: [(Double, Double)] = []
            for (f, face) in faces.enumerated() where f != e.left && f != e.right {
                // A face seen edge-on projects to nothing and hides nothing.
                guard abs(area2[f]) > 1e-9 * scale * scale else { continue }
                let sign: Double = area2[f] > 0 ? 1 : -1
                // The edge's overlap with the face's projection: the segment
                // clipped against each side's half-plane.
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
            // Subtract the union of hidden intervals from [0, 1].
            hidden.sort { $0.0 < $1.0 }
            var t = 0.0
            var pieces: [(Double, Double)] = []
            for (a, b) in hidden {
                if a > t { pieces.append((t, a)) }
                t = max(t, b)
            }
            if t < 1 { pieces.append((t, 1)) }
            let wa = vertices[e.a].v, wb = vertices[e.b].v
            for (a, b) in pieces where b - a > 1e-12 {
                paths.append([P3(wa + a * (wb - wa)), P3(wa + b * (wb - wa))])
            }
        }
        return PolylineSet(paths: paths)
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
}
