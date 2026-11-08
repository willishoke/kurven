import Foundation
import simd

// MARK: - picking a face

/// The face a point of space lies on, as `Face.pick(_:among:within:tie:)`
/// finds it: which face, which copy of it, the point of its parameter plane
/// the point stands at (`Face.foot`), and how far the point is from it.
public struct FacePick: Sendable, Equatable, CustomStringConvertible {
    public let face: FaceID
    /// The copy's place in `Face.copies`: on a tiled landscape, the tile.
    public let copy: Int
    public let parameter: P2<ParamSpace>
    public let distance: Double

    public var description: String { copy == 0 ? "\(face)" : "\(face) on tile \(copy)" }
}

extension FaceID: CustomStringConvertible {
    public var description: String {
        switch self {
        case .graph: "graph"
        case .plateau(let band): "plateau \(band)"
        case .riser(let band): "riser \(band)"
        case .tileBoundaryRiser(let tile, let facing): "riser at the tile boundary, tile \(tile) over \(facing)"
        case .wall(let edge): "wall \(edge)"
        case .surface: "surface"
        case .solid(let face): "face \(face)"
        }
    }
}

public extension Face {
    /// The face a point of space lies on, of `faces`, and the copy: what a
    /// click picks.
    ///
    /// Each copy of each face is asked where the point stands on it (`foot`,
    /// through the copy's tile carried back), and how far the point is from
    /// the face there, along the face's normal: exactly for a flat face and
    /// a parametric surface, whose feet are the nearest points, and to first
    /// order on the graph, whose foot is the point of the domain under it. A
    /// copy is a candidate where the distance is at most `tolerance` and its
    /// trimmed domain holds the foot, or a point `tie` from it
    /// (`Copy.holds(_:within:)`), so a point on a crease is in both faces
    /// that meet there.
    ///
    /// The nearest candidate is the face, and the tie rule settles a point
    /// as near two: candidates within `tie` of the nearest are equally near,
    /// and the first of them in the order the faces come (`FaceID`) wins,
    /// then the copy first in the tiles. So a crease belongs to the face
    /// listed first of the two it bounds: the truncation edge, a wall's top
    /// edge and a riser's top to the graph or the plateau above, a riser's
    /// bottom to the face below, a point where two copies meet to the
    /// copy on the first tile, a polyhedron's edge to the face of lesser
    /// index. Nil where no face is within `tolerance`.
    static func pick(_ p: P3<WorldSpace>, among faces: [Face], within tolerance: Double,
                     tie: Double) -> FacePick? {
        var picks: [FacePick] = []
        for face in faces {
            for (k, copy) in face.copies.enumerated() {
                guard let back = copy.tile.inverse else { continue }
                let q = face.foot(back(p).v)
                let jet = copy.place(face.map(q))
                let n = jet.normal, length = simd_length(n)
                let offset = p.v - jet.position
                let distance = length > 0 ? abs(simd_dot(offset, n)) / length : simd_length(offset)
                guard distance <= tolerance, copy.holds(q, within: tie) else { continue }
                picks.append(FacePick(face: face.id, copy: k, parameter: q, distance: distance))
            }
        }
        guard let nearest = picks.map(\.distance).min() else { return nil }
        return picks.filter { $0.distance <= nearest + tie }
            .min { ($0.face, $0.copy) < ($1.face, $1.copy) }
    }

    /// The boundary of one copy of the face, as curves of its parameter
    /// plane: where its trimmed domain ends. `map`, then the copy's `place`,
    /// carry them into space.
    ///
    /// A polyhedron face's are its loops. The graph's and a plateau's are
    /// read even-odd off the graph's trim loops, the drawn truncation edge
    /// among them, cut to the footprint where the copy stands
    /// (`Surface.topBoundary(_:loops:region:tile:)`). A wall's runs along its
    /// bottom, up its ends and along its top edge as its layer draws it, and
    /// a riser's round each stretch where it stands, from its bottom up to
    /// its top (`Surface.columnLoops`). A parametric surface's are its
    /// parameter rectangle's edges, and on a periodic axis the seam, where
    /// the rectangle is glued: it bounds nothing, but it is where the face's
    /// chart ends (`ParametricSurface.seams`).
    func boundary(of copy: Copy) -> [[P2<ParamSpace>]] {
        bounding(copy.tile).filter { $0.count >= 2 }
    }

    /// The segments of `curves`, each cut where another crosses it or ends
    /// on it, or passes a vertex within `eps` of it: the vertex itself is
    /// the cut, so the pieces of two curves that run along each other share
    /// their ends.
    static func pieces(of curves: [[SIMD2<Double>]], eps: Double) -> [(SIMD2<Double>, SIMD2<Double>)] {
        var segments: [(SIMD2<Double>, SIMD2<Double>)] = []
        for curve in curves {
            for (a, b) in zip(curve, curve.dropFirst()) where a != b && a.x.isFinite && a.y.isFinite
                && b.x.isFinite && b.y.isFinite {
                segments.append((a, b))
            }
        }
        guard !segments.isEmpty else { return [] }
        // A grid of buckets over the segments' box, about one segment a
        // bucket, each segment filed under every bucket its box meets.
        var lo = SIMD2<Double>(repeating: .infinity), hi = SIMD2<Double>(repeating: -.infinity)
        for (a, b) in segments { lo = simd_min(lo, simd_min(a, b)); hi = simd_max(hi, simd_max(a, b)) }
        let side = max(Int(Double(segments.count).squareRoot()), 1)
        let size = simd_max((hi - lo) / Double(side), SIMD2(repeating: max(eps, 1e-300)))
        func cell(_ p: SIMD2<Double>) -> SIMD2<Int> {
            let c = (p - lo) / size
            return SIMD2(min(max(Int(c.x.rounded(.down)), 0), side - 1), min(max(Int(c.y.rounded(.down)), 0), side - 1))
        }
        var buckets = [[Int]](repeating: [], count: side * side)
        for (i, (a, b)) in segments.enumerated() {
            let c0 = cell(simd_min(a, b) - eps), c1 = cell(simd_max(a, b) + eps)
            for y in c0.y...c1.y { for x in c0.x...c1.x { buckets[y * side + x].append(i) } }
        }
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }
        var stamp = [Int](repeating: -1, count: segments.count)
        var out: [(SIMD2<Double>, SIMD2<Double>)] = []
        for (i, (a, b)) in segments.enumerated() {
            let d = b - a, length2 = simd_length_squared(d), length = length2.squareRoot()
            var cuts: [(t: Double, p: SIMD2<Double>)] = []
            /// A point within `eps` of the segment, strictly between its ends.
            func onSegment(_ p: SIMD2<Double>) {
                let t = simd_dot(p - a, d) / length2
                guard t * length > eps, (1 - t) * length > eps, abs(cross(d, p - a)) / length <= eps else { return }
                cuts.append((t, p))
            }
            let c0 = cell(simd_min(a, b) - eps), c1 = cell(simd_max(a, b) + eps)
            for y in c0.y...c1.y {
                for x in c0.x...c1.x {
                    for j in buckets[y * side + x] where j != i && stamp[j] != i {
                        stamp[j] = i
                        let (c, e) = segments[j]
                        onSegment(c)
                        onSegment(e)
                        let f = e - c, det = cross(d, f)
                        guard det != 0 else { continue }
                        let r = c - a
                        let t = cross(r, f) / det, u = cross(r, d) / det
                        guard t > 0, t < 1, u > 0, u < 1 else { continue }
                        cuts.append((t, a + t * d))
                    }
                }
            }
            cuts.sort { $0.t < $1.t }
            var start = a, last = 0.0
            for cut in cuts where (cut.t - last) * length > eps && (1 - cut.t) * length > eps {
                out.append((start, cut.p))
                start = cut.p; last = cut.t
            }
            out.append((start, b))
        }
        return out
    }

    /// Segments with their ends taken onto one point wherever two lie
    /// within `eps`: two curves cut at one crossing land a rounding apart.
    /// A segment left with no length, and the second of two with the same
    /// ends either way round, are dropped.
    static func snapped(_ segments: [(SIMD2<Double>, SIMD2<Double>)], eps: Double)
        -> [(SIMD2<Double>, SIMD2<Double>)] {
        var cells: [SIMD2<Int64>: [SIMD2<Double>]] = [:]
        func snap(_ p: SIMD2<Double>) -> SIMD2<Double> {
            let c = SIMD2<Int64>(Int64((p.x / eps).rounded(.down)), Int64((p.y / eps).rounded(.down)))
            for dx in -1...1 {
                for dy in -1...1 {
                    if let q = cells[c &+ SIMD2(Int64(dx), Int64(dy))]?.first(where: { simd_length($0 - p) <= eps }) {
                        return q
                    }
                }
            }
            cells[c, default: []].append(p)
            return p
        }
        var seen = Set<[Double]>()
        return segments.compactMap { a, b in
            let (p, q) = (snap(a), snap(b))
            guard p != q else { return nil }
            let key = p.x < q.x || (p.x == q.x && p.y < q.y) ? [p.x, p.y, q.x, q.y] : [q.x, q.y, p.x, p.y]
            return seen.insert(key).inserted ? (p, q) : nil
        }
    }

    /// Segments joined end to end into polylines, at ends they share
    /// exactly and no other segment does.
    static func chained(_ segments: [(SIMD2<Double>, SIMD2<Double>)]) -> [[SIMD2<Double>]] {
        var at: [SIMD2<Double>: [Int]] = [:]
        for (i, (a, b)) in segments.enumerated() { at[a, default: []].append(i); at[b, default: []].append(i) }
        var used = [Bool](repeating: false, count: segments.count)
        var out: [[SIMD2<Double>]] = []
        /// The next unused segment from `p`, where `p` joins exactly two.
        func next(from p: SIMD2<Double>) -> Int? {
            guard let ends = at[p], ends.count == 2 else { return nil }
            return ends.first { !used[$0] }
        }
        func extend(_ path: inout [SIMD2<Double>]) {
            while let j = next(from: path[path.count - 1]) {
                used[j] = true
                let (a, b) = segments[j]
                path.append(a == path[path.count - 1] ? b : a)
            }
        }
        for (i, (a, b)) in segments.enumerated() where !used[i] {
            used[i] = true
            var forward = [a, b]
            extend(&forward)
            var backward = [a]
            if forward.last != a { extend(&backward) }
            out.append(backward.dropFirst().reversed() + forward)
        }
        return out
    }
}

public extension Face.Copy {
    /// Whether the copy's trimmed domain holds `p` or a point `tie` from it
    /// in one of sixteen directions round it: a point on its boundary, or at
    /// a corner of it no sharper than 22.5°, is held.
    func holds(_ p: P2<ParamSpace>, within tie: Double) -> Bool {
        if contains(p) { return true }
        guard tie > 0 else { return false }
        return (0..<16).contains { k in
            let a = Double(k) * .pi / 8
            return contains(P2(p.v + tie * SIMD2(cos(a), sin(a))))
        }
    }
}

public extension SceneGeometry {
    /// The face of the solid the scene draws that a point lies on, and the
    /// copy (`Face.pick(_:among:within:tie:)`): on a landscape, a plateau
    /// where the point is at the truncation's height and |f| reaches it, a
    /// riser or a wall where it stands on one, the graph otherwise, on the
    /// tile the inverse tile map finds; on a parametric surface its one face;
    /// on a polyhedron the face whose plane holds the point and whose loops
    /// contain it. `faces` are the scene's (`faces`), passed where they are
    /// kept, so the trim loops a boundary needs are made once. Ties are
    /// settled within a billionth of `scale`.
    func face(at p: P3<WorldSpace>, within tolerance: Double, faces: [Face]? = nil) -> FacePick? {
        Face.pick(p, among: faces ?? self.faces, within: tolerance, tie: 1e-9 * scale)
    }

    /// The boundary of the picked face's copy, in space
    /// (`Face.boundary(of:)`), with each vertex's point of the face's
    /// parameter plane. Each of its segments is cut at the face's parameter
    /// spacing, then halved until its image lies within a hundredth of
    /// `scale` of the chord, twelve deep, so the curve follows a curved face
    /// between the boundary's own vertices: the graph along the domain's
    /// edge, where its boundary has a vertex only at the corners.
    func boundary(of pick: FacePick, faces: [Face]? = nil)
        -> (paths: [[P3<WorldSpace>]], coords: [[P2<ParamSpace>]]) {
        guard let face = (faces ?? self.faces).first(where: { $0.id == pick.face }),
              pick.copy < face.copies.count else { return ([], []) }
        let copy = face.copies[pick.copy]
        let tolerance = 0.01 * scale
        let spacing = min(face.domain.u.spacing, face.domain.v.spacing)
        func point(_ c: SIMD2<Double>) -> SIMD3<Double> { copy.place(face.map(P2(c))).position }
        var paths: [[P3<WorldSpace>]] = [], coords: [[P2<ParamSpace>]] = []
        for curve in face.boundary(of: copy) {
            var cs: [SIMD2<Double>] = [curve[0].v], ps: [SIMD3<Double>] = [point(curve[0].v)]
            func subdivide(_ a: SIMD2<Double>, _ pa: SIMD3<Double>, _ b: SIMD2<Double>, _ pb: SIMD3<Double>, depth: Int) {
                let m = 0.5 * (a + b), pm = point(m)
                if depth < 12, simd_length(pm - 0.5 * (pa + pb)) > tolerance {
                    subdivide(a, pa, m, pm, depth: depth + 1)
                    subdivide(m, pm, b, pb, depth: depth + 1)
                } else {
                    cs.append(b); ps.append(pb)
                }
            }
            for (a, b) in zip(curve, curve.dropFirst()) {
                let n = spacing > 0 ? min(max(Int((simd_length(b.v - a.v) / spacing).rounded(.up)), 1), 4096) : 1
                for k in 1...n {
                    let c0 = cs[cs.count - 1], p0 = ps[ps.count - 1]
                    let c1 = k == n ? b.v : a.v + (b.v - a.v) * (Double(k) / Double(n))
                    subdivide(c0, p0, c1, point(c1), depth: 0)
                }
            }
            paths.append(ps.map(P3.init)); coords.append(cs.map(P2.init))
        }
        return (paths, coords)
    }

    /// The length the scene's tolerances are fractions of: a landscape's
    /// cell, a parametric surface's grid spacing, the spacing of its
    /// neighbouring samples at the widest, a polyhedron's largest face.
    var scale: Double {
        switch self {
        case .heightfield(let h): h.surface.cell
        case .parametric(let s): s.gridSpacing
        case .solid(let s): s.frames.map(\.extent).max() ?? 1
        }
    }

    /// How far a point the preview's depth buffer gives can stand from the
    /// face under it, with `unitsPerPixel` the size of a pixel: a few pixels,
    /// and on a landscape two of the cells the depth pass draws, or on a
    /// parametric surface its grid's widest spacing, which the depth pass
    /// draws its triangles between. A polyhedron's depth pass draws its faces
    /// themselves.
    func pickTolerance(unitsPerPixel: Double) -> Double {
        let pixels = 4 * unitsPerPixel
        switch self {
        case .heightfield(let h): return pixels + 2 * Double(max(h.step, 1)) * h.surface.cell
        case .parametric: return pixels + scale
        case .solid: return pixels
        }
    }
}

public extension ParametricSurface {
    /// The point of the parameter plane whose image is nearest `p`: from the
    /// tubular neighbourhood's foot where the surface has one, else the
    /// nearest grid sample, by Gauss-Newton steps on the distance, each cut
    /// back until it brings the image nearer, a periodic coordinate taken
    /// into its range and a bounded one held to it.
    func nearestParameter(to p: SIMD3<Double>) -> P2<ParamSpace> {
        func wrapped(_ c: SIMD2<Double>) -> SIMD2<Double> {
            func into(_ x: Double, _ axis: ParamAxis) -> Double {
                let lo = min(axis.range.lo, axis.range.hi), hi = max(axis.range.lo, axis.range.hi)
                guard axis.periodic else { return min(max(x, lo), hi) }
                let period = hi - lo
                return x - period * ((x - lo) / period).rounded(.down)
            }
            return SIMD2(into(c.x, u), into(c.y, v))
        }
        var c: SIMD2<Double>
        if let neighbourhood {
            c = neighbourhood.foot(p).coordinate.v
        } else {
            var best = Double.infinity, at = 0
            for (k, q) in positions.enumerated() {
                let d = simd_length_squared(q.v - p)
                if d < best { best = d; at = k }
            }
            c = SIMD2(u.coordinate(at % u.samples), v.coordinate(at / u.samples))
        }
        var r = map(P2(c)).position - p
        for _ in 0..<40 {
            let jet = map(P2(c))
            let a = simd_dot(jet.du, jet.du), b = simd_dot(jet.du, jet.dv), e = simd_dot(jet.dv, jet.dv)
            let det = a * e - b * b
            guard det > 0 else { break }
            let gu = simd_dot(jet.du, r), gv = simd_dot(jet.dv, r)
            var step = -SIMD2((e * gu - b * gv) / det, (a * gv - b * gu) / det)
            let reach = SIMD2(4 * u.spacing, 4 * v.spacing)
            step = simd_clamp(step, -reach, reach)
            var moved = false
            for _ in 0..<20 {
                let next = wrapped(c + step)
                let rn = map(P2(next)).position - p
                if simd_length_squared(rn) < simd_length_squared(r) {
                    c = next; r = rn; moved = true
                    break
                }
                step *= 0.5
            }
            guard moved, simd_reduce_max(abs(step) / SIMD2(u.spacing, v.spacing)) > 1e-14 else { break }
        }
        return P2(c)
    }

    /// The edges of the parameter rectangle, each at the other axis's
    /// samples, at least 64 of them: on a periodic axis the one seam where the
    /// rectangle is glued, on a bounded axis both of its ends.
    var seams: [[P2<ParamSpace>]] {
        func run(_ axis: ParamAxis) -> [Double] {
            let n = max(axis.samples, 64)
            return (0...n).map { axis.range.lo + axis.range.length * Double($0) / Double(n) }
        }
        let us = u.periodic ? [u.range.lo] : [u.range.lo, u.range.hi]
        let vs = v.periodic ? [v.range.lo] : [v.range.lo, v.range.hi]
        return us.map { x in run(v).map { P2(x, $0) } } + vs.map { y in run(u).map { P2($0, y) } }
    }

    /// The widest spacing between neighbouring samples of the grid.
    var gridSpacing: Double {
        var widest = 0.0
        for j in 0..<v.samples {
            for i in 0..<u.samples {
                let p = position(i, j).v
                if i + 1 < u.samples || u.periodic { widest = max(widest, simd_length(position(i + 1, j).v - p)) }
                if j + 1 < v.samples || v.periodic { widest = max(widest, simd_length(position(i, j + 1).v - p)) }
            }
        }
        return widest
    }
}
