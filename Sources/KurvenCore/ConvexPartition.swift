import Foundation
import simd

/// The convex partition of a flat face with holes: what a `Solid` computes
/// from each face's loops, so that visibility and hatching only ever clip
/// against convex polygons.
///
/// Three steps, each the textbook one. The holes are joined to the outer
/// loop by bridges, which makes the face one polygon that runs along each
/// bridge twice. That polygon is cut into triangles by ear clipping. Then
/// Hertel and Mehlhorn (1983): every diagonal whose removal leaves both its
/// ends convex is removed, which merges its two pieces. The result has at
/// most four times as many pieces as the fewest possible. No vertex is
/// added: every piece's corners are the face's own vertices, so the
/// neighbouring faces' sides still meet the pieces' sides end to end.
///
/// A loop passes a vertex twice where the face touches itself there: a hole
/// touching the outer loop is part of the outer loop, and two holes touching
/// each other are one hole. Corners are taken by their place in a loop, not
/// by vertex, so such a vertex is two corners, each with its own two sides.
///
/// Every choice is made in index order, so a face gives the same pieces in
/// every process.
enum ConvexPartition {
    /// The convex pieces of the face bounded by `loops` -- the outer loop
    /// counter-clockwise, then each hole clockwise, as vertex indices --
    /// with `point` placing each vertex in the face's plane. Each piece is
    /// counter-clockwise. `tolerance` is the area below which a corner
    /// counts as straight, a small fraction of the face's own: every test
    /// here is a cross product held to it.
    static func pieces(_ loops: [[Int]], point: (Int) -> SIMD2<Double>, tolerance: Double) -> [[Int]] {
        let outer = loops[0]
        // A loop that passes a vertex twice is never one piece.
        if loops.count == 1, Set(outer).count == outer.count, isConvex(outer, point: point, tolerance: tolerance) {
            return [outer]
        }
        let ring = bridged(loops, point: point, tolerance: tolerance)
        return merged(triangles(ring, point: point, tolerance: tolerance), point: point, tolerance: tolerance)
    }

    static func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }

    /// Every corner turns left or runs straight.
    static func isConvex(_ polygon: [Int], point: (Int) -> SIMD2<Double>, tolerance: Double) -> Bool {
        let n = polygon.count
        return (0..<n).allSatisfy { k in
            let a = point(polygon[k]), b = point(polygon[(k + 1) % n]), c = point(polygon[(k + 2) % n])
            return cross(b - a, c - b) >= -tolerance
        }
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
    static func bridged(_ loops: [[Int]], point: (Int) -> SIMD2<Double>, tolerance: Double) -> [Int] {
        var ring = loops[0]
        var holes = Array(loops.dropFirst())
        holes.sort { a, b in
            let xa = a.map { point($0).x }.max()!, xb = b.map { point($0).x }.max()!
            return xa != xb ? xa > xb : a.min()! < b.min()!
        }
        for (h, hole) in holes.enumerated() {
            // Everything a bridge must not cross: the ring as it stands,
            // this hole, and the holes still to come.
            let walls = [ring, hole] + holes[(h + 1)...]
            var best: (length: Double, r: Int, j: Int)?
            for r in ring.indices {
                for j in hole.indices {
                    let p = point(ring[r]), q = point(hole[j])
                    let length = simd_length_squared(q - p)
                    if let b = best, length >= b.length { continue }
                    guard opensInto(ring, at: r, toward: q, point: point, tolerance: tolerance),
                          opensInto(hole, at: j, toward: p, point: point, tolerance: tolerance),
                          isClear(ring[r], hole[j], walls: walls, point: point, tolerance: tolerance)
                    else { continue }
                    best = (length, r, j)
                }
            }
            guard let b = best else { preconditionFailure("a hole cannot be joined to its face: it is not inside it") }
            let around = Array(hole[b.j...] + hole[..<b.j])
            ring = Array(ring[...b.r]) + around + [hole[b.j], ring[b.r]] + Array(ring[(b.r + 1)...])
        }
        return ring
    }

    /// Whether a segment leaving corner `k` of `loop` toward `q` starts into
    /// the face, which lies to the left of every loop, outer or hole.
    static func opensInto(_ loop: [Int], at k: Int, toward q: SIMD2<Double>,
                          point: (Int) -> SIMD2<Double>, tolerance: Double) -> Bool {
        let n = loop.count
        let a = point(loop[(k + n - 1) % n]), c = point(loop[k]), b = point(loop[(k + 1) % n])
        let d = q - c
        if cross(c - a, b - c) > tolerance {
            // A convex corner: strictly between the two sides.
            return cross(b - c, d) > tolerance && cross(d, a - c) > tolerance
        }
        // A reflex or straight corner: anywhere but on or between the two
        // sides on the outside.
        return !(cross(a - c, d) >= -tolerance && cross(d, b - c) >= -tolerance)
    }

    /// Whether the segment `i -> j` crosses no side of any wall and passes
    /// through no vertex other than its own two.
    static func isClear(_ i: Int, _ j: Int, walls: [[Int]], point: (Int) -> SIMD2<Double>,
                        tolerance: Double) -> Bool {
        let p = point(i), q = point(j)
        let d = q - p
        for wall in walls {
            let n = wall.count
            for k in 0..<n {
                let a = wall[k], b = wall[(k + 1) % n]
                let pa = point(a), pb = point(b)
                if a != i, a != j {
                    // A vertex lying on the segment.
                    let t = simd_dot(pa - p, d) / simd_length_squared(d)
                    if t > 0, t < 1, abs(cross(d, pa - p)) <= tolerance { return false }
                }
                guard a != i, a != j, b != i, b != j else { continue }
                // A proper crossing: each segment's ends on opposite sides of
                // the other.
                let s1 = cross(d, pa - p), s2 = cross(d, pb - p)
                let e = pb - pa
                let s3 = cross(e, p - pa), s4 = cross(e, q - pa)
                if ((s1 > tolerance && s2 < -tolerance) || (s1 < -tolerance && s2 > tolerance))
                    && ((s3 > tolerance && s4 < -tolerance) || (s3 < -tolerance && s4 > tolerance)) {
                    return false
                }
            }
        }
        return true
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
    static func triangles(_ ring: [Int], point: (Int) -> SIMD2<Double>, tolerance: Double) -> [[Int]] {
        var ring = ring
        var out: [[Int]] = []
        func isEar(_ k: Int) -> Bool {
            let n = ring.count
            let a = ring[(k + n - 1) % n], c = ring[k], b = ring[(k + 1) % n]
            let pa = point(a), pc = point(c), pb = point(b)
            guard cross(pc - pa, pb - pc) > tolerance else { return false }
            return !ring.contains { v in
                guard v != a, v != b, v != c else { return false }
                let p = point(v)
                return cross(pc - pa, p - pa) >= -tolerance && cross(pb - pc, p - pc) >= -tolerance
                    && cross(pa - pb, p - pb) >= -tolerance
            }
        }
        var start = 0
        while ring.count > 3 {
            let n = ring.count
            guard let k = (0..<n).lazy.map({ (start + $0) % n }).first(where: isEar) else {
                preconditionFailure("a face has no ear: its loops cross, or a hole is not inside it")
            }
            out.append([ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]])
            ring.remove(at: k)
            start = max(k - 1, 0)
        }
        let pa = point(ring[0]), pb = point(ring[1]), pc = point(ring[2])
        precondition(cross(pb - pa, pc - pb) > tolerance, "a face leaves a degenerate triangle: its loops overlap")
        out.append(ring)
        return out
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
    static func merged(_ triangles: [[Int]], point: (Int) -> SIMD2<Double>, tolerance: Double) -> [[Int]] {
        var pieces: [[Int]?] = triangles
        var owner: [SIMD2<Int>: Int] = [:]
        for (f, t) in triangles.enumerated() {
            for k in 0..<3 { owner[SIMD2(t[k], t[(k + 1) % 3])] = f }
        }
        var changed = true
        while changed {
            changed = false
            for f in pieces.indices {
                guard var piece = pieces[f] else { continue }
                var k = 0
                while k < piece.count {
                    let a = piece[k], b = piece[(k + 1) % piece.count]
                    guard let g = owner[SIMD2(b, a)], g != f, let other = pieces[g],
                          endsStayConvex(piece, k, other, point: point, tolerance: tolerance),
                          let joined = join(piece, other, a, b), isConvex(joined, point: point, tolerance: tolerance)
                    else { k += 1; continue }
                    piece = joined
                    pieces[f] = joined
                    pieces[g] = nil
                    owner[SIMD2(a, b)] = nil
                    owner[SIMD2(b, a)] = nil
                    for j in 0..<joined.count { owner[SIMD2(joined[j], joined[(j + 1) % joined.count])] = f }
                    changed = true
                    k = 0
                }
            }
        }
        return pieces.compactMap { $0 }
    }

    /// Whether `p` and `q` joined across the side `k` of `p`, `a -> b`,
    /// which `q` runs `b -> a`, would turn left or run straight at `a` and
    /// at `b`: from the vertex before `a` in `p` to the one after it in
    /// `q`, and from the vertex before `b` in `q` to the one after it in `p`.
    static func endsStayConvex(_ p: [Int], _ k: Int, _ q: [Int], point: (Int) -> SIMD2<Double>,
                               tolerance: Double) -> Bool {
        let n = p.count, m = q.count
        let a = p[k], b = p[(k + 1) % n]
        let j = q.firstIndex(of: b)!
        func turnsLeft(_ x: Int, _ y: Int, _ z: Int) -> Bool {
            let px = point(x), py = point(y), pz = point(z)
            return cross(py - px, pz - py) >= -tolerance
        }
        return turnsLeft(p[(k + n - 1) % n], a, q[(j + 2) % m]) && turnsLeft(q[(j + m - 1) % m], b, p[(k + 2) % n])
    }

    /// `p` and `q` joined across the side `a -> b` of `p`, which `q` runs
    /// `b -> a`; nil when that would visit a vertex twice.
    static func join(_ p: [Int], _ q: [Int], _ a: Int, _ b: Int) -> [Int]? {
        let i = p.firstIndex(of: b)!, j = q.firstIndex(of: a)!
        // p from b round to a, then q from just after a round to just before b.
        let fromB = Array(p[i...] + p[..<i])
        let fromA = Array(q[j...] + q[..<j])
        let joined = fromB + fromA.dropFirst().dropLast()
        return Set(joined).count == joined.count ? joined : nil
    }
}
