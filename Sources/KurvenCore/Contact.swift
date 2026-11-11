import Foundation
import simd

/// Where a part of a composite meets a landscape's terrain: the curve on the
/// part's surface where its point w has w.z = H(w.x, w.y), with H the
/// terrain's height as the composite's cut has it (`Slab`), so the curve
/// lies where the part's strokes, cut by the landscape, end.
///
/// H is affine over each piece of the terrain's top (`TopPiece`): the slab's
/// prisms, each split where the triangle's plane meets the truncation. The
/// curve is solved piece by piece, so every point of it lies on both
/// surfaces and every chord in one piece, on the terrain; where H steps --
/// a riser, the terrain's edge, the footprint's -- the curve ends on the
/// step.
///
/// - On a polyhedron each face is solved exactly: over each prism, the face's
///   points on the terrain are the sides of the convex polygon the face
///   leaves inside the prism that lie on its top. The pieces are joined end
///   to end and clipped to the face's trim loops (`Contour.clipped`).
/// - On the round torus, over each piece, the curve is where the torus
///   meets the piece's plane. On a level piece it is closed form: the
///   torus stands upright, so the curve is circles about its axis, clipped
///   to the piece. On any other it is the zero set of the torus's distance
///   on the plane, which has Lipschitz constant 1 in space
///   (`TubularNeighbourhood.foot`): blocks of cells, then the piece, then
///   squares are passed over where the bound shows no zero, and split
///   otherwise, down to squares a quarter of a cell, at most an eighth of
///   the tube's radius (`zeros(of:on:side:least:within:loopsRound:)`), so
///   the work goes with the curve's length. Each zero is bisected to the
///   last bit. A loop smaller than a sixty-fourth of a square can be
///   missed.
///
/// Contact within the scene's tolerance (`Slab.start`) of a polyhedron
/// face's boundary, along it, is the part's edge and is not contact; a face
/// within the tolerance of the terrain's plane all over meets it nowhere.
/// So with the torus: a level piece it reaches past by no more than the
/// tolerance, or sinks into by no more, it meets nowhere.
///
/// A prism with no top -- a corner of its triangle not a number, and no
/// truncation over it -- is no terrain, and nothing meets it.
///
/// The curves are in the part's own coordinates, on a polyhedron with the
/// face each lies on, on a surface with each vertex's coordinate.
public struct ContactCurves: Sendable {
    public var paths: [[P3<WorldSpace>]] = []
    /// The face each path lies on, on a polyhedron; empty on a surface.
    public var faces: [Int] = []
    /// Each path's surface coordinates, on a surface; empty on a polyhedron.
    public var coords: [[P2<ParamSpace>]] = []

    public var isEmpty: Bool { paths.isEmpty }
}

extension Composite {
    /// Where part `i` meets each landscape of the composite (`ContactCurves`).
    /// Empty for a landscape, and for a part that meets none.
    public func contact(of i: Int) -> ContactCurves {
        var out = ContactCurves()
        let part = parts[i], placement = part.placement
        for (j, occluder) in occluders.enumerated() where j != i {
            guard case .slab(let slab, let offset) = occluder else { continue }
            switch part.scene.geometry {
            case .heightfield:
                return ContactCurves()
            case .solid(let s):
                for f in s.faces.indices {
                    for path in slab.contact(onFace: f, of: s, placement: placement, offset: offset) {
                        out.paths.append(path); out.faces.append(f)
                    }
                }
            case .parametric(let s):
                guard let e = s.neighbourhood else { continue }
                for path in slab.contact(on: s, placement: placement, offset: offset) {
                    // Into the part's own coordinates, with the coordinate
                    // of each point, unwrapped along the path.
                    let points = path.map { ($0 + offset - placement.offset) / placement.scale }
                    var coords = points.map { e.foot($0).coordinate.v }
                    for k in coords.indices.dropFirst() {
                        for (axis, period) in [(0, s.u.period), (1, s.v.period)] {
                            guard let period else { continue }
                            coords[k][axis] -= period * ((coords[k][axis] - coords[k - 1][axis]) / period).rounded()
                        }
                    }
                    out.paths.append(points.map { P3($0) }); out.coords.append(coords.map { P2($0) })
                }
            }
        }
        return out
    }

    /// Part `i`'s contact curves as a layer, drawn in the part's outline
    /// style, and in view space as the part's own geometry shows them: on a
    /// polyhedron each path tested on its face
    /// (`Solid.visibleCurves(_:onFace:occlusion:)`), on a surface by its
    /// coordinates (`clipOnChords`). Nil where the part meets no landscape.
    /// The layer's curves are carried as given, as a file layer's are.
    public func visibleContact(of i: Int) -> (Layer, PolylineSet<ViewSpace>)? {
        let contact = self.contact(of: i)
        guard !contact.isEmpty else { return nil }
        let scene = parts[i].scene, view = scene.camera.view
        // The part's outline style: its first outline layer's.
        let style = scene.layers.first { $0.spec.role == .outline }?.spec
        let spec = LayerSpec(name: "contact", role: .outline, source: .file(vertices: "", offsets: ""),
                             width: style?.width ?? 0.5, heightPolicy: .surface,
                             color: style?.color ?? "#000000")
        switch scene.geometry {
        case .solid(let s):
            let occlusion = s.occlusion(view: view)
            var seen: [[P3<WorldSpace>]] = []
            var done = Set<Int>()
            for f in contact.faces where done.insert(f).inserted {
                let on = contact.paths.indices.filter { contact.faces[$0] == f }.map { contact.paths[$0] }
                seen += s.visibleCurves(on, onFace: f, occlusion: occlusion)
            }
            return (Layer(spec: spec, paths: PolylineSet(paths: contact.paths)), PolylineSet(paths: seen).mapped(view))
        case .parametric(let s):
            let paths = PolylineSet(paths: contact.paths, coords: contact.coords)
            return (Layer(spec: spec, paths: paths),
                    Composite.clipOnChords(paths.mapped(view), SurfaceVisibility(sphereTracing: s, view: view)))
        case .heightfield:
            return nil
        }
    }

    /// Curves on a surface whose chords lie on a landscape's terrain, cut
    /// where the surface hides them, as `HiddenLine.clip(_:on:onFolds:onCrease:)`
    /// cuts them -- each vertex tested where it lies on the surface, and a
    /// segment along which the verdict changes bisected on the test at the
    /// coordinates along it -- but each cut placed on the chord, at the
    /// fraction of the way the bisection found, so that what is drawn stays
    /// on the terrain: the point of the surface there is off it by the
    /// chord's sagitta.
    static func clipOnChords(_ paths: PolylineSet<ViewSpace>, _ visibility: SurfaceVisibility) -> PolylineSet<ViewSpace> {
        guard let coords = paths.coords else { return PolylineSet(paths: []) }
        let vertices = paths.vertices
        let kept = Heightfield.parallelMap(Array(vertices.indices), chunk: 32) {
            visibility.isVisible(vertices[$0], at: coords[$0])
        }
        var out: [[P3<ViewSpace>]] = []
        for path in 0..<paths.count {
            var run: [P3<ViewSpace>] = []
            let lo = paths.offsets[path], hi = paths.offsets[path + 1]
            for k in lo..<hi {
                if k > lo, kept[k - 1] != kept[k] {
                    let a = coords[k - 1].v, b = coords[k].v
                    var t0 = 0.0, t1 = 1.0
                    while t1 - t0 > 1e-13 {
                        let mid = 0.5 * (t0 + t1)
                        if visibility.test(P2(a + (b - a) * mid), onFolds: false).seen == kept[k - 1] { t0 = mid } else { t1 = mid }
                    }
                    let t = kept[k - 1] ? t0 : t1
                    let cut = P3<ViewSpace>(vertices[k - 1].v + (vertices[k].v - vertices[k - 1].v) * t)
                    if kept[k - 1] {
                        run.append(cut)
                        if run.count >= 2 { out.append(run) }
                        run = []
                    } else {
                        run = [cut]
                    }
                }
                if kept[k] { run.append(vertices[k]) }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }
}

/// A piece of a landscape's terrain over which its height is affine: a
/// convex polygon of the landscape's own (x, y), counter-clockwise, and the
/// height over it, g · p + h. One of a prism's tops (`Slab.tops`).
struct TopPiece: Sendable {
    let polygon: [SIMD2<Double>]
    let gradient: SIMD2<Double>, constant: Double

    func height(_ p: SIMD2<Double>) -> Double { simd_dot(gradient, p) + constant }
    /// The point of the terrain over `p`.
    func point(_ p: SIMD2<Double>) -> SIMD3<Double> { SIMD3(p.x, p.y, height(p)) }
    /// Whether `p` lies in the polygon, or within `slack` of it.
    func contains(_ p: SIMD2<Double>, slack: Double = 0) -> Bool {
        polygon.indices.allSatisfy { k in
            let a = polygon[k], b = polygon[(k + 1) % polygon.count], e = b - a
            return e.x * (p.y - a.y) - e.y * (p.x - a.x) >= -slack * simd_length(e)
        }
    }
}

extension Slab {
    /// The tops of the prisms over one triangle of cell (i, j) in one copy
    /// (`prisms`): each prism's sides bound a convex polygon, and over it
    /// the top is the least of its upward half-spaces, the triangle's plane
    /// and the truncation; the polygon is split where they meet. A prism
    /// with no top -- a corner of its triangle not a number, and no
    /// truncation over it -- has none.
    func tops(_ i: Int, _ j: Int, upper: Bool, copy: Copy) -> [TopPiece] {
        let (fi, fj) = (Double(i), Double(j))
        let square = [SIMD2(fi, fj), SIMD2(fi + 1, fj), SIMD2(fi + 1, fj + 1), SIMD2(fi, fj + 1)]
        let back = copy.linear.inverse
        // A point of index coordinates in the landscape's (x, y).
        func xy(_ q: SIMD2<Double>) -> SIMD2<Double> { back * (q - copy.offset) }
        // A reflected copy turns the polygon over.
        let turned = copy.linear.determinant < 0
        var out: [TopPiece] = []
        for halfSpaces in prisms(i, j, upper: upper, copy: copy) {
            var polygon = square
            for (n, c) in halfSpaces where n.z == 0 {
                polygon = Slab.clip2D(polygon) { n.x * $0.x + n.y * $0.y - c }
                if polygon.count < 3 { break }
            }
            guard polygon.count >= 3 else { continue }
            // Each top as an affine height over (u, v): a · q + b.
            let ups = halfSpaces.filter { $0.n.z > 0 }.map { (a: -SIMD2($0.n.x, $0.n.y) / $0.n.z, b: $0.c / $0.n.z) }
            for (k, top) in ups.enumerated() {
                var region = polygon
                for (l, other) in ups.enumerated() where l != k {
                    region = Slab.clip2D(region) { simd_dot(top.a - other.a, $0) + top.b - other.b }
                    if region.count < 3 { break }
                }
                // A polygon with no area -- a corner or a side two pieces
                // share -- is none.
                var area = 0.0
                for k in region.indices {
                    let p = region[k], q = region[(k + 1) % region.count]
                    area += p.x * q.y - q.x * p.y
                }
                guard region.count >= 3, area > 1e-18 else { continue }
                // Into (x, y): q = L p + o.
                let mapped = region.map(xy)
                out.append(TopPiece(polygon: turned ? mapped.reversed() : mapped,
                                    gradient: copy.linear.transpose * top.a,
                                    constant: simd_dot(top.a, copy.offset) + top.b))
            }
        }
        return out
    }

    /// The cells of a copy a box of the landscape's (x, y) can reach, a
    /// cell beyond on every side; nil where it reaches none.
    func cells(over lo: SIMD2<Double>, _ hi: SIMD2<Double>, copy: Copy) -> (ClosedRange<Int>, ClosedRange<Int>)? {
        let corners = [lo, SIMD2(hi.x, lo.y), hi, SIMD2(lo.x, hi.y)].map { copy.linear * $0 + copy.offset }
        let ulo = corners.map(\.x).min()!, uhi = corners.map(\.x).max()!
        let vlo = corners.map(\.y).min()!, vhi = corners.map(\.y).max()!
        guard nx >= 2, ny >= 2, ulo.isFinite, uhi.isFinite, vlo.isFinite, vhi.isFinite else { return nil }
        let i0 = Int(max(ulo.rounded(.down) - 1, 0)), i1 = Int(min(uhi.rounded(.up), Double(nx - 2)))
        let j0 = Int(max(vlo.rounded(.down) - 1, 0)), j1 = Int(min(vhi.rounded(.up), Double(ny - 2)))
        guard i0 <= i1, j0 <= j1 else { return nil }
        return (i0...i1, j0...j1)
    }

    /// Where face `f` of `solid`, standing at `placement`, meets the terrain
    /// of the landscape standing at `offset`: paths in the solid's own
    /// coordinates, clipped to the face's trim loops.
    ///
    /// The face's plane is X(a, b) = o + a e₁ + b e₂ in the landscape's own
    /// coordinates, (a, b) its frame's (`Solid.FaceFrame`). Over each prism
    /// the plane's points inside it are a convex polygon of (a, b), the
    /// face's rectangle, a little larger than the face, clipped by the
    /// prism's half-spaces; the polygon's sides on the triangle's plane or
    /// the truncation are where X.z = H. A top the face lies within the
    /// scene's tolerance of, all over its rectangle, clips nothing and draws
    /// no side. A stretch of the clipped paths within the tolerance of a
    /// side of the face's loops, along it, is that edge of the solid, and is
    /// left out.
    func contact(onFace f: Int, of solid: Solid, placement: Placement, offset: SIMD3<Double>) -> [[P3<WorldSpace>]] {
        let frame = solid.frames[f], face = solid.face(f)
        let s = placement.scale
        let o = s * frame.origin + placement.offset - offset, e1 = s * frame.u, e2 = s * frame.v
        let (ua, va) = face.domain
        let extent = max(ua.range.length, va.range.length)
        let pad = 0.01 * extent
        let lo = SIMD2(ua.range.lo - pad, va.range.lo - pad), hi = SIMD2(ua.range.hi + pad, va.range.hi + pad)
        let rectangle = [lo, SIMD2(hi.x, lo.y), hi, SIMD2(lo.x, hi.y)]
        var segments: [(SIMD3<Double>, SIMD3<Double>)] = []
        for copy in copies {
            func flat(_ p: SIMD3<Double>, moved: Bool) -> SIMD3<Double> {
                let q = copy.linear * SIMD2(p.x, p.y) + (moved ? copy.offset : .zero)
                return SIMD3(q.x, q.y, p.z)
            }
            let O = flat(o, moved: true), A = flat(e1, moved: false), B = flat(e2, moved: false)
            // The rectangle's box in the landscape's (x, y).
            let corners = rectangle.map { o + $0.x * e1 + $0.y * e2 }
            let xlo = SIMD2(corners.map(\.x).min()!, corners.map(\.y).min()!)
            let xhi = SIMD2(corners.map(\.x).max()!, corners.map(\.y).max()!)
            guard let (ir, jr) = cells(over: xlo, xhi, copy: copy) else { continue }
            for j in jr {
                for i in ir {
                    for upper in [false, true] {
                        // Without the floor, which bounds no top, and on a
                        // slab as thin as a rounding would leave the face
                        // nothing.
                        for prism in prisms(i, j, upper: upper, copy: copy) {
                            let halfSpaces = prism.filter { $0.n.z >= 0 }
                            segments += Slab.tops(rectangle, halfSpaces, origin: O, A, B, size: hi - lo,
                                                  tolerance: start).map { (SIMD3($0.0.x, $0.0.y, 0), SIMD3($0.1.x, $0.1.y, 0)) }
                        }
                    }
                }
            }
        }
        let paths = Slab.chains(segments, tolerance: 1e-9 * extent).map { $0.map { P2<ParamSpace>($0.x, $0.y) } }
        guard let loops = face.trimLoops else { return [] }
        let clipped = Contour.clipped([(0, paths)], over: face.domain, trimLoops: loops)[0].paths
        // Off the face's edges: a segment within the tolerance of one side,
        // all along it, is that side.
        let tolerance = start / s
        let sides = loops.flatMap { loop in zip(loop, loop.dropFirst()).map { ($0.v, $1.v) } }
        func alongSide(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Bool {
            sides.contains { p, q in
                let e = q - p, l2 = simd_length_squared(e)
                guard l2 > 0 else { return false }
                func off(_ x: SIMD2<Double>) -> Double {
                    let t = min(max(simd_dot(x - p, e) / l2, 0), 1)
                    return simd_distance(x, p + t * e)
                }
                return off(a) <= tolerance && off(b) <= tolerance
            }
        }
        var out: [[P3<WorldSpace>]] = []
        for path in clipped {
            var run: [P3<WorldSpace>] = []
            for (a, b) in zip(path, path.dropFirst()) {
                if alongSide(a.v, b.v) {
                    if run.count >= 2 { out.append(run) }
                    run = []
                    continue
                }
                if run.isEmpty { run = [P3(frame.point(a.v))] }
                run.append(P3(frame.point(b.v)))
            }
            if run.count >= 2 { out.append(run) }
        }
        return out
    }

    /// The sides of `polygon` clipped by `halfSpaces` (n · X ≤ c, X =
    /// origin + a A + b B) that lie on a half-space bounding the prism from
    /// above. A top the plane lies within `tolerance` of, all over the
    /// polygon's box of size `size`, clips nothing; a side parallel to the
    /// plane keeps all of the polygon or none.
    static func tops(_ polygon: [SIMD2<Double>], _ halfSpaces: [(n: SIMD3<Double>, c: Double)],
                     origin: SIMD3<Double>, _ A: SIMD3<Double>, _ B: SIMD3<Double>,
                     size: SIMD2<Double>, tolerance: Double) -> [(SIMD2<Double>, SIMD2<Double>)] {
        var polygon = polygon
        let centre = 0.5 * (polygon[0] + polygon[2])
        // The half-space each side lies on; -1 for the rectangle's own.
        var sides = [Int](repeating: -1, count: polygon.count)
        for (k, (n, c)) in halfSpaces.enumerated() {
            let k0 = simd_dot(n, origin) - c, ka = simd_dot(n, A), kb = simd_dot(n, B)
            let reach = abs(ka) * size.x + abs(kb) * size.y
            if n.z > 0 {
                // A top: n · X − c over n.z is how far over it X stands.
                let most = abs(k0 + ka * centre.x + kb * centre.y) + 0.5 * reach
                if most <= tolerance * n.z { continue }
            } else if reach <= 1e-12 * (abs(simd_dot(n, origin)) + abs(c) + simd_length(n)) {
                // Parallel: all of the polygon or none of it.
                if k0 > 0 { return [] }
                continue
            }
            var out: [SIMD2<Double>] = [], outSides: [Int] = []
            for e in polygon.indices {
                let p = polygon[e], q = polygon[(e + 1) % polygon.count]
                let fp = k0 + ka * p.x + kb * p.y, fq = k0 + ka * q.x + kb * q.y
                if fp <= 0 {
                    out.append(p); outSides.append(sides[e])
                    if fq > 0 { out.append(p + (q - p) * (fp / (fp - fq))); outSides.append(k) }
                } else if fq <= 0 {
                    out.append(p + (q - p) * (fp / (fp - fq))); outSides.append(sides[e])
                }
            }
            polygon = out; sides = outSides
            if polygon.count < 3 { return [] }
        }
        return polygon.indices.compactMap { e in
            guard sides[e] >= 0, halfSpaces[sides[e]].n.z > 0 else { return nil }
            let p = polygon[e], q = polygon[(e + 1) % polygon.count]
            return p == q ? nil : (p, q)
        }
    }

    /// Segments joined where their ends meet within `tolerance`: a path
    /// runs through every point where exactly two segments meet and ends
    /// wherever another number do; a closed one ends where it begins. A
    /// segment found twice, along a side two pieces share, is one.
    static func chains(_ segments: [(SIMD3<Double>, SIMD3<Double>)], tolerance: Double) -> [[SIMD3<Double>]] {
        var points: [SIMD3<Double>] = []
        var buckets: [SIMD3<Int64>: [Int]] = [:]
        func key(_ p: SIMD3<Double>) -> SIMD3<Int64> {
            SIMD3(Int64((p.x / tolerance).rounded(.down)), Int64((p.y / tolerance).rounded(.down)),
                  Int64((p.z / tolerance).rounded(.down)))
        }
        func node(_ p: SIMD3<Double>) -> Int {
            let k = key(p)
            for dx in -1...1 {
                for dy in -1...1 {
                    for dz in -1...1 {
                        for n in buckets[k &+ SIMD3(Int64(dx), Int64(dy), Int64(dz))] ?? []
                            where simd_distance(points[n], p) <= tolerance {
                            return n
                        }
                    }
                }
            }
            points.append(p)
            buckets[k, default: []].append(points.count - 1)
            return points.count - 1
        }
        var edges: [SIMD2<Int>] = [], seen = Set<SIMD2<Int>>()
        for (p, q) in segments {
            let a = node(p), b = node(q)
            guard a != b, seen.insert(SIMD2(min(a, b), max(a, b))).inserted else { continue }
            edges.append(SIMD2(a, b))
        }
        var at = [[Int]](repeating: [], count: points.count)
        for (e, ab) in edges.enumerated() { at[ab.x].append(e); at[ab.y].append(e) }
        var used = [Bool](repeating: false, count: edges.count)
        var out: [[SIMD3<Double>]] = []
        func walk(from start: Int, along first: Int) {
            var path = [points[start]], here = start, e = first
            while true {
                used[e] = true
                here = edges[e].x == here ? edges[e].y : edges[e].x
                path.append(points[here])
                guard at[here].count == 2, here != start, let next = at[here].first(where: { !used[$0] }) else { break }
                e = next
            }
            out.append(path)
        }
        for n in points.indices where at[n].count != 2 {
            for e in at[n] where !used[e] { walk(from: n, along: e) }
        }
        for e in edges.indices where !used[e] { walk(from: edges[e].x, along: e) }
        return out
    }

    /// Where `surface` -- the round torus about its z axis, upright as a
    /// placement leaves it, whose distance has Lipschitz constant 1 in space
    /// -- standing at `placement`, meets the terrain of the landscape
    /// standing at `offset`: paths of the landscape's own coordinates, each
    /// point on both surfaces and each chord in one piece of the terrain
    /// (`ContactCurves`).
    func contact(on surface: ParametricSurface, placement: Placement, offset: SIMD3<Double>) -> [[SIMD3<Double>]] {
        guard let e = surface.neighbourhood, nx >= 2, ny >= 2 else { return [] }
        let s = placement.scale, shift = placement.offset - offset
        // The torus's distance from a point of the landscape's coordinates.
        @Sendable func distance(_ x: SIMD3<Double>) -> Double { s * e.foot((x - shift) / s).distance }
        let centre = s * e.centre + shift, radius = s * e.radius
        // The tube's radius and the core's: the ball that holds the torus
        // reaches R + r, and its centre stands R − r off the tube's core.
        let tube = 0.5 * (radius - s * e.foot(e.centre).distance), ring = radius - tube
        let side = min(cell / 4, tube / 8), least = side / 64
        // Where a closed loop of the curve over a piece can be: round a
        // critical point of the distance on the piece's plane. The distance
        // is smooth but on the axis and the core circle, and elsewhere its
        // gradient points away from the core; so its critical points on a
        // plane with normal n are on the axis, on the core circle, or at
        // C(θ) + t n with the core's C(θ) turned toward ±n's horizontal part.
        @Sendable func critical(_ piece: TopPiece) -> [SIMD2<Double>] {
            let up = SIMD3(-piece.gradient.x, -piece.gradient.y, 1), norm = simd_length(up)
            let n = up / norm, c = piece.constant / norm
            let core = centre, flat = SIMD2(n.x, n.y)
            var out: [SIMD3<Double>] = []
            if abs(n.z) > 0 { out.append(SIMD3(core.x, core.y, (c - n.x * core.x - n.y * core.y) / n.z)) }
            func on(_ theta: Double) -> SIMD3<Double> { core + ring * SIMD3(cos(theta), sin(theta), 0) }
            if simd_length(flat) > 0 {
                let theta = atan2(flat.y, flat.x)
                for t in [theta, theta + .pi] { let p = on(t); out.append(p + (c - simd_dot(n, p)) * n) }
                // The core circle in the plane: n · (core + ring θ̂) = c.
                let amplitude = ring * simd_length(flat), d = c - simd_dot(n, core)
                if abs(d) <= amplitude {
                    let phase = atan2(flat.y, flat.x), turn = acos(d / amplitude)
                    out += [on(phase + turn), on(phase - turn)]
                }
            }
            return out.map { SIMD2($0.x, $0.y) }
        }
        let lo = SIMD2(centre.x - radius, centre.y - radius), hi = SIMD2(centre.x + radius, centre.y + radius)
        // The cells the torus can meet, found from blocks of cells: a block
        // is passed over where the torus is farther from the middle of the
        // box that holds its terrain than the box's half-diagonal.
        let ranges = cellRanges()
        var cells: [(copy: Int, i: Int, j: Int)] = []
        for (k, copy) in copies.enumerated() {
            guard let (ir, jr) = self.cells(over: lo, hi, copy: copy) else { continue }
            let back = copy.linear.inverse
            func reaches(_ i0: Int, _ i1: Int, _ j0: Int, _ j1: Int, _ z: (lo: Double, hi: Double)) -> Bool {
                guard z.lo.isFinite, z.hi.isFinite else { return true }
                let corners = [SIMD2(Double(i0), Double(j0)), SIMD2(Double(i1 + 1), Double(j0)),
                               SIMD2(Double(i0), Double(j1 + 1)), SIMD2(Double(i1 + 1), Double(j1 + 1))]
                    .map { back * ($0 - copy.offset) }
                let blo = SIMD3(corners.reduce(corners[0], simd_min), z.lo), bhi = SIMD3(corners.reduce(corners[0], simd_max), z.hi)
                return abs(distance(0.5 * (blo + bhi))) <= 0.5 * simd_distance(blo, bhi) * (1 + 1e-9)
            }
            func descend(_ level: Int, _ bi: Int, _ bj: Int) {
                let size = 1 << level
                let i0 = max(bi * size, ir.lowerBound), i1 = min((bi + 1) * size - 1, ir.upperBound)
                let j0 = max(bj * size, jr.lowerBound), j1 = min((bj + 1) * size - 1, jr.upperBound)
                guard i0 <= i1, j0 <= j1 else { return }
                let (w, z) = (ranges[level].width, ranges[level].z)
                guard reaches(i0, i1, j0, j1, z[bj * w + bi]) else { return }
                if level == 0 { cells.append((k, i0, j0)); return }
                for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] { descend(level - 1, 2 * bi + di, 2 * bj + dj) }
            }
            descend(ranges.count - 1, 0, 0)
        }
        // The pieces over them that the torus can meet.
        let pieces = Heightfield.parallelMap(cells, chunk: 64) { cell -> [TopPiece] in
            [false, true].flatMap { self.tops(cell.i, cell.j, upper: $0, copy: self.copies[cell.copy]) }.filter { piece in
                let points = piece.polygon.map(piece.point)
                let middle = points.reduce(.zero, +) / Double(points.count)
                let reach = points.map { simd_distance($0, middle) }.max()!
                return abs(distance(middle)) <= reach * (1 + 1e-9) + 1e-300
            }
        }.flatMap { $0 }
        let tolerance = start
        let found = Heightfield.parallelMap(pieces, chunk: 4) { piece -> [(SIMD3<Double>, SIMD3<Double>)] in
            if piece.gradient == .zero {
                return Slab.circles(on: piece, axis: SIMD2(centre.x, centre.y), height: centre.z, ring: ring, tube: tube,
                                    least: least, tolerance: tolerance)
            }
            return Slab.zeros(of: distance, on: piece, side: side, least: least, within: (lo, hi), loopsRound: critical(piece))
        }
        return Slab.chains(found.flatMap { $0 }, tolerance: start)
    }

    /// The range of the terrain's height over each cell, and over blocks of
    /// 2ᵏ by 2ᵏ cells, level by level, up to one block: what bounds the top
    /// over a cell, the least of its triangles' planes and the truncation,
    /// from below and above. Unbounded over a cell with a corner not a
    /// number.
    func cellRanges() -> [(width: Int, z: [(lo: Double, hi: Double)])] {
        let least = bands.map(\.height).min() ?? .infinity
        let (cx, cy) = (nx - 1, ny - 1)
        var z = [(lo: Double, hi: Double)](repeating: (0, 0), count: cx * cy)
        for j in 0..<cy {
            for i in 0..<cx {
                let h = [heights[j * nx + i], heights[j * nx + i + 1], heights[(j + 1) * nx + i], heights[(j + 1) * nx + i + 1]]
                z[j * cx + i] = h.allSatisfy(\.isFinite) ? (min(h.min()!, least), h.max()!) : (-.infinity, .infinity)
            }
        }
        var levels = [(width: cx, z: z)]
        var (w, hgt) = (cx, cy)
        while w > 1 || hgt > 1 {
            let (w2, h2) = ((w + 1) / 2, (hgt + 1) / 2)
            let below = levels[levels.count - 1].z
            var up = [(lo: Double, hi: Double)](repeating: (.infinity, -.infinity), count: w2 * h2)
            for j in 0..<hgt {
                for i in 0..<w {
                    let a = below[j * w + i], k = (j / 2) * w2 + i / 2
                    up[k] = (min(up[k].lo, a.lo), max(up[k].hi, a.hi))
                }
            }
            levels.append((w2, up)); (w, hgt) = (w2, h2)
        }
        return levels
    }

    /// Where a level piece, z = h over its polygon, meets the torus about a
    /// vertical axis through `axis`, its core of radius `ring` at `height`
    /// and its tube of radius `tube`: the circles about the axis of radius
    /// ring ± √(tube² − (h − height)²), clipped to the polygon, each arc cut
    /// into chords whose sagitta is under half of `least`. A plane the torus
    /// reaches past by no more than `tolerance`, or not at all, meets it
    /// nowhere.
    static func circles(on piece: TopPiece, axis: SIMD2<Double>, height: Double, ring: Double, tube: Double,
                        least: Double, tolerance: Double) -> [(SIMD3<Double>, SIMD3<Double>)] {
        let h = piece.constant, dz = h - height
        guard tube - abs(dz) > tolerance else { return [] }
        let half = (tube * tube - dz * dz).squareRoot()
        let polygon = piece.polygon
        var out: [(SIMD3<Double>, SIMD3<Double>)] = []
        for rho in [ring - half, ring + half] where rho > 0 {
            func at(_ t: Double) -> SIMD2<Double> { axis + rho * SIMD2(cos(t), sin(t)) }
            // Where the circle crosses each side, solved on the side, and
            // its angle about the axis.
            var crossings: [(angle: Double, point: SIMD2<Double>)] = []
            for k in polygon.indices {
                let a = polygon[k], e = polygon[(k + 1) % polygon.count] - a, w = a - axis
                // |w + t e|² = ρ²: t² (e·e) + 2t (w·e) + (w·w − ρ²) = 0.
                let qa = simd_dot(e, e), qb = simd_dot(w, e), qc = (simd_length(w) - rho) * (simd_length(w) + rho)
                guard qa > 0 else { continue }
                let disc = qb * qb - qa * qc
                guard disc >= 0 else { continue }
                let root = disc.squareRoot()
                // The roots without cancellation.
                let q = -(qb + (qb >= 0 ? root : -root))
                for t in [q / qa, q != 0 ? qc / q : 0] where t >= 0 && t <= 1 {
                    let p = a + t * e
                    crossings.append((atan2(p.y - axis.y, p.x - axis.x), p))
                }
            }
            crossings.sort { $0.angle < $1.angle }
            if crossings.isEmpty { crossings = [(0, at(0))] }
            // The arcs between consecutive crossings whose middle is inside.
            for k in crossings.indices {
                let (t0, p0) = crossings[k]
                let (t1, p1) = k + 1 < crossings.count ? crossings[k + 1] : (crossings[0].angle + 2 * .pi, crossings[0].point)
                guard t1 - t0 > 0, piece.contains(at(0.5 * (t0 + t1))) else { continue }
                let step = 2 * acos(max(1 - 0.5 * least / rho, -1))
                let count = max(Int(((t1 - t0) / step).rounded(.up)), 1)
                var previous = p0
                for m in 1...count {
                    let next = m == count ? p1 : at(t0 + (t1 - t0) * Double(m) / Double(count))
                    out.append((SIMD3(previous.x, previous.y, h), SIMD3(next.x, next.y, h)))
                    previous = next
                }
            }
        }
        return out
    }

    /// The zero set of a function with Lipschitz constant 1 in space on a
    /// piece's plane over its polygon, as segments. The polygon, clipped to
    /// `box`, is passed over where the bound shows no zero and split in four
    /// otherwise, down to squares of `side`. Each side of such a square is
    /// searched for every zero, its halves while the bound allows one, down
    /// to `least`, and each zero bisected to the last bit. A closed loop of
    /// the zero set lies round a critical point of f, so a square that holds
    /// none of `loopsRound` holds arcs alone, each from one zero on its sides
    /// to another: with none it holds no zero, and with two it draws the
    /// chord between them where Newton's step from each of its quarters to
    /// the zero set is within half of `least`. A square with more, or a chord
    /// that strays, or a critical point, is split in four, down to `least`,
    /// where the zeros on its sides are joined in pairs in turn. So every arc
    /// is found that crosses a side of a square apart from its neighbours by
    /// `least`, and every loop larger than `least`.
    static func zeros(of f: @Sendable (SIMD3<Double>) -> Double, on piece: TopPiece, side: Double, least: Double,
                      within box: (lo: SIMD2<Double>, hi: SIMD2<Double>),
                      loopsRound critical: [SIMD2<Double>]) -> [(SIMD3<Double>, SIMD3<Double>)] {
        func value(_ p: SIMD2<Double>) -> Double { f(piece.point(p)) }
        // Every zero along a → b, in order from a.
        func along(_ a: SIMD2<Double>, _ fa: Double, _ b: SIMD2<Double>, _ fb: Double) -> [SIMD2<Double>] {
            let length = simd_distance(piece.point(a), piece.point(b))
            if abs(fa) + abs(fb) > length * (1 + 1e-9) { return [] }
            if simd_distance(a, b) <= least {
                guard (fa > 0) != (fb > 0) else { return [] }
                var under = fa > 0 ? b : a, over = fa > 0 ? a : b
                for _ in 0..<80 {
                    let m = 0.5 * (under + over)
                    guard m != under, m != over else { break }
                    if value(m) > 0 { over = m } else { under = m }
                }
                return [under]
            }
            let m = 0.5 * (a + b), fm = value(m)
            return along(a, fa, m, fm) + along(m, fm, b, fb)
        }
        // Whether Newton's step from p to the zero set, along f's gradient
        // in the plane's (x, y), is within half of `least`.
        func near(_ p: SIMD2<Double>) -> Bool {
            let v = value(p), h = 1e-3 * least
            let g = SIMD2(value(p + SIMD2(h, 0)) - value(p - SIMD2(h, 0)), value(p + SIMD2(0, h)) - value(p - SIMD2(0, h))) / (2 * h)
            return abs(v) <= 0.5 * least * simd_length(g)
        }
        func split(_ cut: [SIMD2<Double>]) -> [[SIMD2<Double>]] {
            let clo = cut.reduce(cut[0], simd_min), chi = cut.reduce(cut[0], simd_max), mid = 0.5 * (clo + chi)
            return [(-1.0, -1.0), (1, -1), (-1, 1), (1, 1)].compactMap { sx, sy in
                let quarter = clip2D(clip2D(cut) { sx * (mid.x - $0.x) }) { sy * (mid.y - $0.y) }
                return quarter.count >= 3 ? quarter : nil
            }
        }
        func inside(_ cut: [SIMD2<Double>]) -> [(SIMD3<Double>, SIMD3<Double>)] {
            let points = cut.map(piece.point)
            let middle = cut.reduce(.zero, +) / Double(cut.count)
            let reach = points.map { simd_distance($0, piece.point(middle)) }.max()!
            guard abs(value(middle)) <= reach * (1 + 1e-9) else { return [] }
            let clo = cut.reduce(cut[0], simd_min), chi = cut.reduce(cut[0], simd_max)
            let size = max(chi.x - clo.x, chi.y - clo.y)
            if size > side * (1 + 1e-9) { return split(cut).flatMap(inside) }
            let values = cut.map(value)
            var crossings: [SIMD2<Double>] = []
            for k in cut.indices {
                let l = (k + 1) % cut.count
                crossings += along(cut[k], values[k], cut[l], values[l])
            }
            let loops = critical.contains { all(simd_min(chi + least, simd_max($0, clo - least)) .== $0) }
            if !loops, crossings.isEmpty { return [] }
            if crossings.count == 2, !loops || size <= least * 2 {
                let a = crossings[0], b = crossings[1]
                if size <= least * 2 || [0.25, 0.5, 0.75].allSatisfy({ near(a + $0 * (b - a)) }) {
                    return [(piece.point(a), piece.point(b))]
                }
            }
            if size <= least * 2 {
                return stride(from: 0, to: crossings.count - 1, by: 2).map {
                    (piece.point(crossings[$0]), piece.point(crossings[$0 + 1]))
                }
            }
            return split(cut).flatMap(inside)
        }
        var cut = piece.polygon
        cut = clip2D(clip2D(cut) { box.lo.x - $0.x }) { $0.x - box.hi.x }
        cut = clip2D(clip2D(cut) { box.lo.y - $0.y }) { $0.y - box.hi.y }
        return cut.count >= 3 ? inside(cut) : []
    }
}
