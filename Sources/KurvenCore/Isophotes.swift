import Foundation
import simd

/// Light on a curved face: its isophotes, and hatching along a coordinate
/// whose widths half-Lambert darkness sets.
///
/// An isophote is a level set of n · l, the outward unit normal against the
/// unit direction toward a light (Koenderink and van Doorn 1980). On a flat
/// face n is one vector, so n · l is constant and has no level set: a flat
/// face has no isophote field, by its kind, not by what its samples happen
/// to round to. A curved face (the graph, a parametric surface) is
/// contoured by the plain routine, as its coordinates and height are.
///
/// A curved face is hatched along a coordinate of its parameter rectangle
/// (D8): lines at a spacing are evenly spaced only where the rectangle is an
/// isometry, and a coordinate is the one direction every curved face has.
/// The tone sets each line's width, read along it, and the width changes
/// where the tone crosses a step, so on a half-Lambert tone a line's width
/// changes exactly on the isophotes n · l = 1 − 2k / widthSteps.
public extension Face {
    /// Whether the face is flat: a plateau, a riser, a wall or a polyhedron
    /// face, whose normal is one vector.
    var isFlat: Bool {
        switch id.kind {
        case .plateau, .riser, .wall, .solid: true
        case .graph, .surface: false
        }
    }

    /// The outward unit normal of `copy` at a point of the parameter plane:
    /// `du × dv` of the copy's jet (`Copy.place`), normalized and turned by
    /// its outward sign. Nil where the face bounds no solid, so has no
    /// outward side, or where the normal vanishes.
    func outwardNormal(at p: P2<ParamSpace>, on copy: Copy) -> SIMD3<Double>? {
        guard let outward = copy.outward else { return nil }
        let n = copy.place(map(p)).normal
        let length = simd_length(n)
        guard length > 0, length.isFinite else { return nil }
        return outward * (n / length)
    }

    /// n · l on `copy`, for a light toward `light`, normalized: the field
    /// whose level sets are the isophotes. Nil on a flat face, on a face with
    /// no outward side, and for a light with no direction. Not a number
    /// where the normal vanishes.
    func isophote(light: SIMD3<Double>, on copy: Copy) -> ScalarField? {
        guard !isFlat, copy.outward != nil, Tone.hasDirection(light) else { return nil }
        let l = simd_normalize(light), face = self
        return ScalarField { p in
            face.outwardNormal(at: p, on: copy).map { simd_dot($0, l) } ?? .nan
        }
    }

    /// The isophotes of `copy` at `levels`, clipped to its trimmed domain
    /// (`contours(of:_:on:)`), each vertex its point of the parameter plane.
    /// Nothing on a flat face or one with no outward side. A level at ±1 or
    /// beyond draws nothing: n · l reaches ±1 only where n is ±l, where it
    /// touches its extreme value instead of crossing it, and whether
    /// marching squares found a curve there would be decided by rounding.
    func isophotes(light: SIMD3<Double>, _ levels: [Double], on copy: Copy)
        -> [(level: Double, paths: [[P2<ParamSpace>]])] {
        guard let field = isophote(light: light, on: copy) else { return [] }
        return contours(of: field, levels.filter { abs($0) < 1 }, on: copy)
    }

    /// The levels of the coordinate across hatching at `spacing`, along
    /// `axis`'s other coordinate: the spacing asks for a count, and the lines
    /// are spread to fit it. On a periodic axis a whole number of gaps round
    /// the period, the first line on the axis's start, as coordinate curves
    /// stand; on a bounded one as a plateau's hatching divides the domain
    /// (`Surface.plateauHatching(axis:spacing:tiled:)`), the two lines on its
    /// ends left out, so on a landscape the graph's lines continue the
    /// plateaus' across the truncation edge.
    static func hatchLevels(across axis: ParamAxis, spacing: Double) -> [Double] {
        guard spacing.isFinite, spacing > 0 else { return [] }
        let (lo, length) = (axis.range.lo, axis.range.length)
        if axis.periodic {
            let n = max(1, Int((abs(length) / spacing).rounded()))
            return (0..<n).map { lo + length * Double($0) / Double(n) }
        }
        let n = Surface.strokeCount(length: axis.range.hi - lo, spacing: spacing)
        return (1..<max(n - 1, 1)).map { lo + (axis.range.hi - lo) * Double($0) / Double(n - 1) }
    }

    /// Hatching along a coordinate of a curved face, on `copy`, by width
    /// step in increasing order, each piece a polyline of the parameter
    /// plane. Lines along u (angle 0) at levels of v, along v (angle π/2) at
    /// levels of u (`hatchLevels(across:spacing:)`, the spacing in the
    /// coordinate's own units), each line the contour of the coordinate
    /// across it (`contours(of:_:on:)`): clipped to the trimmed domain
    /// exactly, with a vertex wherever it crosses a sample of the parameter
    /// rectangle. Each line is then cut where its width step changes: the
    /// step of the tone (`Hatching.widthStep(of:)`) read at every vertex,
    /// and a change between two vertices bisected on it to the last bit of
    /// the chord between them (`Solid.graded(from:to:step:widthStep:)`), so a
    /// piece has one width and ends where the next begins. A change that
    /// reverses between two vertices is not seen: the vertices are the
    /// readings. A piece of step 0 is not drawn.
    ///
    /// The tone: uniform, or half-Lambert darkness `(1 − n · l) / 2` at each
    /// point (`Tone.darkness(normal:light:)`), n the outward unit normal
    /// there (`outwardNormal(at:on:)`). Nothing on a flat face, at an angle
    /// along neither coordinate, under Turing's tone, or, for half-Lambert,
    /// on a face with no outward side.
    func coordinateHatching(_ hatching: Hatching, on copy: Copy)
        -> [(widthStep: Int, paths: [[P2<ParamSpace>]])] {
        guard !isFlat, let axis = DomainAxis(angle: hatching.angle) else { return [] }
        let tone: @Sendable (P2<ParamSpace>) -> Double
        switch hatching.tone {
        case .uniform:
            tone = { _ in 1 }
        case .halfLambert(let light):
            guard copy.outward != nil, Tone.hasDirection(light) else { return [] }
            let face = self
            tone = { p in face.outwardNormal(at: p, on: copy).map { Tone.darkness(normal: $0, light: light) } ?? 0 }
        case .turing:
            return []
        }
        // Along u, the lines are levels of v; along v, of u.
        let alongU = axis == .real
        let across = alongU ? domain.v : domain.u
        let coordinate = ScalarField(period: across.period) { alongU ? $0.y : $0.x }
        let lines = contours(of: coordinate, Face.hatchLevels(across: across, spacing: hatching.spacing), on: copy)
        var byStep: [Int: [[P2<ParamSpace>]]] = [:]
        for line in lines.flatMap(\.paths) {
            for (step, piece) in Face.graded(line, widthStep: { hatching.widthStep(of: tone($0)) }) where step > 0 {
                byStep[step, default: []].append(piece)
            }
        }
        return byStep.keys.sorted().map { ($0, byStep[$0]!) }
    }

    /// A polyline cut into pieces of one width step each: the step read at
    /// every vertex, and each change between two bisected along the chord
    /// between them, the piece before ending and the next beginning there.
    static func graded(_ path: [P2<ParamSpace>], widthStep: (P2<ParamSpace>) -> Int)
        -> [(widthStep: Int, piece: [P2<ParamSpace>])] {
        guard let first = path.first, path.count >= 2 else { return [] }
        var out: [(widthStep: Int, piece: [P2<ParamSpace>])] = []
        var current = widthStep(first), piece = [first]
        for (a, b) in zip(path, path.dropFirst()) {
            func at(_ t: Double) -> P2<ParamSpace> {
                t <= 0 ? a : t >= 1 ? b : P2(a.v + (b.v - a.v) * t)
            }
            for (start, _, step) in Solid.graded(from: 0, to: 1, step: 1, widthStep: { widthStep(at($0)) })
                where step != current {
                // A change at `start`: the piece so far ends there, and the
                // next begins there.
                let p = at(start)
                piece.append(p)
                if piece.count >= 2 { out.append((current, piece)) }
                current = step
                piece = [p]
            }
            piece.append(b)
        }
        if piece.count >= 2 { out.append((current, piece)) }
        return out
    }
}

public extension ParametricSurface {
    /// Hatching along a coordinate of the surface's one face, by width step
    /// in increasing order (`Face.coordinateHatching(_:on:)`), each piece
    /// carried into space with its surface coordinate beside every vertex.
    func coordinateHatching(_ hatching: Hatching) -> [(widthStep: Int, paths: PolylineSet<WorldSpace>)] {
        let face = self.face, map = self.map
        return face.coordinateHatching(hatching, on: face.copies[0]).map { step, coords in
            (step, PolylineSet(paths: coords.map { $0.map { P3(map($0).position) } }, coords: coords))
        }
    }

    /// The isophotes of the surface's one face at `levels`, carried into
    /// space with their surface coordinates (`Face.isophotes(light:_:on:)`).
    func isophotes(light: SIMD3<Double>, _ levels: [Double]) -> PolylineSet<WorldSpace> {
        let face = self.face, map = self.map
        let coords = face.isophotes(light: light, levels, on: face.copies[0]).flatMap(\.paths)
        return PolylineSet(paths: coords.map { $0.map { P3(map($0).position) } }, coords: coords)
    }
}

public extension Surface {
    /// Hatching along a coordinate of the graph, by width step in increasing
    /// order, in world space: each copy's (`Face.coordinateHatching(_:on:)`),
    /// lifted to |f| and carried to its tile, on every tile where `tiled`.
    /// Lines along Re z at angle 0, along Im z at π/2, at the levels the
    /// plateaus' hatching takes.
    func graphHatching(_ hatching: Hatching, tiled: Bool, in context: HatchContext,
                       refine: ContourRefine? = nil) -> [(widthStep: Int, paths: [[P3<WorldSpace>]])] {
        guard let face = graph(tiles: tiled ? context.tiles : [.identity], region: context.region,
                               refine: refine) else { return [] }
        let magnitude = faceMagnitude(refine)
        var byStep: [Int: [[P3<WorldSpace>]]] = [:]
        for copy in face.copies {
            for (step, pieces) in face.coordinateHatching(hatching, on: copy) {
                byStep[step, default: []] += pieces.map { piece in
                    piece.map { p in copy.tile(P3<WorldSpace>(p.x, p.y, magnitude(P2(p.x, p.y)))) }
                }
            }
        }
        return byStep.keys.sorted().map { ($0, byStep[$0]!) }
    }
}
