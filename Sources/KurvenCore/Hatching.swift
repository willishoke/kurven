import Foundation
import simd

/// Hatching: parallel lines at an angle and a spacing, each drawn at a width
/// set by a tone read along it.
///
/// kurven draws strokes only, never areas, so a face is shaded by hatching
/// it. The tone decides two things about each hatch line: where it is drawn,
/// and how wide. It is read along the line every `step`, and wherever the
/// width changes between two readings the change is bisected on the tone, so
/// a line ends on the tone's own level set to the last bit rather than at
/// the nearest reading. The lines are then tested for visibility like any
/// curve on a face (`Solid.visibleCurves`): the drawing depends on the
/// solid, the tone and the camera, and not on a resolution.
public struct Hatching: Sendable, Equatable {
    /// The direction of the lines in the face's plane, in radians from its
    /// `u` axis (`Solid.Face`).
    public var angle: Double
    /// Between neighbouring lines, in world units.
    public var spacing: Double
    /// How often the tone is read along a line, in world units. A change of
    /// width narrower than this can fall between two readings. The spacing
    /// by default, so the tone is read as finely as it is drawn.
    public var step: Double
    public var tone: Tone
    /// How many widths a line can have. One: a line is drawn at the layer's
    /// width where the tone is positive and nowhere else. More: the tone in
    /// `(0, 1]` is cut into that many steps and the width follows it,
    /// `k / widthSteps` of the layer's width on step `k`.
    public var widthSteps: Int

    public init(angle: Double, spacing: Double, step: Double? = nil,
                tone: Tone = .uniform, widthSteps: Int = 1) {
        precondition(spacing > 0 && spacing.isFinite, "hatching has a positive spacing")
        precondition(widthSteps >= 1, "hatching has at least one width")
        self.angle = angle; self.spacing = spacing; self.step = step ?? spacing
        self.tone = tone; self.widthSteps = widthSteps
    }

    /// The width step a tone falls on: 0 for none, `widthSteps` for the
    /// full width.
    public func widthStep(of tone: Double) -> Int {
        guard tone > 0 else { return 0 }
        return min(widthSteps, Int((tone * Double(widthSteps)).rounded(.up)))
    }

    // MARK: manifest

    public init(json: JSONValue) throws {
        let o = try json.object("LayerSource.hatching")
        let tone: Tone
        let t = try o.value("tone", "LayerSource.hatching").object("LayerSource.hatching.tone")
        switch try t.string("kind", "LayerSource.hatching.tone") {
        case "uniform":
            tone = .uniform
        case "turing":
            tone = .turing(scale: try t.double("scale", "LayerSource.hatching.tone"),
                           seed: UInt64(bitPattern: Int64(try t.int("seed", "LayerSource.hatching.tone"))))
        case "lambert":
            let l = try t.doubles("light", "LayerSource.hatching.tone")
            guard l.count == 3 else {
                throw JSONError.wrongType("light", expected: "three numbers", found: "LayerSource.hatching.tone")
            }
            tone = .lambert(light: SIMD3(l[0], l[1], l[2]))
        case let other:
            throw ManifestError.unknownKind(other, of: "Tone", known: ["uniform", "turing", "lambert"])
        }
        let spacing = try o.double("spacing", "LayerSource.hatching")
        let widthSteps = try o.int("widthSteps", "LayerSource.hatching")
        guard spacing > 0, spacing.isFinite else {
            throw JSONError.wrongType("spacing", expected: "a positive number", found: "LayerSource.hatching")
        }
        guard widthSteps >= 1 else {
            throw JSONError.wrongType("widthSteps", expected: "a positive integer", found: "LayerSource.hatching")
        }
        self.init(angle: try o.double("angle", "LayerSource.hatching"), spacing: spacing,
                  step: o.optionalDouble("step"), tone: tone, widthSteps: widthSteps)
    }

    public var json: JSONValue {
        let tone: JSONValue
        switch self.tone {
        case .uniform:
            tone = .object(["kind": .string("uniform")])
        case .turing(let scale, let seed):
            tone = .object(["kind": .string("turing"), "scale": .double(scale),
                            "seed": .int(Int(Int64(bitPattern: seed)))])
        case .lambert(let light):
            tone = .object(["kind": .string("lambert"),
                            "light": .array([.double(light.x), .double(light.y), .double(light.z)])])
        }
        return .object(["kind": .string("hatching"), "angle": .double(angle), "spacing": .double(spacing),
                        "step": .double(step), "widthSteps": .int(widthSteps), "tone": tone])
    }
}

/// The tone hatching reads, as a function on a face's plane with values in
/// `[-1, 1]`. Lines are drawn where it is positive.
public enum Tone: Sendable, Equatable {
    /// One everywhere: the whole face, hatched.
    case uniform
    /// Turing's dappled pattern (`Turing`), in face coordinates divided by
    /// `scale`, with signs drawn from `seed`.
    case turing(scale: Double, seed: UInt64)
    /// Lambert's cosine law, remapped to `[0, 1]`: `(1 - n·l) / 2` for the
    /// face's outward normal `n` and the unit direction `l` toward the
    /// light. Nothing on a face square to the light, full width on one
    /// turned away from it, one value per face.
    case lambert(light: SIMD3<Double>)

    /// The tone on one face, in that face's coordinates.
    public func value(on face: Solid.Face) -> @Sendable (SIMD2<Double>) -> Double {
        switch self {
        case .uniform:
            return { _ in 1 }
        case .turing(let scale, let seed):
            return { c in Turing.value(c / scale, seed: seed) }
        case .lambert(let light):
            let w = (1 - simd_dot(face.normal, simd_normalize(light))) / 2
            return { _ in w }
        }
    }
}

/// Turing's dappled pattern, in closed form: unit Gaussians centred on a
/// square lattice, each with a random sign,
///
///     f(x, y) = Σ U(r, s) exp(-((x - hr)² + (y - hs)²) / 2),  U(r, s) = ±1,
///
/// with h = 0.7. This is the closed form behind the dappled figure of
/// Turing's "The chemical basis of morphogenesis" (Phil. Trans. R. Soc. B
/// 237, 1952), as reconstructed in "Turing Patterns" (willishoke.github.io,
/// 2022-06-01). It is not a reaction-diffusion simulation. The signs are a
/// hash of the lattice point and the seed, so the pattern is a function on
/// the whole plane with no edge and no array behind it.
public enum Turing {
    /// The lattice spacing h.
    public static let spacing = 0.7
    /// How far a kernel reaches before it is dropped: six, where it is
    /// under 2e-8 of its peak.
    static let reach = 6.0
    /// The sum's standard deviation: the kernels' squares summed over the
    /// lattice, `π / h²` by the integral, rooted.
    public static let deviation = Double.pi.squareRoot() / spacing

    /// The sign at lattice point `(r, s)` under `seed`: a mix of the three,
    /// its lowest bit.
    public static func sign(_ r: Int, _ s: Int, seed: UInt64) -> Double {
        var z = seed ^ (UInt64(bitPattern: Int64(r)) &* 0x9E37_79B9_7F4A_7C15)
            ^ (UInt64(bitPattern: Int64(s)) &* 0xD1B5_4A32_D192_ED03)
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return z & 1 == 0 ? -1 : 1
    }

    /// The raw sum at `p`.
    public static func sum(_ p: SIMD2<Double>, seed: UInt64) -> Double {
        let r0 = Int(((p.x - reach) / spacing).rounded(.up)), r1 = Int(((p.x + reach) / spacing).rounded(.down))
        let s0 = Int(((p.y - reach) / spacing).rounded(.up)), s1 = Int(((p.y + reach) / spacing).rounded(.down))
        var total = 0.0
        guard r0 <= r1, s0 <= s1 else { return 0 }
        for r in r0...r1 {
            let dx = p.x - Double(r) * spacing
            for s in s0...s1 {
                let dy = p.y - Double(s) * spacing
                total += sign(r, s, seed: seed) * exp(-0.5 * (dx * dx + dy * dy))
            }
        }
        return total
    }

    /// The pattern: the sum over twice its deviation, clamped to `[-1, 1]`.
    public static func value(_ p: SIMD2<Double>, seed: UInt64) -> Double {
        min(max(sum(p, seed: seed) / (2 * deviation), -1), 1)
    }
}

// MARK: - hatching a solid

public extension Solid {
    /// One hatch line: the face it lies on, its width step, and its path in
    /// world space.
    struct HatchLine: Sendable {
        public let face: Int
        public let widthStep: Int
        public let path: [P3<WorldSpace>]
    }

    /// The hatching laid over every face, before visibility.
    func hatchLines(_ hatching: Hatching) -> [HatchLine] {
        faces.indices.flatMap { hatchLines(hatching, onFace: $0) }
    }

    /// Every hatch line's path, before visibility: the curves a hatching
    /// layer carries before the camera is known.
    func hatchPaths(_ hatching: Hatching) -> PolylineSet<WorldSpace> {
        PolylineSet(paths: hatchLines(hatching).map(\.path))
    }

    /// The hatching laid over face `i`.
    ///
    /// The face's plane is covered by lines at the hatching's angle, one
    /// every spacing, through the frame's origin and its multiples, so a
    /// face's hatching is fixed by its own frame and not by which of its
    /// pieces is asked. Each line is clipped to every piece of the face -- a
    /// convex polygon, so an interval -- and the intervals are joined where
    /// they meet, which is across every diagonal. The two pieces compute the
    /// crossing from the same two vertices but not always to the same bit;
    /// intervals within `1e-12` of the face's extent of each other are joined.
    /// A line lying on a side of the face is not drawn: that is an edge, and
    /// the edge is drawn. One lying on a diagonal is. The tone is then read
    /// along each run every `step`, and where the width step changes between
    /// two readings the change is bisected on the tone.
    func hatchLines(_ hatching: Hatching, onFace i: Int) -> [HatchLine] {
        let face = face(i)
        let polygons: [[SIMD2<Double>]] = face.pieces.map { p in
            pieces[p].map { face.coordinates(vertices[$0].v) }
        }
        // Per side of each piece, whether it is a diagonal of the face
        // rather than an edge of the solid.
        let diagonals = Set(edges.indices.filter { isDiagonal[$0] }.map { SIMD2(edges[$0].a, edges[$0].b) })
        let onDiagonal: [[Bool]] = face.pieces.map { p in
            let piece = pieces[p]
            return piece.indices.map { j in
                let a = piece[j], b = piece[(j + 1) % piece.count]
                return diagonals.contains(SIMD2(min(a, b), max(a, b)))
            }
        }
        let dir = SIMD2(cos(hatching.angle), sin(hatching.angle))
        let per = SIMD2(-dir.y, dir.x)
        let tone = hatching.tone.value(on: face)
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }
        var lo = Double.infinity, hi = -Double.infinity
        for poly in polygons { for p in poly {
            let c = simd_dot(p, per)
            lo = min(lo, c); hi = max(hi, c)
        } }
        guard lo.isFinite, hi.isFinite, hi > lo else { return [] }
        let extent = hi - lo
        var out: [HatchLine] = []
        let kLo = Int((lo / hatching.spacing).rounded(.up)), kHi = Int((hi / hatching.spacing).rounded(.down))
        guard kLo <= kHi else { return [] }
        for k in kLo...kHi {
            let c = Double(k) * hatching.spacing
            let base = c * per
            // The line's overlap with each piece: half-plane by half-plane.
            var intervals: [(Double, Double)] = []
            for (poly, diagonal) in zip(polygons, onDiagonal) {
                var t0 = -Double.infinity, t1 = Double.infinity
                var empty = false
                for j in 0..<poly.count {
                    let p = poly[j], q = poly[(j + 1) % poly.count]
                    let e = q - p
                    let f0 = cross(e, base - p), f1 = cross(e, dir)
                    if abs(f1) <= 1e-15 * extent {
                        // Parallel to the side. A line on it is outside the
                        // piece when the side is an edge, so a line along
                        // the face's boundary is left to the edge, and
                        // inside when it is a diagonal, so a line along a
                        // diagonal is drawn, once, across the join.
                        let on = abs(f0) <= 1e-12 * extent * simd_length(e)
                        if on ? !diagonal[j] : f0 < 0 { empty = true; break }
                    } else if f1 > 0 {
                        t0 = max(t0, -f0 / f1)
                    } else {
                        t1 = min(t1, -f0 / f1)
                    }
                }
                if !empty, t0 < t1, t0.isFinite, t1.isFinite { intervals.append((t0, t1)) }
            }
            intervals.sort { $0.0 < $1.0 }
            var runs: [(Double, Double)] = []
            for (a, b) in intervals {
                if let last = runs.last, a <= last.1 + 1e-12 * extent {
                    runs[runs.count - 1].1 = max(last.1, b)
                } else {
                    runs.append((a, b))
                }
            }
            for (a, b) in runs where b - a > 1e-12 * extent {
                func widthStep(_ t: Double) -> Int { hatching.widthStep(of: tone(base + t * dir)) }
                for (ta, tb, w) in Solid.graded(from: a, to: b, step: hatching.step, widthStep: widthStep)
                    where w > 0 {
                    out.append(HatchLine(face: i, widthStep: w,
                                         path: [P3(face.point(base + ta * dir)),
                                                P3(face.point(base + tb * dir))]))
                }
            }
        }
        return out
    }

    /// `[a, b]` cut into intervals of one width step each: the step read
    /// every `step`, and every change between two readings bisected on the
    /// width step itself, to the last bit of the parameter. A change that
    /// reverses between two readings is not seen; that is what the reading
    /// step means.
    static func graded(from a: Double, to b: Double, step: Double,
                       widthStep: (Double) -> Int) -> [(Double, Double, Int)] {
        let length = b - a
        let readings = step > 0 && step.isFinite ? max(1, Int((length / step).rounded(.up))) : 1
        var intervals: [(Double, Double, Int)] = []
        var start = a
        var current = widthStep(a)
        // `known` is the last point whose width step is `current`.
        var known = a
        for j in 1...readings {
            let t = j == readings ? b : a + length * Double(j) / Double(readings)
            // Between two readings of different steps: the first change,
            // then whatever follows it, until the reading's step is reached.
            // Each change is bisected until `lo` and `hi` are neighbouring
            // numbers, where the midpoint no longer moves.
            while widthStep(t) != current {
                var lo = known, hi = t
                while true {
                    let mid = 0.5 * (lo + hi)
                    guard mid > lo, mid < hi else { break }
                    if widthStep(mid) == current { lo = mid } else { hi = mid }
                }
                intervals.append((start, hi, current))
                start = hi; known = hi; current = widthStep(hi)
            }
            known = t
        }
        intervals.append((start, b, current))
        // Neighbours of one width step are one interval.
        var merged: [(Double, Double, Int)] = []
        for p in intervals where p.1 > p.0 {
            if let last = merged.last, last.2 == p.2, last.1 == p.0 {
                merged[merged.count - 1].1 = p.1
            } else {
                merged.append(p)
            }
        }
        return merged
    }

    /// The hatching for a camera: per width step, the hatch lines before
    /// and after visibility, so a layer per width step can be drawn at its
    /// width.
    func visibleHatching(_ hatching: Hatching, view: Transform<WorldSpace, ViewSpace>)
        -> [(widthStep: Int, all: PolylineSet<WorldSpace>, visible: PolylineSet<WorldSpace>)] {
        let occlusion = self.occlusion(view: view)
        let lines = hatchLines(hatching)
        var out: [(widthStep: Int, all: PolylineSet<WorldSpace>, visible: PolylineSet<WorldSpace>)] = []
        for w in 1...hatching.widthSteps {
            let mine = lines.filter { $0.widthStep == w }
            guard !mine.isEmpty else { continue }
            var visible: [[P3<WorldSpace>]] = []
            for i in faces.indices {
                let paths = mine.filter { $0.face == i }.map(\.path)
                guard !paths.isEmpty else { continue }
                visible += visibleCurves(paths, onFace: i, occlusion: occlusion)
            }
            out.append((w, PolylineSet(paths: mine.map(\.path)), PolylineSet(paths: visible)))
        }
        return out
    }
}
