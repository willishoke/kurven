import Foundation
import simd

/// A pattern ruled over the facets of a solid: parallel strokes at an angle
/// and a spacing, each stroke weighted by a field read at the point.
///
/// A fill is ink, not paint. The plates have no grey: a face is shaded by
/// ruling it, and a pattern on a face is where the ruling is and how heavy
/// it is. So every fill here is a hatch, and the field decides two things
/// about each stroke -- where it exists, and at what weight -- in closed
/// form along the stroke: the field is read along it, and wherever its
/// weight steps the step is bisected on the field, so a stroke ends on the
/// pattern's edge to the last bit, not at the nearest reading. The strokes
/// are then judged like any ink in a face (`Solid.visibleInk`): the drawing
/// is a function of the solid, the field and the camera, and of no
/// resolution.
public struct Fill: Sendable, Equatable {
    /// The rulings' direction in the facet's plane, in radians from its `u`
    /// axis (`Solid.Facet`).
    public var angle: Double
    /// Between neighbouring rulings, in world units.
    public var spacing: Double
    /// How often the field is read along a ruling, in world units. A change
    /// of weight narrower than this can fall between two readings; the
    /// spacing by default, so the pattern is read as finely as it is drawn.
    public var pitch: Double
    public var field: FillField
    /// How many weights a stroke can have. One: the stroke is drawn at the
    /// layer's width where the field is positive and nowhere else -- a hatch
    /// clipped to the pattern. More: the field's height in `(0, 1]` is cut
    /// into that many steps and the stroke's width follows it, `k / levels`
    /// of the layer's width on step `k`, thinning to nothing at the
    /// pattern's edge.
    public var levels: Int

    public init(angle: Double, spacing: Double, pitch: Double? = nil,
                field: FillField = .solid, levels: Int = 1) {
        precondition(spacing > 0 && spacing.isFinite, "a fill has a positive spacing")
        precondition(levels >= 1, "a fill has at least one level")
        self.angle = angle; self.spacing = spacing; self.pitch = pitch ?? spacing
        self.field = field; self.levels = levels
    }

    /// The step a weight falls on: 0 for none, `levels` for the full width.
    public func level(of weight: Double) -> Int {
        guard weight > 0 else { return 0 }
        return min(levels, Int((weight * Double(levels)).rounded(.up)))
    }

    // MARK: manifest

    public init(json: JSONValue) throws {
        let o = try json.object("LayerSource.fill")
        let field: FillField
        let f = try o.value("field", "LayerSource.fill").object("LayerSource.fill.field")
        switch try f.string("kind", "LayerSource.fill.field") {
        case "solid":
            field = .solid
        case "turing":
            field = .turing(scale: try f.double("scale", "LayerSource.fill.field"),
                            seed: UInt64(bitPattern: Int64(try f.int("seed", "LayerSource.fill.field"))))
        case "shade":
            let l = try f.doubles("light", "LayerSource.fill.field")
            guard l.count == 3 else {
                throw JSONError.wrongType("light", expected: "three numbers", found: "LayerSource.fill.field")
            }
            field = .shade(light: SIMD3(l[0], l[1], l[2]))
        case let other:
            throw ManifestError.unknownKind(other, of: "FillField", known: ["solid", "turing", "shade"])
        }
        let spacing = try o.double("spacing", "LayerSource.fill")
        let levels = try o.int("levels", "LayerSource.fill")
        guard spacing > 0, spacing.isFinite else {
            throw JSONError.wrongType("spacing", expected: "a positive number", found: "LayerSource.fill")
        }
        guard levels >= 1 else {
            throw JSONError.wrongType("levels", expected: "a positive integer", found: "LayerSource.fill")
        }
        self.init(angle: try o.double("angle", "LayerSource.fill"), spacing: spacing,
                  pitch: o.optionalDouble("pitch"), field: field, levels: levels)
    }

    public var json: JSONValue {
        let field: JSONValue
        switch self.field {
        case .solid:
            field = .object(["kind": .string("solid")])
        case .turing(let scale, let seed):
            field = .object(["kind": .string("turing"), "scale": .double(scale),
                             "seed": .int(Int(Int64(bitPattern: seed)))])
        case .shade(let light):
            field = .object(["kind": .string("shade"),
                             "light": .array([.double(light.x), .double(light.y), .double(light.z)])])
        }
        return .object(["kind": .string("fill"), "angle": .double(angle), "spacing": .double(spacing),
                        "pitch": .double(pitch), "levels": .int(levels), "field": field])
    }
}

/// What a fill reads its weight from, as a function on the facet's plane
/// with values in `[-1, 1]`; ink exists where it is positive.
public enum FillField: Sendable, Equatable {
    /// One everywhere: the whole facet, hatched.
    case solid
    /// Turing's dappling, as the 2022 post drew it: a sum of unit Gaussians
    /// on a lattice of spacing 0.7, each with a random sign, in facet
    /// coordinates divided by `scale`. The signs are a hash of the lattice
    /// point and `seed`, so the field is a function on the whole plane with
    /// no edge and no array behind it. Normalized by twice the sum's
    /// standard deviation, so it mostly lies within `[-1, 1]`, and clamped
    /// there.
    case turing(scale: Double, seed: UInt64)
    /// Flat shading by a light: `(1 - n·l) / 2` for the facet's outward
    /// normal `n` and the unit direction `l` toward the light. Nothing on a
    /// facet square to the light, full weight on one turned from it; one
    /// weight per facet.
    case shade(light: SIMD3<Double>)

    /// The field on one facet, in that facet's coordinates.
    public func weight(on facet: Solid.Facet) -> @Sendable (SIMD2<Double>) -> Double {
        switch self {
        case .solid:
            return { _ in 1 }
        case .turing(let scale, let seed):
            return { c in Turing.value(c / scale, seed: seed) }
        case .shade(let light):
            let w = (1 - simd_dot(facet.normal, simd_normalize(light))) / 2
            return { _ in w }
        }
    }
}

/// Turing's closed-form dappling (1952, §10 of the morphogenesis paper):
/// unit Gaussians on a square lattice with random signs.
public enum Turing {
    /// The lattice spacing, Turing's own.
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

    /// The field: the sum over twice its deviation, clamped to `[-1, 1]`.
    public static func value(_ p: SIMD2<Double>, seed: UInt64) -> Double {
        min(max(sum(p, seed: seed) / (2 * deviation), -1), 1)
    }
}

// MARK: - ruling a solid

public extension Solid {
    /// One stroke of a fill: which facet it lies in, its weight, and its
    /// path in world space.
    struct Ruling: Sendable {
        public let facet: Int
        public let level: Int
        public let path: [P3<WorldSpace>]
    }

    /// The fill ruled over every facet, unjudged.
    func rulings(_ fill: Fill) -> [Ruling] {
        facets.indices.flatMap { rulings(fill, facet: $0) }
    }

    /// Every ruling's path, unjudged: the ink a fill layer carries before
    /// the camera is known.
    func fillPaths(_ fill: Fill) -> PolylineSet<WorldSpace> {
        PolylineSet(paths: rulings(fill).map(\.path))
    }

    /// The fill ruled over facet `i`.
    ///
    /// The facet's plane is ruled by lines at the fill's angle, one every
    /// spacing, through the frame's origin and its multiples, so a facet's
    /// ruling is fixed by its own frame and not by which of its faces is
    /// asked. Each line is clipped to every face of the facet -- a convex
    /// polygon, so an interval -- and the intervals are joined where they
    /// meet, which is across every soft edge: the two faces compute the
    /// seam's crossing from the same two vertices and agree to the bit. The
    /// field is then read along each run, every pitch, and where its level
    /// changes between two readings the change is bisected on the field.
    func rulings(_ fill: Fill, facet i: Int) -> [Ruling] {
        let facet = facet(i)
        let polygons: [[SIMD2<Double>]] = facet.faces.map { f in
            faces[f].map { facet.coordinates(vertices[$0].v) }
        }
        let dir = SIMD2(cos(fill.angle), sin(fill.angle))
        let per = SIMD2(-dir.y, dir.x)
        let weight = fill.field.weight(on: facet)
        func cross(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.y - a.y * b.x }
        var lo = Double.infinity, hi = -Double.infinity
        for poly in polygons { for p in poly {
            let c = simd_dot(p, per)
            lo = min(lo, c); hi = max(hi, c)
        } }
        guard lo.isFinite, hi.isFinite, hi > lo else { return [] }
        let extent = hi - lo
        var out: [Ruling] = []
        let kLo = Int((lo / fill.spacing).rounded(.up)), kHi = Int((hi / fill.spacing).rounded(.down))
        guard kLo <= kHi else { return [] }
        for k in kLo...kHi {
            let c = Double(k) * fill.spacing
            let base = c * per
            // The line's overlap with each face: half-plane by half-plane.
            var intervals: [(Double, Double)] = []
            for poly in polygons {
                var t0 = -Double.infinity, t1 = Double.infinity
                var empty = false
                for j in 0..<poly.count {
                    let p = poly[j], q = poly[(j + 1) % poly.count]
                    let e = q - p
                    let f0 = cross(e, base - p), f1 = cross(e, dir)
                    if abs(f1) <= 1e-15 * extent {
                        if f0 < 0 { empty = true; break }
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
                func level(_ t: Double) -> Int { fill.level(of: weight(base + t * dir)) }
                for (ta, tb, lv) in Solid.graded(from: a, to: b, pitch: fill.pitch, level: level)
                    where lv > 0 {
                    out.append(Ruling(facet: i, level: lv,
                                      path: [P3(facet.point(base + ta * dir)),
                                             P3(facet.point(base + tb * dir))]))
                }
            }
        }
        return out
    }

    /// `[a, b]` cut into pieces of one level each: the level read every
    /// `pitch`, and every change between two readings bisected on the level
    /// itself, to the last bit of the parameter. A change that reverses
    /// between two readings is not seen; that is what the pitch means.
    static func graded(from a: Double, to b: Double, pitch: Double,
                       level: (Double) -> Int) -> [(Double, Double, Int)] {
        let length = b - a
        let steps = pitch > 0 && pitch.isFinite ? max(1, Int((length / pitch).rounded(.up))) : 1
        var pieces: [(Double, Double, Int)] = []
        var start = a
        var current = level(a)
        // `known` is the last point whose level is `current`.
        var known = a
        let tolerance = 1e-13 * max(length, 1)
        for j in 1...steps {
            let t = j == steps ? b : a + length * Double(j) / Double(steps)
            // Between two readings of different levels: the first change,
            // then whatever follows it, until the reading's level is reached.
            while level(t) != current {
                var lo = known, hi = t
                while hi - lo > tolerance {
                    let mid = 0.5 * (lo + hi)
                    if level(mid) == current { lo = mid } else { hi = mid }
                }
                pieces.append((start, hi, current))
                start = hi; known = hi; current = level(hi)
            }
            known = t
        }
        pieces.append((start, b, current))
        // Neighbours of one level are one piece.
        var merged: [(Double, Double, Int)] = []
        for p in pieces where p.1 > p.0 {
            if let last = merged.last, last.2 == p.2, last.1 == p.0 {
                merged[merged.count - 1].1 = p.1
            } else {
                merged.append(p)
            }
        }
        return merged
    }

    /// The fill judged for a camera: per level, the rulings as carried and
    /// as seen, so a layer per level can be drawn at its width.
    func visibleFill(_ fill: Fill, view: Transform<WorldSpace, ViewSpace>)
        -> [(level: Int, unjudged: PolylineSet<WorldSpace>, judged: PolylineSet<WorldSpace>)] {
        let cast = self.cast(view: view)
        let all = rulings(fill)
        var out: [(level: Int, unjudged: PolylineSet<WorldSpace>, judged: PolylineSet<WorldSpace>)] = []
        for level in 1...fill.levels {
            let mine = all.filter { $0.level == level }
            guard !mine.isEmpty else { continue }
            var judged: [[P3<WorldSpace>]] = []
            for i in facets.indices {
                let paths = mine.filter { $0.facet == i }.map(\.path)
                guard !paths.isEmpty else { continue }
                judged += visibleInk(paths, inFacet: i, cast: cast)
            }
            out.append((level, PolylineSet(paths: mine.map(\.path)), PolylineSet(paths: judged)))
        }
        return out
    }
}
