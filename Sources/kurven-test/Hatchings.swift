import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake

// MARK: - fills
//
// A fill is ink in a face: ruled over the facet in closed form, ended on the
// field's own level by bisection, and judged by the same cast as the edges.
// So it is held to the same oracle -- a point-by-point ray cast -- and to
// the same property: nothing in it has a pixel, so the bake at 64 is the
// bake at 4096 to the bit.

/// Whether `p`, in the plane of facet `f` of `s`, lies in one of its faces.
func inFacet(_ s: Solid, _ f: Int, _ p: SIMD3<Double>) -> Bool {
    let facet = s.facet(f)
    guard abs(simd_dot(p - facet.origin, facet.normal)) < 1e-9 else { return false }
    for face in facet.faces {
        let poly = s.faces[face]
        var inside = true
        for k in 0..<poly.count {
            let a = s.vertices[poly[k]].v, b = s.vertices[poly[(k + 1) % poly.count]].v
            if simd_dot(simd_cross(b - a, p - a), facet.normal) < -1e-9 { inside = false; break }
        }
        if inside { return true }
    }
    return false
}

/// Whether `p` lies on an edge of one of facet `f`'s faces.
func onBoundary(_ s: Solid, _ f: Int, _ p: SIMD3<Double>) -> Bool {
    for face in s.facets[f] {
        let poly = s.faces[face]
        for k in 0..<poly.count {
            let a = s.vertices[poly[k]].v, b = s.vertices[poly[(k + 1) % poly.count]].v
            let t = simd_dot(p - a, b - a) / simd_length_squared(b - a)
            if t >= -1e-9, t <= 1 + 1e-9, simd_length(p - (a + t * (b - a))) < 1e-6 { return true }
        }
    }
    return false
}

func fillTests() {
    Check.suite("fills: Turing's field is a function of the seed, mean zero, half positive") {
        let p = SIMD2(3.1, -2.7)
        Check.expect(Turing.value(p, seed: 7) == Turing.value(p, seed: 7)
                     && Turing.value(p, seed: 7) != Turing.value(p, seed: 8),
                     "the same point under the same seed is the same value, under another seed another")
        var total = 0.0, positive = 0, n = 0
        var lo = 1.0, hi = -1.0
        for i in 0..<120 { for j in 0..<120 {
            let v = Turing.value(SIMD2(Double(i) * 0.5, Double(j) * 0.5), seed: 3)
            total += v; n += 1
            if v > 0 { positive += 1 }
            lo = min(lo, v); hi = max(hi, v)
        } }
        let mean = total / Double(n), share = Double(positive) / Double(n)
        Check.expect(abs(mean) < 0.05 && abs(share - 0.5) < 0.05 && lo < -0.5 && hi > 0.5,
                     "over a 60 by 60 window the mean is near zero and half the field is positive",
                     String(format: "mean %.3f, %.0f%% positive, range [%.2f, %.2f]", mean, share * 100, lo, hi))
    }

    Check.suite("fills: a ruling lies in its facet, runs unbroken across a seam, and ends on the field's own level") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        let plain = Fill(angle: 0.6, spacing: 0.07)
        let rulings = s.rulings(plain)
        var inPlane = true, crossSeam = 0, endsOnSeam = 0
        // The front annulus: facet of the first face, which is a front
        // trapezoid; its seams run from each outer top corner to the inner.
        let front = s.facetIndex(of: 0)
        let seams: [(SIMD3<Double>, SIMD3<Double>)] = (0..<4).map { (s.vertices[8 + $0].v, s.vertices[12 + $0].v) }
        func onSeam(_ p: SIMD3<Double>) -> Bool {
            seams.contains { a, b in
                let t = simd_dot(p - a, b - a) / simd_length_squared(b - a)
                return t > 1e-6 && t < 1 - 1e-6 && simd_length(p - (a + t * (b - a))) < 1e-9
            }
        }
        for r in rulings {
            for v in r.path where !inFacet(s, r.facet, v.v) { inPlane = false }
            guard r.facet == front, r.path.count == 2 else { continue }
            for v in r.path where onSeam(v.v) { endsOnSeam += 1 }
            // Crosses a seam: the seam's ends lie on either side of the
            // ruling's line and the crossing is within the ruling.
            let a = r.path[0].v, b = r.path[1].v
            for (p, q) in seams {
                let n = simd_cross(b - a, s.normals[0])
                let fp = simd_dot(p - a, n), fq = simd_dot(q - a, n)
                guard (fp > 0) != (fq > 0) else { continue }
                let x = p + (q - p) * (fp / (fp - fq))
                let t = simd_dot(x - a, b - a) / simd_length_squared(b - a)
                if t > 1e-6, t < 1 - 1e-6 { crossSeam += 1 }
            }
        }
        Check.expect(inPlane && rulings.count > 100, "every ruling lies in its facet, within its faces",
                     "\(rulings.count) rulings")
        Check.expect(crossSeam > 0 && endsOnSeam == 0,
                     "on the front annulus rulings cross the seams between its trapezoids whole, and none ends on one",
                     "\(crossSeam) crossings, \(endsOnSeam) ends on a seam")

        // Clipped to Turing's field: an end not on the facet's boundary is
        // a zero of the field, to the last bit. Graded: on a level.
        let cube = Solid.platonic(.cube, circumradius: 2).resting(on: 0)
        for levels in [1, 4] {
            let fill = Fill(angle: 0.3, spacing: 0.03, field: .turing(scale: 0.6, seed: 11), levels: levels)
            var interior = 0, offLevel = 0, badLevel = 0
            for r in cube.rulings(fill) {
                if r.level < 1 || r.level > levels { badLevel += 1 }
                let facet = cube.facet(r.facet)
                let weight = fill.field.weight(on: facet)
                for v in r.path {
                    guard !onBoundary(cube, r.facet, v.v) else { continue }
                    interior += 1
                    let w = weight(facet.coordinates(v.v))
                    // The end is where the level steps: at a threshold
                    // k / levels, within the bisection's resolution.
                    let steps = (0...levels).map { Double($0) / Double(levels) }
                    if !steps.contains(where: { abs(w - $0) < 1e-7 }) { offLevel += 1 }
                }
            }
            Check.expect(interior > 50 && offLevel == 0 && badLevel == 0,
                         levels == 1 ? "clipped to Turing's field, every interior end is a zero of it"
                                     : "graded into \(levels), every interior end lies on a step and every level is in range",
                         "\(interior) interior ends, \(offLevel) off a step, \(badLevel) bad levels")
        }
    }

    Check.suite("fills: judged by the cast, a fill agrees with a ray cast, through the frame's hole included") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        let fill = Fill(angle: 0.5, spacing: 0.08)
        var compared = 0, wrong = 0, cameras = 0, cut = 0
        var example = ""
        for (name, view) in stressCameras() {
            cameras += 1
            let d = -view.sightLine
            let seen = s.visibleFill(fill, view: view)
            guard seen.count == 1 else { Check.expect(false, "one level"); return }
            let judged = seen[0].judged
            // Every judged piece as (facet, a, b).
            var pieces: [(Int, SIMD3<Double>, SIMD3<Double>)] = []
            for i in 0..<judged.count {
                let p = judged[path: i]
                for (a, b) in zip(p, p.dropFirst()) {
                    guard let f = s.facets.indices.first(where: { inFacet(s, $0, a.v) && inFacet(s, $0, b.v) }) else {
                        wrong += 1; continue
                    }
                    pieces.append((f, a.v, b.v))
                }
            }
            for r in s.rulings(fill) {
                // A facet seen exactly edge-on projects to a line, and whether
                // its ink is drawn is rounding: set aside, as for the edges.
                guard abs(simd_dot(s.normals[s.facets[r.facet][0]], d)) > 1e-9 else { continue }
                let a = r.path[0].v, b = r.path[1].v
                let own = Set(s.facets[r.facet])
                var runs: [(Double, Double)] = []
                for (f, p, q) in pieces where f == r.facet {
                    let len2 = simd_length_squared(b - a)
                    let t0 = simd_dot(p - a, b - a) / len2, t1 = simd_dot(q - a, b - a) / len2
                    if simd_length(p - (a + t0 * (b - a))) < 1e-9, simd_length(q - (a + t1 * (b - a))) < 1e-9,
                       t0 >= -1e-9, t1 <= 1 + 1e-9 {
                        runs.append((t0, t1))
                        if t0 > 1e-9 || t1 < 1 - 1e-9 { cut += 1 }
                    }
                }
                for k in 0..<20 {
                    let t = (Double(k) + 0.5) / 20
                    if runs.contains(where: { abs(t - $0.0) < 1e-6 || abs(t - $0.1) < 1e-6 }) { continue }
                    let p = a + t * (b - a)
                    let seenByCast = runs.contains { t > $0.0 && t < $0.1 }
                    let truth = rayVisible(s, at: p, own: own, toEye: d)
                    compared += 1
                    if truth != seenByCast {
                        wrong += 1
                        if example.isEmpty {
                            example = String(format: "e.g. %@ facet %d t=%.3f truth %@", name, r.facet, t,
                                             truth ? "seen" : "hidden")
                        }
                    }
                }
            }
        }
        Check.expect(cut > 0, "under \(cameras) cameras some strokes are drawn in part: the front hides the walls in the hole",
                     "\(cut) pieces are parts of strokes")
        Check.expect(wrong == 0 && compared > 20_000,
                     "and every stroke agrees with a point-by-point ray cast against the faces",
                     "\(wrong) of \(compared) sampled points disagree \(example)")
    }

    Check.suite("fills: on a convex solid facing is the whole test, and the bake is the same at every resolution") {
        let s = Solid.platonic(.dodecahedron, circumradius: 2).resting(on: 0)
        // A face is about 1.4 across; at this scale that is a dozen lattice
        // spacings, enough spots for every weight to occur on it. At half a
        // unit a face sits inside a single spot.
        let fill = Fill(angle: 1.1, spacing: 0.05, field: .turing(scale: 0.12, seed: 5), levels: 3)
        let camera = Camera.plate(PlateProjection(shear: 0.3, xAngle: -55, zAngle: -70, flipX: false, yScale: nil))
        let d = -camera.view.sightLine
        var wholeOrAbsent = true, levelsSeen = Set<Int>()
        let rulings = s.rulings(fill)
        for (level, unjudged, judged) in s.visibleFill(fill, view: camera.view) {
            levelsSeen.insert(level)
            // Each facet's rulings are all kept, whole and to the bit, or all
            // dropped, by its facing.
            let kept = Set(judged.paths.map { Array($0) })
            for r in rulings where r.level == level {
                let faces = simd_dot(s.normals[s.facets[r.facet][0]], d) > 0
                if kept.contains(r.path) != faces { wholeOrAbsent = false }
            }
            if judged.count > unjudged.count { wholeOrAbsent = false }
        }
        Check.expect(wholeOrAbsent && levelsSeen == [1, 2, 3],
                     "every stroke of a facing face is drawn whole and every other dropped, at all three weights",
                     "levels \(levelsSeen.sorted())")
        let layers = [
            Layer(spec: LayerSpec(name: "edges", role: .outline, source: .edges, width: 0.5, heightPolicy: .surface),
                  paths: s.edgePaths),
            Layer(spec: LayerSpec(name: "fill", role: .scaffold, source: .fill(fill), width: 0.3, heightPolicy: .surface),
                  paths: s.fillPaths(fill)),
        ]
        let scene = Scene(solid: s, layers: layers, camera: camera)
        let renderer = try MetalRenderer()
        let coarse = try renderer.bake(scene, options: BakeOptions(resolution: 64))
        let fine = try renderer.bake(scene, options: BakeOptions(resolution: 2048))
        let same = coarse.strokes.layers.count == fine.strokes.layers.count
            && zip(coarse.strokes.layers, fine.strokes.layers).allSatisfy {
                $0.paths == $1.paths && $0.style == $1.style
            }
        let widths = coarse.strokes.layers.map(\.style.width)
        let expected = [0.5, 0.1, 0.2, 0.3]
        let widthsRight = widths.count == expected.count
            && zip(widths, expected).allSatisfy { abs($0 - $1) < 1e-12 }
        Check.expect(same && coarse.strokes.layers.count == 4 && widthsRight,
                     "the bake has the edges and a layer per weight at a third, two thirds and the width, the same at 64 px and 2048 px",
                     "\(coarse.strokes.layers.count) layers, widths \(widths)")
    }

    Check.suite("fills: a fill layer round-trips through the manifest") {
        let fills = [
            Fill(angle: 0.4, spacing: 0.1),
            Fill(angle: -1.2, spacing: 0.05, pitch: 0.01, field: .turing(scale: 0.7, seed: 0xDEAD_BEEF_0000_0001), levels: 5),
            Fill(angle: 0, spacing: 0.2, field: .shade(light: SIMD3(0.1, -0.3, 0.9)), levels: 2),
        ]
        for fill in fills {
            let source = LayerSource.fill(fill)
            let back = try LayerSource(json: source.json)
            Check.expect(back == source, "\(source.json.canonical) decodes to itself")
        }
    }
}
