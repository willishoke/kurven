import Foundation
import simd
import KurvenCore
import KurvenMetal
import KurvenBake

// MARK: - hatching
//
// Hatching on a solid is a set of curves on its faces: laid over each face in
// closed form, ended on the tone's own level set by bisection, and tested for
// visibility by the same occlusion test as the edges. So it is held to the
// same oracle -- a point-by-point ray cast -- and to the same property:
// nothing in it has a pixel, so the bake at 64 is the bake at 4096 to the bit.

/// Whether `p`, in the plane of face `f` of `s`, lies in one of its pieces.
func onFace(_ s: Solid, _ f: Int, _ p: SIMD3<Double>) -> Bool {
    let face = s.face(f)
    guard abs(simd_dot(p - face.origin, face.normal)) < 1e-9 else { return false }
    for piece in face.pieces {
        let poly = s.pieces[piece]
        var inside = true
        for k in 0..<poly.count {
            let a = s.vertices[poly[k]].v, b = s.vertices[poly[(k + 1) % poly.count]].v
            if simd_dot(simd_cross(b - a, p - a), face.normal) < -1e-9 { inside = false; break }
        }
        if inside { return true }
    }
    return false
}

/// Whether `p` lies on a side of one of face `f`'s pieces.
func onPieceBoundary(_ s: Solid, _ f: Int, _ p: SIMD3<Double>) -> Bool {
    for piece in s.faces[f] {
        let poly = s.pieces[piece]
        for k in 0..<poly.count {
            let a = s.vertices[poly[k]].v, b = s.vertices[poly[(k + 1) % poly.count]].v
            let t = simd_dot(p - a, b - a) / simd_length_squared(b - a)
            if t >= -1e-9, t <= 1 + 1e-9, simd_length(p - (a + t * (b - a))) < 1e-6 { return true }
        }
    }
    return false
}

func hatchingTests() {
    Check.suite("hatching: Turing's pattern is a function of the seed, mean zero, half positive") {
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
                     "over a 60 by 60 window the mean is near zero and half the pattern is positive",
                     String(format: "mean %.3f, %.0f%% positive, range [%.2f, %.2f]", mean, share * 100, lo, hi))
    }

    Check.suite("hatching: a hatch line lies on its face, runs unbroken across a diagonal, and ends on the tone's own level set") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        let plain = Hatching(angle: 0.6, spacing: 0.07)
        let lines = s.hatchLines(plain)
        var inPlane = true, crossDiagonal = 0, endsOnDiagonal = 0
        // The front annulus: the face of piece 0, a front trapezoid. Its
        // diagonals run from each outer top corner to the inner one.
        let front = s.faceIndex(ofPiece: 0)
        let diagonals: [(SIMD3<Double>, SIMD3<Double>)] = (0..<4).map { (s.vertices[8 + $0].v, s.vertices[12 + $0].v) }
        func onDiagonal(_ p: SIMD3<Double>) -> Bool {
            diagonals.contains { a, b in
                let t = simd_dot(p - a, b - a) / simd_length_squared(b - a)
                return t > 1e-6 && t < 1 - 1e-6 && simd_length(p - (a + t * (b - a))) < 1e-9
            }
        }
        for line in lines {
            for v in line.path where !onFace(s, line.face, v.v) { inPlane = false }
            guard line.face == front, line.path.count == 2 else { continue }
            for v in line.path where onDiagonal(v.v) { endsOnDiagonal += 1 }
            // Crosses a diagonal: the diagonal's ends lie on either side of
            // the hatch line and the crossing is within it.
            let a = line.path[0].v, b = line.path[1].v
            for (p, q) in diagonals {
                let n = simd_cross(b - a, s.normals[0])
                let fp = simd_dot(p - a, n), fq = simd_dot(q - a, n)
                guard (fp > 0) != (fq > 0) else { continue }
                let x = p + (q - p) * (fp / (fp - fq))
                let t = simd_dot(x - a, b - a) / simd_length_squared(b - a)
                if t > 1e-6, t < 1 - 1e-6 { crossDiagonal += 1 }
            }
        }
        Check.expect(inPlane && lines.count > 100, "every hatch line lies on its face, within its pieces",
                     "\(lines.count) hatch lines")
        Check.expect(crossDiagonal > 0 && endsOnDiagonal == 0,
                     "on the front annulus hatch lines cross the diagonals between its trapezoids whole, and none ends on one",
                     "\(crossDiagonal) crossings, \(endsOnDiagonal) ends on a diagonal")

        // With Turing's tone and one width, an end away from the pieces'
        // sides is a zero of the tone, to the last bit. With several widths
        // it lies on a step.
        let cube = Solid.platonic(.cube, circumradius: 2).resting(onFace: 0)
        for widthSteps in [1, 4] {
            let hatching = Hatching(angle: 0.3, spacing: 0.03, tone: .turing(scale: 0.6, seed: 11),
                                    widthSteps: widthSteps)
            var interior = 0, offStep = 0, outOfRange = 0
            for line in cube.hatchLines(hatching) {
                if line.widthStep < 1 || line.widthStep > widthSteps { outOfRange += 1 }
                let face = cube.face(line.face)
                let tone = hatching.tone.value(on: face)
                for v in line.path {
                    guard !onPieceBoundary(cube, line.face, v.v) else { continue }
                    interior += 1
                    let w = tone(face.coordinates(v.v))
                    // The end is where the width steps: at a threshold
                    // k / widthSteps, within the bisection's resolution.
                    let steps = (0...widthSteps).map { Double($0) / Double(widthSteps) }
                    if !steps.contains(where: { abs(w - $0) < 1e-7 }) { offStep += 1 }
                }
            }
            Check.expect(interior > 50 && offStep == 0 && outOfRange == 0,
                         widthSteps == 1 ? "with one width, every interior end is a zero of Turing's tone"
                                         : "with \(widthSteps) widths, every interior end lies on a step and every width is in range",
                         "\(interior) interior ends, \(offStep) off a step, \(outOfRange) out of range")
        }
    }

    Check.suite("hatching: tested by occlusion, hatching agrees with a ray cast, through the frame's hole included") {
        let s = Solid.frame(outer: 4, inner: 2, depth: 1)
        let hatching = Hatching(angle: 0.5, spacing: 0.08)
        var compared = 0, wrong = 0, cameras = 0, cut = 0
        var example = ""
        for (name, view) in stressCameras() {
            cameras += 1
            let d = -view.sightLine
            let seen = s.visibleHatching(hatching, view: view)
            guard seen.count == 1 else { Check.expect(false, "one width"); return }
            let visible = seen[0].visible
            // Every visible segment as (face, a, b).
            var segments: [(Int, SIMD3<Double>, SIMD3<Double>)] = []
            for i in 0..<visible.count {
                let p = visible[path: i]
                for (a, b) in zip(p, p.dropFirst()) {
                    guard let f = s.faces.indices.first(where: { onFace(s, $0, a.v) && onFace(s, $0, b.v) }) else {
                        wrong += 1; continue
                    }
                    segments.append((f, a.v, b.v))
                }
            }
            for line in s.hatchLines(hatching) {
                // A face seen exactly edge-on projects to a line, and whether
                // its hatching is drawn is rounding: set aside, as for the edges.
                guard abs(simd_dot(s.normals[s.faces[line.face][0]], d)) > 1e-9 else { continue }
                let a = line.path[0].v, b = line.path[1].v
                let own = Set(s.faces[line.face])
                var runs: [(Double, Double)] = []
                for (f, p, q) in segments where f == line.face {
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
                    let seenByOcclusion = runs.contains { t > $0.0 && t < $0.1 }
                    let truth = rayVisible(s, at: p, own: own, toEye: d)
                    compared += 1
                    if truth != seenByOcclusion {
                        wrong += 1
                        if example.isEmpty {
                            example = String(format: "e.g. %@ face %d t=%.3f truth %@", name, line.face, t,
                                             truth ? "seen" : "hidden")
                        }
                    }
                }
            }
        }
        Check.expect(cut > 0, "under \(cameras) cameras some hatch lines are drawn in part: the front hides the walls in the hole",
                     "\(cut) spans are parts of lines")
        Check.expect(wrong == 0 && compared > 20_000,
                     "and every hatch line agrees with a point-by-point ray cast against the pieces",
                     "\(wrong) of \(compared) sampled points disagree \(example)")
    }

    Check.suite("hatching: on a convex solid facing is the whole test, and the bake is the same at every resolution") {
        let s = Solid.platonic(.dodecahedron, circumradius: 2).resting(onFace: 0)
        // A face is about 1.4 across; at this scale that is a dozen Turing
        // lattice spacings, enough spots for every width to occur on it. At
        // half a unit a face sits inside a single spot.
        let hatching = Hatching(angle: 1.1, spacing: 0.05, tone: .turing(scale: 0.12, seed: 5), widthSteps: 3)
        let camera = Camera.plate(PlateProjection(shear: 0.3, xAngle: -55, zAngle: -70, flipX: false, yScale: nil))
        let d = -camera.view.sightLine
        var wholeOrAbsent = true, widthsSeen = Set<Int>()
        let lines = s.hatchLines(hatching)
        for (w, all, visible) in s.visibleHatching(hatching, view: camera.view) {
            widthsSeen.insert(w)
            // Each face's hatch lines are all kept, whole and to the bit, or
            // all dropped, by its facing.
            let kept = Set(visible.paths.map { Array($0) })
            for line in lines where line.widthStep == w {
                let facing = simd_dot(s.normals[s.faces[line.face][0]], d) > 0
                if kept.contains(line.path) != facing { wholeOrAbsent = false }
            }
            if visible.count > all.count { wholeOrAbsent = false }
        }
        Check.expect(wholeOrAbsent && widthsSeen == [1, 2, 3],
                     "every hatch line on a facing face is drawn whole and every other dropped, at all three widths",
                     "widths \(widthsSeen.sorted())")
        let layers = [
            Layer(spec: LayerSpec(name: "edges", role: .outline, source: .edges, width: 0.5, heightPolicy: .surface),
                  paths: s.edgePaths),
            Layer(spec: LayerSpec(name: "hatching", role: .scaffold, source: .hatching(hatching), width: 0.3,
                                  heightPolicy: .surface),
                  paths: s.hatchPaths(hatching)),
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
                     "the bake has the edges and a layer per width step at a third, two thirds and the width, the same at 64 px and 2048 px",
                     "\(coarse.strokes.layers.count) layers, widths \(widths)")
    }

    Check.suite("hatching: a hatching layer round-trips through the manifest") {
        let hatchings = [
            Hatching(angle: 0.4, spacing: 0.1),
            Hatching(angle: -1.2, spacing: 0.05, step: 0.01, tone: .turing(scale: 0.7, seed: 0xDEAD_BEEF_0000_0001),
                     widthSteps: 5),
            Hatching(angle: 0, spacing: 0.2, tone: .lambert(light: SIMD3(0.1, -0.3, 0.9)), widthSteps: 2),
        ]
        for hatching in hatchings {
            let source = LayerSource.hatching(hatching)
            let back = try LayerSource(json: source.json)
            Check.expect(back == source, "\(source.json.canonical) decodes to itself")
        }
    }
}
