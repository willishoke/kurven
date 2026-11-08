import Foundation
import KurvenCore
import KurvenLandscape
import KurvenDynamics

// MARK: - painting a parametric surface
//
// What the app's "Curves on" section does to a surface's request and
// layers, without the window: a function painted on the plain torus, and a
// request edit that carries a field's contours over by its class
// (`FieldClass`). The window's own chain -- pick, paint, bake, undo -- is
// held to the CLI by `scripts/check.sh`.

func paintTests() {
    Check.suite("paint: sn painted on the plain torus over its period rectangle draws surface sn") {
        let torus = SurfacePreset.named("torus")!.request
        let sn = SurfacePreset.named("sn")!.request
        guard case .sn(let m, _, _, _) = sn.shape, let periods = sn.periods else {
            Check.expect(false, "the catalog's sn has a modulus and a period rectangle")
            return
        }
        var painted = try torus.painting("sn(z, \(m))", periods: periods)
        painted.lines = nil
        let built = try SurfaceDrawing.build(sn)
        let families = PeriodicTorus.contours(ceiling: 3)
        let drawn = try SurfaceDrawing.build(painted).restyled(families + built.layers.map(\.spec).filter {
            $0.source == .foldCurves
        })
        Check.expect(drawn.layers.map(\.spec) == built.layers.map(\.spec)
                     && drawn.layers.map(\.paths) == built.layers.map(\.paths)
                     && drawn.surface.positions == built.surface.positions,
                     "its four families at the catalog's levels, and the folds, are surface sn's, curve for curve",
                     "\(built.layers.map(\.paths.count))")
        Check.expect(painted.shape == .periodic(expression: "sn(z, 0.64)", periods: periods, major: 2, minor: 1,
                                                grid: 600) && painted.grid == torus.grid && painted.name.isEmpty,
                     "the torus keeps its radii and grid, and f is sampled as the catalog's sn is")

        func refusal(_ request: SurfaceRequest, _ expression: String, _ p: SIMD2<Double>) -> String? {
            do { _ = try request.painting(expression, periods: p); return nil } catch { return "\(error)" }
        }
        let forced = SurfacePreset.named("forced")!.request
        Check.expect(refusal(torus, "sn(z,", periods)?.contains("stops in the middle") == true
                     && refusal(torus, "frob(z)", periods)?.contains("no 'frob'") == true
                     && refusal(torus, "  ", periods) == "the expression is empty"
                     && refusal(torus, "sn(z, 0.64)", SIMD2(-1, 2))?.contains("a period must be a positive") == true
                     && refusal(torus, "sn(z, 0.64)", SIMD2(1, .infinity)) != nil
                     && refusal(forced, "sn(z, 0.64)", periods)?.contains("forced torus") == true,
                     "an expression that does not parse, a period that is not positive, a forced torus: refused, "
                     + "saying why")
    }

    Check.suite("paint: a request edit carries a coordinate's contours over, and contours the height again") {
        // The plain torus with a contour of each class of field the plain
        // routine draws: its rectangle keeps its samples when the radii move.
        let request = SurfaceRequest(.torus(major: 2, minor: 1), grid: 128, lines: SIMD2(12, 6), name: "torus")
        func contour(_ field: ContourField, _ levels: [Double]) -> LayerSpec {
            LayerSpec(name: field.rawValue, role: .scaffold,
                      source: .contour(field: field, levels: levels, restriction: .all, faces: .surface),
                      width: 0.2, heightPolicy: .surface)
        }
        let first = try SurfaceDrawing.build(request)
        let kept = first.layers.map(\.spec) + [contour(.u, [0.5, 2]), contour(.v, [1, 4]), contour(.height, [0.25])]
        let drawn = try first.restyled(kept)
        var wider = request
        wider[.major] = 2.5
        wider[.minor] = 1.25
        let edited = try SurfaceDrawing.build(wider, reusing: drawn).keeping(kept, from: drawn)
        let fresh = try SurfaceDrawing.build(wider).restyled(kept)
        func paths(_ d: SurfaceDrawing, _ name: String) -> PolylineSet<WorldSpace>? {
            d.layers.first { $0.spec.name == name }?.paths
        }
        Check.expect(edited.layers.map(\.spec) == fresh.layers.map(\.spec)
                     && edited.layers.map(\.paths) == fresh.layers.map(\.paths),
                     "every layer is what the edited request draws afresh, curve for curve")
        Check.expect(paths(edited, "u")?.coords == paths(drawn, "u")?.coords
                     && paths(edited, "u")?.vertices != paths(drawn, "u")?.vertices
                     && paths(edited, "height")?.coords != paths(drawn, "height")?.coords,
                     "u keeps its parameter coordinates and moves in space; the height's level set is a new one")
        var fewer = request
        fewer.lines = SIMD2(6, 3)
        let restyledOnly = try SurfaceDrawing.build(fewer, reusing: drawn).keeping(kept, from: drawn)
        let fewerFresh = try SurfaceDrawing.build(fewer).keeping(kept)
        Check.expect(restyledOnly.content == drawn.content
                     && paths(restyledOnly, "height") == paths(drawn, "height")
                     && restyledOnly.layers.map(\.paths) == fewerFresh.layers.map(\.paths),
                     "an edit that keeps the surface keeps the height's contours, and draws what it draws afresh")
        Check.expect(ContourField.allCases.map(\.fieldClass) == [.function, .function, .coordinate, .coordinate,
                                                                 .embedding],
                     "|f| and arg f are the function's, u and v coordinates, the height the embedding's")

        // sn's torus sizes its tube's samples by its radii, so a radius
        // moves its rectangle's samples and a coordinate is contoured again.
        let sn = SurfaceRequest(.sn(modulus: 0.64, major: 2, minor: 1, grid: 120), grid: 128, name: "sn")
        let snFirst = try SurfaceDrawing.build(sn)
        let snKept = snFirst.layers.map(\.spec) + [contour(.v, [1, 2])]
        let snDrawn = try snFirst.restyled(snKept)
        var fatter = sn
        fatter[.minor] = 1.4
        let snEdited = try SurfaceDrawing.build(fatter, reusing: snDrawn).keeping(snKept, from: snDrawn)
        let snFresh = try SurfaceDrawing.build(fatter).restyled(snKept)
        Check.expect(snEdited.surface.v != snDrawn.surface.v
                     && snEdited.layers.map(\.paths) == snFresh.layers.map(\.paths),
                     "where the rectangle's samples move, v is contoured again: what the edit draws afresh")
    }
}
