import Foundation

/// Stroke output.
public struct Style: Sendable, Equatable {
    public var color: String
    /// Line width in points, as matplotlib means it.
    public var width: Double
    public init(color: String, width: Double) { self.color = color; self.width = width }

    public init(_ spec: LayerSpec) { self.init(color: spec.color, width: spec.width) }
}

/// The finished drawing: strokes in plate coordinates, in draw order.
public struct Strokes: Sendable {
    public var layers: [(style: Style, paths: PolylineSet<PlateSpace>)]

    public init(layers: [(style: Style, paths: PolylineSet<PlateSpace>)]) {
        self.layers = layers
    }

    public var bounds: AABB<PlateSpace>? {
        layers.compactMap(\.paths.bounds).reduce(nil) { acc, b in
            acc.map { $0.union(b) } ?? b
        }
    }

    public var pathCount: Int { layers.reduce(0) { $0 + $1.paths.count } }
    public var inkLength: Double { layers.reduce(0) { $0 + $1.paths.inkLength } }
}

/// `Strokes` as an SVG document.
///
/// Deliberately not matplotlib's SVG: no figure, no axes, no clip groups, no
/// transform stack -- one `<g>` per layer and one `<polyline>` per path, in a
/// viewBox that is the drawing's own bounds. Equivalence with the Python plate
/// is asserted on strokes, not on pixels, so there is nothing to gain from
/// reproducing matplotlib's chrome and a great deal to lose in trying.
public enum SVG {
    /// Points per inch, and the CSS pixels per inch an SVG viewer assumes.
    static let pointsPerInch = 72.0
    static let pixelsPerInch = 96.0

    /// Render at a given physical width, in inches.
    ///
    /// The width matters because a `LayerSpec.width` is a matplotlib line width
    /// in *points*, and points are only a size once you know how big the drawing
    /// is. matplotlib resolves this with the figure size; the examples use 16
    /// inches, so that is the default here and a stroke comes out the same
    /// physical thickness it does on the plate. Changing `figureWidth` scales
    /// the paper, not the ink -- exactly as re-saving a matplotlib figure at a
    /// different `figsize` would.
    public static func render(_ strokes: Strokes, figureWidth: Double = 16,
                              margin: Double = 0.02, decimals: Int = 4) -> String {
        guard let b = strokes.bounds else {
            return #"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>"# + "\n"
        }
        let pad = max(b.size.x, b.size.y) * margin
        let x0 = b.lo.x - pad, y0 = b.lo.y - pad
        let w = b.size.x + 2 * pad, h = b.size.y + 2 * pad
        // Plate units per point at this figure width.
        let unitsPerPoint = w / (figureWidth * pointsPerInch)

        var out = """
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" \
        viewBox="\(fmt(x0, decimals)) \(fmt(y0, decimals)) \(fmt(w, decimals)) \(fmt(h, decimals))" \
        width="\(fmt(figureWidth * pixelsPerInch, 3))" \
        height="\(fmt(figureWidth * pixelsPerInch * h / w, 3))">
        <g fill="none" stroke-linecap="round" stroke-linejoin="round" \
        transform="translate(0,\(fmt(2 * y0 + h, decimals))) scale(1,-1)">

        """
        // The y flip: SVG's y grows downward, the plate's grows upward. Doing it
        // once on the outer group means every coordinate below is a plate
        // coordinate, unnegated, so a stroke list can be read against the
        // Python arrays directly.
        for (style, paths) in strokes.layers where !paths.isEmpty {
            out += #"<g stroke="\#(style.color)" stroke-width="\#(fmt(style.width * unitsPerPoint, 6))">"# + "\n"
            for i in 0..<paths.count {
                out += "<polyline points=\""
                out += paths[path: i].map { "\(fmt($0.x, decimals)),\(fmt($0.y, decimals))" }
                    .joined(separator: " ")
                out += "\"/>\n"
            }
            out += "</g>\n"
        }
        out += "</g>\n</svg>\n"
        return out
    }

    static func fmt(_ v: Double, _ decimals: Int) -> String {
        var s = String(format: "%.\(decimals)f", v)
        if s.contains(".") {
            while s.hasSuffix("0") { s.removeLast() }
            if s.hasSuffix(".") { s.removeLast() }
        }
        return s == "-0" ? "0" : s
    }
}
