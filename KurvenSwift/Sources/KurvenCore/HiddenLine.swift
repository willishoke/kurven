import Foundation

/// Hidden-line removal: a pure function of a depth image and some polylines.
///
/// A line-for-line port of `outline.clip_hidden_lines`. A vertex is visible when
/// `z + margin > buffer[pixel]`; hidden vertices split a path into runs; runs
/// shorter than two vertices draw nothing and are dropped. Because it takes a
/// `DepthImage` *value*, it is tested against a Python-dumped buffer and
/// Python-dumped polylines with no GPU anywhere in the loop -- which is what
/// makes "the bake reproduces the plate" a checkable claim rather than a
/// rendering that looks about right.
///
/// The realtime preview does the same test per *fragment* in a shader, which can
/// disagree with this on sub-pixel runs. That disagreement is deliberate and
/// bounded: the preview is for navigating, the bake is the artifact, and the
/// bake is this.
public enum HiddenLine {
    public static func clip(_ paths: PolylineSet<ViewSpace>, against depth: DepthImage,
                            margin: Double) -> PolylineSet<PlateSpace> {
        var out: [[P3<PlateSpace>]] = []
        for i in 0..<paths.count {
            var run: [P3<PlateSpace>] = []
            for v in paths[path: i] {
                if v.z + margin > depth.depth(under: v.xy) {
                    run.append(P3(v.x, v.y, v.z))
                } else {
                    if run.count >= 2 { out.append(run) }
                    run = []
                }
            }
            if run.count >= 2 { out.append(run) }
        }
        return PolylineSet(paths: out)
    }

    /// Ink that is not depth-tested: a cut-face hatch lies *in* the wall it
    /// hatches, so testing it would erase about half of it to no purpose.
    /// Passing it through this rather than casting keeps "which space am I in"
    /// a decision the type system witnesses.
    public static func pass(_ paths: PolylineSet<ViewSpace>) -> PolylineSet<PlateSpace> {
        PolylineSet(vertices: paths.vertices.map { P3<PlateSpace>($0.v) },
                    offsets: paths.offsets)
    }
}
