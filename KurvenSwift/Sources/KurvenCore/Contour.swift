import Foundation
import simd

/// Marching squares on a sample grid.
///
/// The one stage that sits on the fence between the two halves of the pipeline:
/// cheap enough to port, and porting it is what turns a level set from something
/// baked into a bundle into something you can drag a slider on. contourpy
/// remains the oracle -- `tests/make_fixtures.py` dumps its answers and
/// `kurven-test` compares against them -- but it is no longer in the loop.
///
/// Seam-free by construction: this contours the whole grid as one region, so
/// there are no chunk boundaries to stitch and none of the thread-completion
/// nondeterminism that makes the Python side pass `chunk_count = 1` to get a
/// reproducible answer.
public enum Contour {
    /// A crossing sits on a grid edge, and a grid edge is named by an integer.
    ///
    /// Linking segments into paths by comparing *coordinates* means deciding
    /// how close is close enough. Naming the edge instead makes the join exact:
    /// two cells that share an edge compute the same crossing from the same two
    /// samples, and they agree because they are the same arithmetic, not because
    /// they are within a tolerance.
    struct EdgeID: Hashable {
        let horizontal: Bool
        let x: Int
        let y: Int
    }

    /// Iso-lines of `grid` at `level`, in domain coordinates.
    ///
    /// Closed loops come back with their first vertex repeated at the end, which
    /// is what contourpy does and what a consumer drawing a polyline needs.
    public static func lines(of grid: Grid2D<Float>,
                             level: Double) -> [[P2<DomainSpace>]] {
        guard grid.width >= 2, grid.height >= 2 else { return [] }

        // Every cell contributes zero, one or two segments, each joining two of
        // its four edges. Collected first, linked second.
        var from: [EdgeID: EdgeID] = [:]
        var points: [EdgeID: P2<DomainSpace>] = [:]
        var hasPredecessor = Set<EdgeID>()

        let w = grid.width, h = grid.height
        for y in 0..<(h - 1) {
            for x in 0..<(w - 1) {
                let z0 = Double(grid[x, y])          // bottom-left
                let z1 = Double(grid[x + 1, y])      // bottom-right
                let z2 = Double(grid[x + 1, y + 1])  // top-right
                let z3 = Double(grid[x, y + 1])      // top-left

                var code = 0
                if z0 > level { code |= 1 }
                if z1 > level { code |= 2 }
                if z2 > level { code |= 4 }
                if z3 > level { code |= 8 }
                if code == 0 || code == 15 { continue }

                // Edge ids for this cell: bottom, right, top, left.
                let e0 = EdgeID(horizontal: true, x: x, y: y)
                let e1 = EdgeID(horizontal: false, x: x + 1, y: y)
                let e2 = EdgeID(horizontal: true, x: x, y: y + 1)
                let e3 = EdgeID(horizontal: false, x: x, y: y)

                func crossing(_ id: EdgeID) -> P2<DomainSpace> {
                    if let p = points[id] { return p }
                    let p: P2<DomainSpace>
                    switch id {
                    case e0: p = interpolate(grid, level, x, y, x + 1, y)
                    case e1: p = interpolate(grid, level, x + 1, y, x + 1, y + 1)
                    case e2: p = interpolate(grid, level, x, y + 1, x + 1, y + 1)
                    default: p = interpolate(grid, level, x, y, x, y + 1)
                    }
                    points[id] = p
                    return p
                }

                func link(_ a: EdgeID, _ b: EdgeID) {
                    _ = crossing(a); _ = crossing(b)
                    from[a] = b
                    hasPredecessor.insert(b)
                }

                // Directed so the region above `level` is on the left. The
                // direction is arbitrary for drawing but not for linking: with a
                // consistent rule, a segment's end edge is the next cell's start
                // edge, and the walk is a lookup rather than a search.
                switch code {
                case 1:  link(e0, e3)
                case 2:  link(e1, e0)
                case 3:  link(e1, e3)
                case 4:  link(e2, e1)
                case 6:  link(e2, e0)
                case 7:  link(e2, e3)
                case 8:  link(e3, e2)
                case 9:  link(e0, e2)
                case 11: link(e1, e2)
                case 12: link(e3, e1)
                case 13: link(e0, e1)
                case 14: link(e3, e0)

                // The two saddles, resolved by the cell's mean -- the same rule
                // contourpy uses. Whether the high corners are joined through
                // the middle or separated is genuinely ambiguous from four
                // samples; the mean is the linear interpolant's own answer.
                case 5:   // c0 and c2 above
                    if (z0 + z1 + z2 + z3) / 4 > level {
                        link(e0, e1); link(e2, e3)   // above connects; c1, c3 isolated
                    } else {
                        link(e0, e3); link(e2, e1)   // c0 and c2 isolated
                    }
                case 10:  // c1 and c3 above
                    if (z0 + z1 + z2 + z3) / 4 > level {
                        link(e3, e0); link(e1, e2)   // above connects; c0, c2 isolated
                    } else {
                        link(e1, e0); link(e3, e2)   // c1 and c3 isolated
                    }
                default: break
                }
            }
        }
        return walk(from: from, points: points, hasPredecessor: hasPredecessor)
    }

    /// Iso-lines at several levels, in level order -- the shape
    /// `contours.contour_levels` returns.
    public static func levels(of grid: Grid2D<Float>,
                              _ levels: [Double]) -> [(level: Double, paths: [[P2<DomainSpace>]])] {
        levels.map { ($0, lines(of: grid, level: $0)) }
    }

    private static func interpolate(_ grid: Grid2D<Float>, _ level: Double,
                                    _ ax: Int, _ ay: Int,
                                    _ bx: Int, _ by: Int) -> P2<DomainSpace> {
        let za = Double(grid[ax, ay]), zb = Double(grid[bx, by])
        // The denominator cannot be zero: this edge is only asked about when one
        // end is above the level and the other is not.
        let t = min(max((level - za) / (zb - za), 0), 1)
        let a = grid.position(x: ax, y: ay), b = grid.position(x: bx, y: by)
        return P2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
    }

    /// Chain the segments into polylines: open runs first, from their starts,
    /// then whatever is left, which is closed loops.
    private static func walk(from: [EdgeID: EdgeID], points: [EdgeID: P2<DomainSpace>],
                             hasPredecessor: Set<EdgeID>) -> [[P2<DomainSpace>]] {
        var out: [[P2<DomainSpace>]] = []
        var visited = Set<EdgeID>()

        func trace(_ start: EdgeID, closing: Bool) {
            var path: [P2<DomainSpace>] = []
            var cursor: EdgeID? = start
            while let id = cursor, !visited.contains(id) {
                visited.insert(id)
                if let p = points[id] { path.append(p) }
                cursor = from[id]
            }
            // A closed loop ends where it began; repeating the first vertex is
            // what contourpy does and what a polyline needs to close.
            if closing, let first = path.first, path.count >= 2 { path.append(first) }
            if path.count >= 2 { out.append(path) }
        }

        // Open chains: a start is an edge nothing leads into.
        for id in from.keys where !hasPredecessor.contains(id) {
            trace(id, closing: false)
        }
        // What remains is cycles.
        for id in from.keys where !visited.contains(id) {
            trace(id, closing: true)
        }
        return out
    }
}
