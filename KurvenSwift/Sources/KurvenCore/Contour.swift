import Foundation
import Dispatch
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

    /// A coarse min/max summary of the grid, so a level can skip the blocks it
    /// cannot cross.
    ///
    /// Contouring a level means asking every cell whether the level passes
    /// through it, and for a smooth field the answer is almost always no. On a
    /// 1200-square grid with ninety levels that is a hundred and thirty million
    /// questions with a known answer. Blocks of 32 make it a few hundred
    /// thousand.
    ///
    /// Built once per grid and reused across levels, which is why it is a value
    /// rather than something `lines` computes for itself.
    public struct Index: Sendable {
        static let block = 32
        let width: Int, height: Int
        let blocksAcross: Int, blocksDown: Int
        let lo: [Float], hi: [Float]

        public init(_ grid: Grid2D<Float>) {
            width = grid.width; height = grid.height
            let cellsX = max(grid.width - 1, 0), cellsY = max(grid.height - 1, 0)
            blocksAcross = (cellsX + Index.block - 1) / Index.block
            blocksDown = (cellsY + Index.block - 1) / Index.block
            var lo = [Float](repeating: .infinity, count: max(blocksAcross * blocksDown, 0))
            var hi = [Float](repeating: -.infinity, count: lo.count)
            for by in 0..<blocksDown {
                let y0 = by * Index.block, y1 = min(y0 + Index.block + 1, grid.height)
                for bx in 0..<blocksAcross {
                    let x0 = bx * Index.block, x1 = min(x0 + Index.block + 1, grid.width)
                    var l = Float.infinity, h = -Float.infinity
                    for y in y0..<y1 {
                        for x in x0..<x1 {
                            let v = grid[x, y]
                            if v < l { l = v }
                            if v > h { h = v }
                        }
                    }
                    lo[by * blocksAcross + bx] = l
                    hi[by * blocksAcross + bx] = h
                }
            }
            self.lo = lo; self.hi = hi
        }

        /// A block can only contribute when some corner is above the level and
        /// some corner is not -- the same test the cell code performs, lifted to
        /// thirty-two cells at a time.
        func crosses(_ b: Int, _ level: Double) -> Bool {
            Double(hi[b]) > level && Double(lo[b]) <= level
        }
    }

    /// Iso-lines of `grid` at `level`, in domain coordinates.
    ///
    /// Closed loops come back with their first vertex repeated at the end, which
    /// is what contourpy does and what a consumer drawing a polyline needs.
    public static func lines(of grid: Grid2D<Float>, level: Double,
                             index: Index? = nil) -> [[P2<DomainSpace>]] {
        guard grid.width >= 2, grid.height >= 2 else { return [] }
        let index = index ?? Index(grid)

        // Every cell contributes zero, one or two segments, each joining two of
        // its four edges. Collected first, linked second.
        var from: [EdgeID: EdgeID] = [:]
        var points: [EdgeID: P2<DomainSpace>] = [:]
        var hasPredecessor = Set<EdgeID>()

        let w = grid.width, h = grid.height
        for by in 0..<index.blocksDown {
            for bx in 0..<index.blocksAcross {
                guard index.crosses(by * index.blocksAcross + bx, level) else { continue }
                let ys = (by * Index.block)..<min((by + 1) * Index.block, h - 1)
                let xs = (bx * Index.block)..<min((bx + 1) * Index.block, w - 1)
                for y in ys {
            for x in xs {
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
            }
        }
        return walk(from: from, points: points, hasPredecessor: hasPredecessor)
    }

    /// Iso-lines at several levels, in level order -- the shape
    /// `contours.contour_levels` returns.
    ///
    /// Levels are independent of each other, so they run concurrently. That
    /// they *are* independent is a property of contouring the whole grid at
    /// once: the Python side's threaded backend parallelizes by splitting the
    /// grid into chunks instead, which is why it has seams to stitch and why
    /// its output depends on the order the threads finish in.
    public static func levels(of grid: Grid2D<Float>, _ levels: [Double])
        -> [(level: Double, paths: [[P2<DomainSpace>]])] {
        guard !levels.isEmpty else { return [] }
        let index = Index(grid)
        guard levels.count > 1 else {
            return [(levels[0], lines(of: grid, level: levels[0], index: index))]
        }
        let results = UnsafeMutablePointer<[[P2<DomainSpace>]]>.allocate(capacity: levels.count)
        results.initialize(repeating: [], count: levels.count)
        defer { results.deinitialize(count: levels.count); results.deallocate() }
        DispatchQueue.concurrentPerform(iterations: levels.count) { i in
            results[i] = lines(of: grid, level: levels[i], index: index)
        }
        return (0..<levels.count).map { (levels[$0], results[$0]) }
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
