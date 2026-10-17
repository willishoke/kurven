import Foundation
import Synchronization

public extension ContourRefine {
    /// The ceiling a remembering refiner is asked about for `height`: the
    /// next power of 2^(1/4) at or above it, so a cap dragged through a fifth
    /// of its height asks one question. A rung is its own rung, so one
    /// remembering refiner wrapped in another asks what it was asked.
    static func rung(above height: Double) -> Double {
        guard height.isFinite, height > 0 else { return height > 0 ? .infinity : height }
        return Foundation.pow(2, (4 * Foundation.log2(height) - 1e-9).rounded(.up) / 4)
    }

    /// The same refiner, remembering what it placed.
    ///
    /// Placing a contour on f is a function of the field, the level and the
    /// grid's paths and of nothing else -- not the cap, not the hatching, not
    /// which other layers there are. A restyle derives every layer again from
    /// the same grids, so without this a cap slider re-solved every phase line
    /// of the landscape on each step to clip them at a new height: 4.5 s on
    /// zeta, of which 5 ms was the clipping. With it, a restyle places only
    /// the curves it has not placed before -- the rim at the new cap, the
    /// magnitude rings at the new levels -- and the rest is a lookup.
    ///
    /// The answer for a key is the answer the refiner gives, so the ink is
    /// the same with or without it, to the bit. What it costs is memory:
    /// every placed curve and the grid paths it came from, up to `budget`
    /// points, past which the least recently asked are forgotten.
    ///
    /// A ceiling is rounded up to a rung of `ContourRefine.rung`'s ladder
    /// before it is asked about, so on one rung a cap slider's steps are the
    /// same question.
    func remembering(budget: Int = 4_000_000) -> ContourRefine {
        remembering(in: ContourMemo(budget: budget))
    }

    /// Remembering in `memo`, and so are its forks: a fork is the same
    /// function, so what one thread placed another need not place again.
    internal func remembering(in memo: ContourMemo) -> ContourRefine {
        let contours = self.contours
        let forked: (@Sendable () -> ContourRefine)? = self.fork.map { f in
            let remembered: @Sendable () -> ContourRefine = { f().remembering(in: memo) }
            return remembered
        }
        return ContourRefine(contours: { field, level, paths, ceiling in
                                 let rung = ContourRefine.rung(above: ceiling)
                                 return memo.placed(field, level, paths, rung) {
                                     contours(field, level, paths, rung)
                                 }
                             },
                             magnitude: magnitude, zero: zero, tolerance: tolerance,
                             fork: forked)
    }
}

/// The table behind `ContourRefine.remembering`. Its own type so a test can
/// ask how often it answered without solving.
final class ContourMemo: Sendable {
    struct Key: Hashable, Sendable {
        let field: ContourField
        /// By its bits, so -0 and 0 are different levels, as they are to
        /// anyone comparing the answers.
        let level: UInt64
        let ceiling: UInt64
        let paths: [[P2<DomainSpace>]]
    }


    private struct Entry {
        var placed: [[P2<DomainSpace>]]
        var points: Int
        var used: UInt64
    }

    private struct Table {
        var entries: [Key: Entry] = [:]
        var points = 0
        var clock: UInt64 = 0
        var hits = 0
        var misses = 0
    }

    let budget: Int
    private let table = Mutex(Table())

    init(budget: Int) { self.budget = budget }

    var counts: (hits: Int, misses: Int) {
        table.withLock { ($0.hits, $0.misses) }
    }

    /// The answer for these paths, solved by `solve` the first time it is
    /// asked. The lock is held for the lookup and the insert, never for the
    /// solve: a layer's levels are placed in parallel, and two threads asking
    /// the same new key both solve it, which costs time and never a
    /// different answer.
    func placed(_ field: ContourField, _ level: Double, _ paths: [[P2<DomainSpace>]],
                _ ceiling: Double, solve: () -> [[P2<DomainSpace>]]) -> [[P2<DomainSpace>]] {
        let key = Key(field: field, level: level.bitPattern, ceiling: ceiling.bitPattern, paths: paths)
        let found: [[P2<DomainSpace>]]? = table.withLock { t in
            t.clock += 1
            guard var entry = t.entries[key] else { t.misses += 1; return nil }
            t.hits += 1
            entry.used = t.clock
            t.entries[key] = entry
            return entry.placed
        }
        if let found { return found }
        let placed = solve()
        let points = paths.reduce(0) { $0 + $1.count } + placed.reduce(0) { $0 + $1.count }
        guard points <= budget else { return placed }
        table.withLock { t in
            t.clock += 1
            if let old = t.entries.updateValue(Entry(placed: placed, points: points, used: t.clock),
                                               forKey: key) {
                t.points -= old.points
            }
            t.points += points
            guard t.points > budget else { return }
            // Forget the least recently asked until the table fits. A restyle
            // asks for every curve on screen, so those are the last forgotten.
            for (k, e) in t.entries.sorted(by: { $0.value.used < $1.value.used }) {
                guard t.points > budget else { break }
                t.entries[k] = nil
                t.points -= e.points
            }
        }
        return placed
    }
}
