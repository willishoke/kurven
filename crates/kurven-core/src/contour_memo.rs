//! A refiner that remembers what it placed. The port of
//! `Sources/KurvenCore/ContourMemo.swift`: an extension of [`ContourRefine`]
//! and the table behind it.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};

use kurven_math::real;

use crate::manifest::ContourField;
use crate::spaces::{DomainSpace, P2};
use crate::surface::{ContourRefine, Contours, Fork, Rings};

type Paths = Vec<Vec<P2<DomainSpace>>>;

impl ContourRefine {
    /// The ceiling a remembering refiner is asked about for `height`: the
    /// next power of 2^(1/4) at or above it, so a truncation dragged through a fifth
    /// of its height asks one question. A rung is its own rung, so one
    /// remembering refiner wrapped in another asks what it was asked.
    pub fn rung(height: f64) -> f64 {
        if !(height.is_finite() && height > 0.0) {
            return if height > 0.0 { f64::INFINITY } else { height };
        }
        real::pow(2.0, (4.0 * real::log2(height) - 1e-9).ceil() / 4.0)
    }

    /// The same refiner, remembering what it placed, up to Swift's default
    /// budget of 4,000,000 points ([`ContourRefine::remembering_within`]).
    pub fn remembering(&self) -> ContourRefine {
        self.remembering_within(4_000_000)
    }

    /// The same refiner, remembering what it placed.
    ///
    /// Placing a contour on f is a function of the field, the level and the
    /// grid's paths and of nothing else -- not the truncation, not the hatching, not
    /// which other layers there are. A restyle derives every layer again from
    /// the same grids, so without this a truncation slider re-solved every phase line
    /// of the landscape on each step to clip them at a new height: 4.5 s on
    /// zeta, of which 5 ms was the clipping. With it, a restyle places only
    /// the curves it has not placed before -- the truncation edge at the new truncation, the
    /// magnitude rings at the new levels -- and the rest is a lookup. The
    /// rings round the zeros at a level are a function of the level and the
    /// curves placed there, and are remembered by those.
    ///
    /// The answer for a key is the answer the refiner gives, so the curves are
    /// the same with or without it, to the bit. What it costs is memory:
    /// every placed curve and the grid paths it came from, up to `budget`
    /// points, past which the least recently asked are forgotten.
    ///
    /// A ceiling is rounded up to a rung of `ContourRefine::rung`'s ladder
    /// before it is asked about, so on one rung a truncation slider's steps are the
    /// same question.
    pub fn remembering_within(&self, budget: usize) -> ContourRefine {
        self.remembering_in(Arc::new(ContourMemo::new(budget)))
    }

    /// Remembering in `memo`, and so are its forks: a fork is the same
    /// function, so what one thread placed another need not place again.
    pub fn remembering_in(&self, memo: Arc<ContourMemo>) -> ContourRefine {
        let contours = self.contours.clone();
        let forked: Option<Fork> = self.fork.clone().map(|f| {
            let memo = memo.clone();
            let remembered: Fork = Arc::new(move || f().remembering_in(memo.clone()));
            remembered
        });
        let rings: Option<Rings> = self.rings.clone().map(|seeded| {
            let memo = memo.clone();
            let remembered: Rings = Arc::new(move |level, existing| {
                memo.placed(
                    ContourField::Magnitude,
                    level,
                    existing,
                    f64::INFINITY,
                    ContourRefine::EVERYWHERE,
                    true,
                    || seeded(level, existing),
                )
            });
            remembered
        });
        let remembered: Contours = Arc::new(move |field, level, paths, ceiling, span| {
            let rung = ContourRefine::rung(ceiling);
            memo.placed(field, level, paths, rung, span.clone(), false, || {
                contours(field, level, paths, rung, span.clone())
            })
        });
        ContourRefine {
            contours: remembered,
            rings,
            magnitude: self.magnitude.clone(),
            phase: self.phase.clone(),
            zero: self.zero.clone(),
            tolerance: self.tolerance,
            fork: forked,
        }
    }
}

/// A memo key. The paths are compared as Swift's `Hashable` points compare,
/// by `==` on their coordinates (so -0 is 0); a key with a NaN coordinate,
/// which Swift's table could never find again, is not stored at all.
#[derive(Debug, Clone)]
struct Key {
    field: ContourField,
    /// By its bits, so -0 and 0 are different levels, as they are to
    /// anyone comparing the answers.
    level: u64,
    ceiling: u64,
    span: (u64, u64),
    /// The paths placed, or for the rings at the level the curves
    /// already placed there.
    paths: Paths,
    rings: bool,
}

impl PartialEq for Key {
    fn eq(&self, o: &Key) -> bool {
        self.field == o.field
            && self.level == o.level
            && self.ceiling == o.ceiling
            && self.span == o.span
            && self.rings == o.rings
            && self.paths == o.paths
    }
}

// Reflexive: keys holding a NaN never reach the table.
impl Eq for Key {}

impl Hash for Key {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.field.hash(h);
        self.level.hash(h);
        self.ceiling.hash(h);
        self.span.hash(h);
        self.rings.hash(h);
        self.paths.len().hash(h);
        for path in &self.paths {
            path.len().hash(h);
            for p in path {
                // + 0.0 makes -0 into 0, so equal coordinates hash alike.
                (p.x() + 0.0).to_bits().hash(h);
                (p.y() + 0.0).to_bits().hash(h);
            }
        }
    }
}

struct Entry {
    placed: Paths,
    points: usize,
    used: u64,
}

#[derive(Default)]
struct Table {
    entries: HashMap<Key, Entry>,
    points: usize,
    clock: u64,
    hits: usize,
    misses: usize,
}

/// The table behind `ContourRefine::remembering`. Its own type so a test can
/// ask how often it answered without solving.
pub struct ContourMemo {
    budget: usize,
    table: Mutex<Table>,
}

impl ContourMemo {
    pub fn new(budget: usize) -> ContourMemo {
        ContourMemo { budget, table: Mutex::new(Table::default()) }
    }

    fn table(&self) -> std::sync::MutexGuard<'_, Table> {
        // A panic while the lock was held leaves nothing half-written that
        // matters: an entry is inserted whole.
        self.table.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// How often it answered from the table, and how often it solved.
    pub fn counts(&self) -> (usize, usize) {
        let t = self.table();
        (t.hits, t.misses)
    }

    /// The answer for these paths, solved by `solve` the first time it is
    /// asked; with `rings`, the rings at the level given these curves. The
    /// lock is held for the lookup and the insert, never for the solve.
    #[allow(clippy::too_many_arguments)]
    pub fn placed(
        &self,
        field: ContourField,
        level: f64,
        paths: &[Vec<P2<DomainSpace>>],
        ceiling: f64,
        span: RangeInclusive<f64>,
        rings: bool,
        solve: impl FnOnce() -> Paths,
    ) -> Paths {
        if paths.iter().flatten().any(|p| p.x().is_nan() || p.y().is_nan()) {
            // Swift stores such a key and can never find it again: the same
            // answer, solved every time.
            self.table().misses += 1;
            return solve();
        }
        let key = Key {
            field,
            level: level.to_bits(),
            ceiling: ceiling.to_bits(),
            span: (span.start().to_bits(), span.end().to_bits()),
            paths: paths.to_vec(),
            rings,
        };
        let found = {
            let mut t = self.table();
            t.clock += 1;
            let clock = t.clock;
            match t.entries.get_mut(&key) {
                Some(entry) => {
                    entry.used = clock;
                    let placed = entry.placed.clone();
                    t.hits += 1;
                    Some(placed)
                }
                None => {
                    t.misses += 1;
                    None
                }
            }
        };
        if let Some(found) = found {
            return found;
        }
        let placed = solve();
        let points = paths.iter().map(Vec::len).sum::<usize>() + placed.iter().map(Vec::len).sum::<usize>();
        if points > self.budget {
            return placed;
        }
        let mut t = self.table();
        t.clock += 1;
        let used = t.clock;
        if let Some(old) = t.entries.insert(key, Entry { placed: placed.clone(), points, used }) {
            t.points -= old.points;
        }
        t.points += points;
        if t.points > self.budget {
            // Forget the least recently asked until the table fits. A restyle
            // asks for every curve on screen, so those are the last forgotten.
            let mut order: Vec<(u64, Key)> = t.entries.iter().map(|(k, e)| (e.used, k.clone())).collect();
            order.sort_by_key(|(used, _)| *used);
            for (_, k) in order {
                if t.points <= self.budget {
                    break;
                }
                if let Some(e) = t.entries.remove(&k) {
                    t.points -= e.points;
                }
            }
        }
        placed
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn counting() -> (ContourRefine, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let contours: Contours = Arc::new(move |_, level, paths, ceiling, _| {
            c.fetch_add(1, Ordering::SeqCst);
            paths
                .iter()
                .map(|p| p.iter().map(|q| P2::new(q.x() + level, q.y() + ceiling.min(1e300))).collect())
                .collect()
        });
        (ContourRefine::new(contours, None), calls)
    }

    #[test]
    fn a_rung_is_its_own_rung() {
        for h in [0.3, 1.0, 4.0, 5.0, 6.0, 1e6] {
            let r = ContourRefine::rung(h);
            assert!(r >= h && r < h * real::pow(2.0, 0.25) * (1.0 + 1e-12), "{h} -> {r}");
            assert_eq!(ContourRefine::rung(r), r, "the rung of {h}'s rung");
        }
        assert_eq!(ContourRefine::rung(f64::INFINITY), f64::INFINITY);
        assert_eq!(ContourRefine::rung(0.0), 0.0);
        assert!(ContourRefine::rung(f64::NAN).is_nan());
        assert_eq!(ContourRefine::rung(4.9), ContourRefine::rung(4.95), "a fifth of a step is one question");
    }

    #[test]
    fn a_remembering_refiner_places_each_question_once() {
        let (refine, calls) = counting();
        let memo = Arc::new(ContourMemo::new(1000));
        let remembered = refine.remembering_in(memo.clone());
        let paths = vec![vec![P2::new(0.0, 0.0), P2::new(1.0, 0.0)]];
        let a = (remembered.contours)(ContourField::Phase, 0.5, &paths, 4.9, ContourRefine::EVERYWHERE);
        let b = (remembered.contours)(ContourField::Phase, 0.5, &paths, 4.95, ContourRefine::EVERYWHERE);
        assert_eq!(a, b, "one rung, one answer");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(memo.counts(), (1, 1));
        // Asked with the rung, as the refiner was.
        assert_eq!(a[0][0].y(), ContourRefine::rung(4.9));
        // -0 and 0 are different levels; -0 and 0 in the paths are not.
        let _ = (remembered.contours)(ContourField::Phase, -0.0, &paths, 4.9, ContourRefine::EVERYWHERE);
        let _ = (remembered.contours)(ContourField::Phase, 0.0, &paths, 4.9, ContourRefine::EVERYWHERE);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        let negative = vec![vec![P2::new(-0.0, 0.0), P2::new(1.0, -0.0)]];
        let _ = (remembered.contours)(ContourField::Phase, 0.0, &negative, 4.9, ContourRefine::EVERYWHERE);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        // A NaN in the paths is never found again: solved each time.
        let nan = vec![vec![P2::new(f64::NAN, 0.0), P2::new(1.0, 0.0)]];
        let _ = (remembered.contours)(ContourField::Phase, 0.0, &nan, 4.9, ContourRefine::EVERYWHERE);
        let _ = (remembered.contours)(ContourField::Phase, 0.0, &nan, 4.9, ContourRefine::EVERYWHERE);
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn past_its_budget_it_forgets_the_least_recently_asked() {
        let (refine, calls) = counting();
        // Each question is 2 path points and 2 placed: 4.
        let remembered = refine.remembering_within(8);
        let at = |x: f64| vec![vec![P2::new(x, 0.0), P2::new(x + 1.0, 0.0)]];
        let ask = |x: f64| {
            (remembered.contours)(ContourField::Magnitude, 1.0, &at(x), f64::INFINITY, ContourRefine::EVERYWHERE)
        };
        ask(0.0);
        ask(1.0);
        ask(0.0); // 0 is now the more recent
        ask(2.0); // over budget: 1 is forgotten
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        ask(0.0);
        assert_eq!(calls.load(Ordering::SeqCst), 3, "0 remembered");
        ask(1.0);
        assert_eq!(calls.load(Ordering::SeqCst), 4, "1 forgotten");
    }
}
