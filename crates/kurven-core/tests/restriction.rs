//! "restriction: a contour's copy is cut to the footprint where it lands,
//! exactly", from `Sources/kurven-test/Faces.swift`, with Swift's
//! tolerances, over the landscapes of Swift's `faceLandscapes()` that are
//! drawn on the grid: the contract's bundles and the hatch fixture's
//! variants. Those on f (the towers) need the refiner, and are
//! `kurven-landscape/tests/faces.rs`'s.

// Swift's `!(d <= tolerance)` keeps an infinite or NaN distance outside.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::HashSet;

use kurven_core::bundle::KurvenBundle;
use kurven_core::fixtures;
use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceSelector;
use kurven_core::json::Members;
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, ContourField, Domain, DomainAxis, HeightPolicy, Interval, LayerSource, PerimeterEdge,
    RealBand, Region, Restriction, Truncation,
};
use kurven_core::polyline::PolylineSet;
use kurven_core::simd::Vec2;
use kurven_core::spaces::{P2, P3, WorldSpace};
use kurven_core::surface::Surface;
use kurven_math::real;

struct Landscape {
    name: String,
    surface: Surface,
    tiles: Vec<Affine2>,
    region: Region,
}

impl Landscape {
    /// Whether any sample of the grid is under the truncation.
    fn has_graph(&self) -> bool {
        let g = &*self.surface.height;
        (0..g.width).any(|i| {
            let h = self.surface.truncation_height_at(g.position(i, 0).x());
            (0..g.height).any(|j| f64::from(g.at(i, j)) < h)
        })
    }
}

fn footprint(corners: &[(f64, f64)], density: i64) -> BoundaryPerimeter {
    BoundaryPerimeter::new(
        (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density }
            })
            .collect(),
    )
}

fn affine(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Affine2 {
    Affine2 { a, b, tx, c, d, ty }
}

/// Swift's `faceLandscapes()`, those drawn on the grid.
fn landscapes() -> Vec<Landscape> {
    let mut out: Vec<Landscape> = Vec::new();
    let mut contract: Vec<_> = std::fs::read_dir(fixtures::shared("contract"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "kurven"))
        .collect();
    contract.sort();
    for url in contract {
        let Ok(bundle) = KurvenBundle::read(&url) else { continue };
        let o = &bundle.manifest.occluder;
        out.push(Landscape {
            name: url.file_name().unwrap().to_string_lossy().into_owned(),
            surface: bundle.surface.clone(),
            tiles: o.tiles.clone(),
            region: o.region.clone(),
        });
    }
    let index = fixtures::shared_json("hatch/index.json");
    let index = index.as_object().unwrap();
    let shape = index.value("shape", "hatch").unwrap().object("hatch.shape").unwrap();
    let grid = Grid2D::new(
        shape.int("nx", "hatch.shape").unwrap() as usize,
        shape.int("ny", "hatch.shape").unwrap() as usize,
        Domain::from_json(index.value("domain", "hatch").unwrap()).unwrap(),
        fixtures::shared_npy("hatch/height.npy").floats().unwrap(),
    );
    let truncations = index.value("truncation", "hatch").unwrap().object("hatch.truncations").unwrap();
    let banded = Surface::new(
        grid.clone(),
        None,
        Truncation::from_json(truncations.value("bands", "hatch.truncations").unwrap()).unwrap(),
        false,
    );
    let low = Surface::new(
        grid.clone(),
        None,
        Truncation::RealBands {
            bands: vec![RealBand { below: -1.0, height: 0.3 }, RealBand { below: 0.4, height: 0.7 }],
            beyond: 0.45,
        },
        false,
    );
    let mut add = |name: &str, surface: &Surface, tiles: Vec<Affine2>, region: Region| {
        out.push(Landscape { name: name.to_string(), surface: surface.clone(), tiles, region });
    };
    for (name, surface, corners) in [
        (
            "bands, on a notch from Re z = 0.5 to 1.5",
            &banded,
            vec![(-2.0, -1.0), (0.5, -1.0), (0.5, 0.0), (1.5, 0.0), (1.5, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)],
        ),
        (
            "low bands, on a notch from one boundary to the other",
            &low,
            vec![
                (-2.0, -1.0),
                (-1.0, -1.0),
                (-1.0, 0.6),
                (0.4, 0.6),
                (0.4, -1.0),
                (2.5, -1.0),
                (2.5, 1.5),
                (-2.0, 1.5),
            ],
        ),
        (
            "low bands, on notches from the left edge to one boundary and from the other to 1.5",
            &low,
            vec![
                (-1.0, -1.0),
                (0.4, -1.0),
                (0.4, 0.6),
                (1.5, 0.6),
                (1.5, -1.0),
                (2.5, -1.0),
                (2.5, 1.5),
                (-2.0, 1.5),
                (-2.0, 0.6),
                (-1.0, 0.6),
            ],
        ),
    ] {
        add(
            &format!("the hatch fixture, {name}"),
            surface,
            vec![Affine2::IDENTITY],
            Region::Inside(footprint(&corners, 9)),
        );
    }
    let reflected = affine(-1.0, 0.0, 5.0, 0.0, 1.0, 0.0);
    let turned = affine(0.0, -1.0, 10.0, 1.0, 0.0, 0.3);
    let pi = core::f64::consts::PI;
    let reflected_by_angle = affine(real::cos(pi), real::sin(pi), 5.0, real::sin(pi), -real::cos(pi), 0.0);
    let turned_by_angle =
        affine(real::cos(pi / 2.0), -real::sin(pi / 2.0), 10.0, real::sin(pi / 2.0), real::cos(pi / 2.0), 0.3);
    let higher_reflected = vec![(4.6, -1.0), (5.9, -1.0), (5.9, 1.5), (4.6, 1.5)];
    let lower_reflected = vec![(3.0, -1.0), (4.6, -1.0), (4.6, 1.5), (3.0, 1.5)];
    let higher_turned = vec![(8.6, -0.5), (10.9, -0.5), (10.9, 0.7), (8.6, 0.7)];
    let lower_turned = vec![(8.6, 0.7), (10.9, 0.7), (10.9, 2.7), (8.6, 2.7)];
    for (name, tile, corners) in [
        ("reflected, the footprint on the higher band's side", reflected, &higher_reflected),
        ("reflected, the footprint on the lower band's side", reflected, &lower_reflected),
        ("turned, the footprint on the higher band's side", turned, &higher_turned),
        ("turned, the footprint on the lower band's side", turned, &lower_turned),
        ("reflected by an angle, the footprint on the higher band's side", reflected_by_angle, &higher_reflected),
        ("reflected by an angle, the footprint on the lower band's side", reflected_by_angle, &lower_reflected),
        ("turned by an angle, the footprint on the higher band's side", turned_by_angle, &higher_turned),
        ("turned by an angle, the footprint on the lower band's side", turned_by_angle, &lower_turned),
    ] {
        add(
            &format!("the hatch fixture, low bands, a wall on a boundary's image in a tile {name}"),
            &low,
            vec![Affine2::IDENTITY, tile],
            Region::Inside(footprint(corners, 9)),
        );
    }
    let notched: Vec<(f64, f64)> =
        [(-2.0, -1.0), (-1.0, -1.0), (-1.0, 0.6), (0.4, 0.6), (0.4, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)]
            .iter()
            .map(|&(x, y)| {
                let p = turned.apply(P3::new(x, y, 0.0));
                (p.x(), p.y())
            })
            .collect();
    add(
        "the hatch fixture, low bands, on a notch from one boundary to the other, in a tile turned",
        &low,
        vec![Affine2::IDENTITY, turned],
        Region::Inside(footprint(&notched, 9)),
    );
    let reflections = vec![
        Affine2::IDENTITY,
        affine(-1.0, 0.0, 5.0, 0.0, 1.0, 0.0),
        affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        affine(-1.0, 0.0, 5.0, 0.0, -1.0, 3.0),
    ];
    let across = footprint(&[(-1.5, -0.5), (5.5, -0.8), (6.6, 1.2), (6.2, 3.6), (1.0, 3.3), (-1.8, 2.0)], 9);
    add(
        "the hatch fixture, low bands, in four tiles, on a footprint across them",
        &low,
        reflections,
        Region::Inside(across),
    );
    let unit = Domain::new(Interval::new(0.0, 1.0), Interval::new(0.0, 1.0));
    let parabola = Surface::new(
        Grid2D::new(
            21,
            21,
            unit,
            (0..441)
                .map(|k| {
                    let t = 2.0 * (k % 21) as f64 / 20.0 - 1.0;
                    (3.0 * real::pow(t, 2.0) + 0.2) as f32
                })
                .collect(),
        ),
        None,
        Truncation::RealBands { bands: vec![RealBand { below: 0.5, height: 1.0 }], beyond: 2.0 },
        false,
    );
    for (tile_name, tile) in
        [("beside it", affine(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)), ("turned", affine(0.0, 1.0, 1.0, -1.0, 0.0, 1.0))]
    {
        for (side, corners) in [
            ("second", vec![(1.0, 0.2), (1.8, 0.2), (1.8, 0.8), (1.0, 0.8)]),
            ("first", vec![(0.2, 0.2), (1.0, 0.2), (1.0, 0.8), (0.2, 0.8)]),
        ] {
            add(
                &format!("a wall on the tile boundary with a tile {tile_name}, the footprint in the {side}"),
                &parabola,
                vec![Affine2::IDENTITY, tile],
                Region::Inside(footprint(&corners, 9)),
            );
        }
    }
    add(
        "copies meeting with different tops, a tile turned, on a footprint across the tile boundary, on the grid",
        &parabola,
        vec![Affine2::IDENTITY, affine(0.0, 1.0, 1.0, -1.0, 0.0, 1.0)],
        Region::Inside(footprint(&[(0.2, 0.2), (1.8, 0.2), (1.8, 0.8), (0.2, 0.8)], 9)),
    );
    out
}

/// Polylines as segments, bucketed so that a point is met only with those
/// near it: the distance to them within a radius.
struct Segments {
    a: Vec<Vec2>,
    b: Vec<Vec2>,
    lo: Vec2,
    size: f64,
    columns: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
}

impl Segments {
    fn new(paths: &[Vec<Vec2>], size: f64) -> Segments {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for path in paths {
            for w in path.windows(2) {
                if w[0] != w[1] {
                    a.push(w[0]);
                    b.push(w[1]);
                }
            }
        }
        let mut lo = Vec2::splat(f64::INFINITY);
        let mut hi = -lo;
        for (p, q) in a.iter().zip(&b) {
            lo = lo.min(p.min(*q));
            hi = hi.max(p.max(*q));
        }
        if a.is_empty() {
            lo = Vec2::ZERO;
            hi = Vec2::ZERO;
        }
        let columns = (((hi.x - lo.x) / size).floor() as usize + 1).clamp(1, 4096);
        let rows = (((hi.y - lo.y) / size).floor() as usize + 1).clamp(1, 4096);
        let mut cells = vec![Vec::new(); columns * rows];
        for k in 0..a.len() {
            let (l, h) = (a[k].min(b[k]), a[k].max(b[k]));
            let (i0, i1) = (Segments::index(l.x, lo.x, size, columns), Segments::index(h.x, lo.x, size, columns));
            let (j0, j1) = (Segments::index(l.y, lo.y, size, rows), Segments::index(h.y, lo.y, size, rows));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    cells[j * columns + i].push(k);
                }
            }
        }
        Segments { a, b, lo, size, columns, rows, cells }
    }

    fn index(x: f64, lo: f64, size: f64, count: usize) -> usize {
        let f = ((x - lo) / size).floor();
        if f < 0.0 {
            0
        } else if f >= count as f64 {
            count - 1
        } else {
            f as usize
        }
    }

    /// The distance from `p` to the nearest segment, or infinity where none
    /// lies within `r`.
    fn distance(&self, p: Vec2, r: f64) -> f64 {
        if self.a.is_empty() {
            return f64::INFINITY;
        }
        let (i0, i1) = (
            Segments::index(p.x - r, self.lo.x, self.size, self.columns),
            Segments::index(p.x + r, self.lo.x, self.size, self.columns),
        );
        let (j0, j1) = (
            Segments::index(p.y - r, self.lo.y, self.size, self.rows),
            Segments::index(p.y + r, self.lo.y, self.size, self.rows),
        );
        let mut best = f64::INFINITY;
        for j in j0..=j1 {
            for i in i0..=i1 {
                for &k in &self.cells[j * self.columns + i] {
                    let d = self.b[k] - self.a[k];
                    let t = ((p - self.a[k]).dot(d) / d.length_squared()).clamp(0.0, 1.0);
                    best = best.min((p - (self.a[k] + t * d)).length());
                }
            }
        }
        if best <= r { best } else { f64::INFINITY }
    }
}

fn key(p: P2<WorldSpace>) -> (u64, u64) {
    let bits = |x: f64| if x == 0.0 { 0 } else { x.to_bits() };
    (bits(p.x()), bits(p.y()))
}

/// "restriction: a contour's copy is cut to the footprint where it lands, exactly"
#[test]
fn a_contours_copy_is_cut_to_the_footprint_where_it_lands_exactly() {
    let landscapes: Vec<Landscape> =
        landscapes().into_iter().filter(|l| matches!(l.region, Region::Inside(_)) && l.has_graph()).collect();
    assert!(landscapes.len() >= 15, "the grid's landscapes on a footprint: {}", landscapes.len());
    let mut checked = 0;
    for l in &landscapes {
        for tiled in [true, false] {
            let identity = vec![Affine2::IDENTITY];
            let copies = if tiled { &l.tiles } else { &identity };
            // Untiled, only where the footprint cuts the domain, holding a
            // twentieth of it and leaving a twentieth.
            if !tiled {
                let d = l.surface.domain();
                let held = (0..400)
                    .filter(|k| {
                        l.region.contains(P2::new(
                            d.real.lo + d.real.length() * ((k % 20) as f64 + 0.5) / 20.0,
                            d.imag.lo + d.imag.length() * ((k / 20) as f64 + 0.5) / 20.0,
                        ))
                    })
                    .count();
                if !(20..=380).contains(&held) {
                    continue;
                }
            } else if l.tiles.len() == 1 {
                continue;
            }
            let Region::Inside(perimeter) = &l.region else { continue };
            let surface = &l.surface;
            let (real, imag) = (surface.domain().real, surface.domain().imag);
            let size = real.length().abs().max(imag.length().abs());
            let cell = (real.length().abs() / (surface.height.width - 1) as f64)
                .max(imag.length().abs() / (surface.height.height - 1) as f64);
            let tolerance = 1e-9 * size;
            let mut finite: Vec<f64> =
                surface.height.values.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v)).collect();
            finite.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = (finite[finite.len() / 10], finite[9 * finite.len() / 10].min(6.0));
            let levels: Vec<f64> = (1..=7).map(|k| lo + (hi - lo) * k as f64 / 8.0).collect();
            let band = (imag.lo + 0.3 * imag.length(), imag.lo + 0.7 * imag.length());
            let mut corners: Vec<Vec2> = perimeter.edges().iter().map(|e| e.start.v).collect();
            corners.push(perimeter.edges()[0].start.v);
            let outline = Segments::new(&[corners], cell);
            let cover = |w: Vec2| -> usize {
                l.tiles
                    .iter()
                    .filter(|tile| {
                        let Some(back) = tile.inverse() else { return false };
                        let q = back.apply(P3::new(w.x, w.y, 0.0));
                        q.x() >= real.lo.min(real.hi)
                            && q.x() <= real.lo.max(real.hi)
                            && q.y() >= imag.lo.min(imag.hi)
                            && q.y() <= imag.lo.max(imag.hi)
                    })
                    .count()
            };
            for (what, restriction, banded) in [
                ("the footprint", Restriction::Footprint, false),
                (
                    "the footprint and a band of Im z",
                    Restriction::Intersection(vec![
                        Restriction::Footprint,
                        Restriction::Band { axis: DomainAxis::Imag, lo: band.0, hi: band.1 },
                    ]),
                    true,
                ),
            ] {
                let derived = |r: Restriction| -> PolylineSet<WorldSpace> {
                    surface.derive(
                        &LayerSource::Contour {
                            field: ContourField::Magnitude,
                            levels: levels.clone(),
                            restriction: r,
                            faces: FaceSelector::Graph { tiled },
                            light: None,
                        },
                        HeightPolicy::Magnitude,
                        &l.region,
                        &l.tiles,
                        None,
                    )
                };
                let whole = derived(Restriction::All);
                let kept = derived(restriction);
                assert!(
                    whole.count() % copies.len() == 0,
                    "{}, {what}: the whole contours come a copy per tile",
                    l.name
                );
                let per_copy = whole.count() / copies.len();
                let whole_vertices: HashSet<(u64, u64)> = whole.vertices.iter().map(|v| key(v.xy())).collect();
                let runs: Vec<Vec<Vec2>> =
                    (0..kept.count()).map(|k| kept.path(k).iter().map(|v| v.xy().v).collect()).collect();
                let pieces = Segments::new(&runs, cell);
                let (mut outside, mut ends_off, mut misclipped) = (Vec::new(), Vec::new(), Vec::new());
                let (mut in_samples, mut out_samples) = (0, 0);
                for run in &runs {
                    for w in run.windows(2) {
                        for k in 0..=16 {
                            let q = w[0] + (w[1] - w[0]) * (k as f64 / 16.0);
                            if !l.region.contains(P2::from_vec(q)) && !(outline.distance(q, cell) <= tolerance) {
                                outside.push(q);
                            }
                        }
                    }
                    for e in [run[0], run[run.len() - 1]] {
                        if !whole_vertices.contains(&key(P2::from_vec(e))) && !(outline.distance(e, cell) <= tolerance)
                        {
                            ends_off.push(e);
                        }
                    }
                }
                for c in 0..whole.count() {
                    let tile = copies[c / per_copy];
                    let Some(back) = tile.inverse() else { continue };
                    let path: Vec<Vec2> = whole.path(c).iter().map(|v| v.xy().v).collect();
                    for w in path.windows(2) {
                        let (a, b) = (w[0], w[1]);
                        let ya = back.apply(P3::new(a.x, a.y, 0.0)).y();
                        let yb = back.apply(P3::new(b.x, b.y, 0.0)).y();
                        let in_band = |y: f64| !banded || (y > band.0 && y < band.1);
                        // Not where a vertex lies on the band's edge, which
                        // carried back and forth is a rounding either side.
                        let on_edge = |y: f64| banded && (y - band.0).abs().min((y - band.1).abs()) <= tolerance;
                        if !(in_band(ya) == in_band(yb) && !on_edge(ya) && !on_edge(yb)) {
                            continue;
                        }
                        for k in 0..16 {
                            let q = a + (b - a) * ((k as f64 + 0.5) / 16.0);
                            let d = outline.distance(q, cell);
                            if !((d.is_infinite() || d > 1e-6 * size) && (!tiled || cover(q) == 1)) {
                                continue;
                            }
                            let expected = l.region.contains(P2::from_vec(q)) && in_band(ya);
                            if expected {
                                in_samples += 1;
                            } else {
                                out_samples += 1;
                            }
                            let on_piece = pieces.distance(q, cell) <= tolerance;
                            if on_piece != expected {
                                misclipped.push((q, c / per_copy, on_piece));
                            }
                        }
                    }
                }
                assert!(
                    in_samples > 200
                        && out_samples > 200
                        && outside.is_empty()
                        && ends_off.is_empty()
                        && misclipped.is_empty(),
                    "{}, {}, kept to {what}: each copy cut to the footprint where it lands, exactly: \
                     {} points kept outside, {} of {} ends off the perimeter, {} of {in_samples} in and {out_samples} out \
                     misclipped: {:?} {:?} {:?}",
                    l.name,
                    if tiled { "tiled" } else { "untiled" },
                    outside.len(),
                    ends_off.len(),
                    2 * runs.len(),
                    misclipped.len(),
                    outside.first(),
                    ends_off.first(),
                    misclipped.first()
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "{checked} cases checked");
}
