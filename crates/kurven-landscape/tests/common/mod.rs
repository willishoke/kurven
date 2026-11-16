//! What the face suites share: landscapes built on f as
//! `NativeLandscape.build(_:refine: true)` builds them (its grids and its
//! refiner; the bundle around them is the build's own, not ported yet), and
//! Swift's `Segments`, polylines bucketed for distance and even-odd
//! membership.

#![allow(dead_code)]

use kurven_core::grid::Grid2D;
use kurven_core::manifest::{Domain, Interval, RealBand, Truncation};
use kurven_core::simd::Vec2;
use kurven_core::surface::{ContourRefine, Surface};
use kurven_landscape::{ContourRefiner, NativeLandscape};
use kurven_math::Expression;

pub fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
    Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
}

pub fn bands(bands: &[(f64, f64)], beyond: f64) -> Truncation {
    Truncation::RealBands { bands: bands.iter().map(|&(below, height)| RealBand { below, height }).collect(), beyond }
}

/// A landscape on f: its surface, as `NativeLandscape.build` makes it from
/// the float32 grids of |f| and arg f, and the refiner's hook.
pub struct Built {
    pub surface: Surface,
    pub refine: ContourRefine,
    pub shape: kurven_landscape::GridShape,
}

impl Built {
    pub fn magnitude(&self) -> &kurven_core::surface::Magnitude {
        self.refine.magnitude.as_ref().expect("the refiner has |f|")
    }
}

/// `NativeLandscape.build(LandscapeRequest(expression:domain:resolution:
/// truncation:), refine: true)`, its surface and refiner.
pub fn build(expression: &str, domain: Domain, resolution: i64, truncation: Truncation) -> Built {
    let compiled = Expression::compile(expression).unwrap_or_else(|e| panic!("{expression}: {e:?}"));
    let samples = NativeLandscape::sample(&compiled, domain, resolution).expect("the domain samples");
    let s = samples.shape;
    let height = Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.magnitude));
    let phase = Grid2D::new(s.n_real, s.n_imag, domain, NativeLandscape::quantize(&samples.phase));
    let surface = Surface::new(height, Some(phase), truncation, false);
    let refine = ContourRefiner::new(compiled, domain, s.n_real, s.n_imag).refine_hook();
    Built { surface, refine, shape: s }
}

/// Polylines as segments, bucketed so that a point is met only with those
/// near it: the distance to them within a radius, and even-odd membership by
/// a ray along +x (Swift's `Segments`).
pub struct Segments {
    a: Vec<Vec2>,
    b: Vec<Vec2>,
    lo: Vec2,
    size: f64,
    columns: usize,
    rows: usize,
    cells: Vec<Vec<usize>>,
    /// By rows alone, for the ray: each segment once in a row it spans.
    strips: Vec<Vec<usize>>,
}

impl Segments {
    pub fn new(paths: &[Vec<Vec2>], size: f64) -> Segments {
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
        let count = |extent: f64| -> usize { (((extent / size).floor() as i64 + 1).clamp(1, 4096)) as usize };
        let (columns, rows) = (count(hi.x - lo.x), count(hi.y - lo.y));
        let mut cells = vec![Vec::new(); columns * rows];
        let mut strips = vec![Vec::new(); rows];
        for k in 0..a.len() {
            let (l, h) = (a[k].min(b[k]), a[k].max(b[k]));
            let (i0, i1) = (Segments::index(l.x, lo.x, size, columns), Segments::index(h.x, lo.x, size, columns));
            let (j0, j1) = (Segments::index(l.y, lo.y, size, rows), Segments::index(h.y, lo.y, size, rows));
            for j in j0..=j1 {
                strips[j].push(k);
                for i in i0..=i1 {
                    cells[j * columns + i].push(k);
                }
            }
        }
        Segments { a, b, lo, size, columns, rows, cells, strips }
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
    pub fn distance(&self, p: Vec2, r: f64) -> f64 {
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
                    let t = (p - self.a[k]).dot(d) / d.length_squared();
                    let t = 0.0f64.max(1.0f64.min(t));
                    best = best.min((p - (self.a[k] + t * d)).length());
                }
            }
        }
        if best <= r { best } else { f64::INFINITY }
    }

    /// Whether the segments wind round `p` an odd number of times: a ray
    /// along +x crosses an odd number of them, each half open in y.
    pub fn inside(&self, p: Vec2) -> bool {
        if self.a.is_empty() || !(p.y >= self.lo.y && p.y <= self.lo.y + self.rows as f64 * self.size) {
            return false;
        }
        let mut inside = false;
        for &k in &self.strips[Segments::index(p.y, self.lo.y, self.size, self.rows)] {
            let (a, b) = (self.a[k], self.b[k]);
            if (a.y > p.y) == (b.y > p.y) {
                continue;
            }
            if p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y) {
                inside = !inside;
            }
        }
        inside
    }
}

use kurven_core::bundle::KurvenBundle;
use kurven_core::face::Face;
use kurven_core::fixtures;
use kurven_core::json::Members;
use kurven_core::manifest::{Affine2, BoundaryPerimeter, PerimeterEdge, Region, Walls};
use kurven_core::mesh::Mesh;
use kurven_core::scene::Heightfield;
use kurven_core::spaces::P2;
use kurven_math::real;

/// A landscape as the faces are built from it, and as the drawing reads it
/// (Faces.swift's `Landscape`).
pub struct Landscape {
    pub name: String,
    pub surface: Surface,
    pub tiles: Vec<Affine2>,
    pub region: Region,
    pub walls: Walls,
    pub refine: Option<ContourRefine>,
    pub faces: Vec<Face>,
    /// Whether two copies meet at a tile boundary with different tops where
    /// the footprint holds both sides: the solid drops there.
    pub drops_at_a_tile_boundary: bool,
}

impl Landscape {
    pub fn new(
        name: &str,
        surface: Surface,
        tiles: Vec<Affine2>,
        region: Region,
        walls: Walls,
        refine: Option<ContourRefine>,
    ) -> Landscape {
        let faces = surface.faces(&tiles, &region, &walls, refine.as_ref());
        Landscape {
            name: name.to_string(),
            surface,
            tiles,
            region,
            walls,
            refine,
            faces,
            drops_at_a_tile_boundary: false,
        }
    }

    /// One tile, the whole region, no walls, no function.
    pub fn plain(name: &str, surface: Surface) -> Landscape {
        Landscape::new(name, surface, vec![Affine2::IDENTITY], Region::Full, Walls::None, None)
    }

    pub fn heightfield(&self) -> Heightfield {
        Heightfield::new(
            self.surface.clone(),
            Mesh::empty(),
            self.tiles.clone(),
            self.region.clone(),
            1,
            self.refine.clone(),
            self.walls.clone(),
        )
    }

    /// Whether any sample of the grid is under the truncation: a landscape
    /// all plateau has no graph to contour or clip.
    pub fn has_graph(&self) -> bool {
        let g = &*self.surface.height;
        (0..g.width).any(|i| {
            let h = self.surface.truncation_height_at(g.position(i, 0).x());
            (0..g.height).any(|j| f64::from(g.at(i, j)) < h)
        })
    }
}

pub fn affine(a: f64, b: f64, tx: f64, c: f64, d: f64, ty: f64) -> Affine2 {
    Affine2 { a, b, tx, c, d, ty }
}

pub fn footprint(corners: &[(f64, f64)], density: i64) -> BoundaryPerimeter {
    BoundaryPerimeter::new(
        (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density }
            })
            .collect(),
    )
}

/// The hatch fixture's grid and perimeter, and its truncations by name.
pub struct HatchFixture {
    pub grid: Grid2D<f32>,
    pub perimeter: BoundaryPerimeter,
    pub truncations: kurven_core::json::JsonObject,
}

pub fn hatch_fixture() -> HatchFixture {
    let index = fixtures::shared_json("hatch/index.json");
    let index = index.as_object().unwrap();
    let shape = index.value("shape", "hatch").unwrap().object("hatch.shape").unwrap();
    let grid = Grid2D::new(
        shape.int("nx", "hatch.shape").unwrap() as usize,
        shape.int("ny", "hatch.shape").unwrap() as usize,
        Domain::from_json(index.value("domain", "hatch").unwrap()).unwrap(),
        fixtures::shared_npy("hatch/height.npy").floats().unwrap(),
    );
    let perimeter = BoundaryPerimeter::from_json(index.value("perimeter", "hatch").unwrap()).unwrap();
    let truncations = index.value("truncation", "hatch").unwrap().object("hatch.truncations").unwrap().clone();
    HatchFixture { grid, perimeter, truncations }
}

impl HatchFixture {
    pub fn truncation(&self, name: &str) -> Truncation {
        Truncation::from_json(self.truncations.value(name, "hatch.truncations").unwrap()).unwrap()
    }
}

/// A landscape on f with the walls, tiles and region `NativeLandscape.build`
/// gives it: one tile, the whole region, walls on the domain's rectangle at
/// the grid's density.
pub fn built_landscape(name: &str, built: &Built, domain: Domain) -> Landscape {
    let density = NativeLandscape::wall_density(domain, built.shape).expect("a density");
    let perimeter = NativeLandscape::rectangle(domain, &density);
    Landscape::new(
        name,
        built.surface.clone(),
        vec![Affine2::IDENTITY],
        Region::Full,
        Walls::Perimeter { perimeter, base: 0.0 },
        Some(built.refine.clone()),
    )
}

/// Faces.swift's `faceLandscapes()`: the contract fixtures, the hatch
/// fixture under its truncations, banded on notched footprints, with a band
/// boundary on its far edge, in reflected, turned and translated tiles, the
/// towers on f, and Γ banded on f. The contract bundles' faces are built
/// from their occluder as `Scene(bundle:preset:)`'s geometry builds them.
pub fn face_landscapes() -> Vec<Landscape> {
    let mut landscapes: Vec<Landscape> = Vec::new();
    let mut contract: Vec<_> = std::fs::read_dir(fixtures::shared("contract"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            name.ends_with(".kurven") && !name.starts_with("polyhedron_") && !name.starts_with("surface")
        })
        .collect();
    contract.sort();
    for url in contract {
        let bundle = KurvenBundle::read(&url).unwrap_or_else(|e| panic!("{}: {e}", url.display()));
        let o = &bundle.manifest.occluder;
        landscapes.push(Landscape::new(
            &url.file_name().unwrap().to_string_lossy(),
            bundle.surface.clone(),
            o.tiles.clone(),
            o.region.clone(),
            o.walls.clone(),
            bundle.refine.clone(),
        ));
    }
    let hatch = hatch_fixture();
    let grid = hatch.grid.clone();
    let perimeter = hatch.perimeter.clone();
    let walled = |p: &BoundaryPerimeter, base: f64| Walls::Perimeter { perimeter: p.clone(), base };
    let id = || vec![Affine2::IDENTITY];
    for name in ["uniform", "bands"] {
        let surface = Surface::new(grid.clone(), None, hatch.truncation(name), false);
        landscapes.push(Landscape::new(
            &format!("the hatch fixture, {name}"),
            surface,
            id(),
            Region::Full,
            walled(&perimeter, 0.0),
            None,
        ));
    }
    let banded = Surface::new(grid.clone(), None, hatch.truncation("bands"), false);
    let low = Surface::new(grid.clone(), None, bands(&[(-1.0, 0.3), (0.4, 0.7)], 0.45), false);
    let notches: Vec<(&str, &Surface, BoundaryPerimeter)> = vec![
        (
            "bands, on a notch from Re z = 0.5 to 1.5",
            &banded,
            footprint(
                &[(-2.0, -1.0), (0.5, -1.0), (0.5, 0.0), (1.5, 0.0), (1.5, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)],
                9,
            ),
        ),
        (
            "low bands, on a notch from one boundary to the other",
            &low,
            footprint(
                &[
                    (-2.0, -1.0),
                    (-1.0, -1.0),
                    (-1.0, 0.6),
                    (0.4, 0.6),
                    (0.4, -1.0),
                    (2.5, -1.0),
                    (2.5, 1.5),
                    (-2.0, 1.5),
                ],
                9,
            ),
        ),
        (
            "low bands, on notches from the left edge to one boundary and from the other to 1.5",
            &low,
            footprint(
                &[
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
                9,
            ),
        ),
    ];
    for (name, surface, notched) in notches {
        landscapes.push(Landscape::new(
            &format!("the hatch fixture, {name}"),
            surface.clone(),
            id(),
            Region::Inside(notched.clone()),
            walled(&notched, -0.5),
            None,
        ));
    }
    let edge_band = Surface::new(grid.clone(), None, bands(&[(-1.0, 1.2), (2.5, 2.5)], 0.05), false);
    landscapes.push(Landscape::new(
        "the hatch fixture, a band boundary on the far edge",
        edge_band.clone(),
        id(),
        Region::Full,
        walled(&perimeter, 0.0),
        None,
    ));
    landscapes.push(Landscape::plain("the hatch fixture, a band boundary on the far edge, no walls", edge_band));
    let reflected = affine(-1.0, 0.0, 5.0, 0.0, 1.0, 0.0);
    let turned = affine(0.0, -1.0, 10.0, 1.0, 0.0, 0.3);
    let pi = std::f64::consts::PI;
    let reflected_by_angle = affine(real::cos(pi), real::sin(pi), 5.0, real::sin(pi), -real::cos(pi), 0.0);
    let turned_by_angle =
        affine(real::cos(pi / 2.0), -real::sin(pi / 2.0), 10.0, real::sin(pi / 2.0), real::cos(pi / 2.0), 0.3);
    let higher_r = [(4.6, -1.0), (5.9, -1.0), (5.9, 1.5), (4.6, 1.5)];
    let lower_r = [(3.0, -1.0), (4.6, -1.0), (4.6, 1.5), (3.0, 1.5)];
    let higher_t = [(8.6, -0.5), (10.9, -0.5), (10.9, 0.7), (8.6, 0.7)];
    let lower_t = [(8.6, 0.7), (10.9, 0.7), (10.9, 2.7), (8.6, 2.7)];
    for (name, tile, corners) in [
        ("reflected, the footprint on the higher band's side", reflected, higher_r),
        ("reflected, the footprint on the lower band's side", reflected, lower_r),
        ("turned, the footprint on the higher band's side", turned, higher_t),
        ("turned, the footprint on the lower band's side", turned, lower_t),
        ("reflected by an angle, the footprint on the higher band's side", reflected_by_angle, higher_r),
        ("reflected by an angle, the footprint on the lower band's side", reflected_by_angle, lower_r),
        ("turned by an angle, the footprint on the higher band's side", turned_by_angle, higher_t),
        ("turned by an angle, the footprint on the lower band's side", turned_by_angle, lower_t),
    ] {
        let walls = footprint(&corners, 9);
        landscapes.push(Landscape::new(
            &format!("the hatch fixture, low bands, a wall on a boundary's image in a tile {name}"),
            low.clone(),
            vec![Affine2::IDENTITY, tile],
            Region::Inside(walls.clone()),
            walled(&walls, -0.5),
            None,
        ));
    }
    landscapes.push(Landscape::new(
        "the hatch fixture, uniform, a second tile sheared and stretched",
        Surface::new(grid.clone(), None, Truncation::Uniform(2.0), false),
        vec![Affine2::IDENTITY, affine(1.0, 0.5, 5.0, 0.2, 2.0, 0.0)],
        Region::Full,
        Walls::None,
        None,
    ));
    let notched: Vec<(f64, f64)> =
        [(-2.0, -1.0), (-1.0, -1.0), (-1.0, 0.6), (0.4, 0.6), (0.4, -1.0), (2.5, -1.0), (2.5, 1.5), (-2.0, 1.5)]
            .iter()
            .map(|&(x, y)| {
                let q = turned.apply(kurven_core::spaces::P3::new(x, y, 0.0));
                (q.x(), q.y())
            })
            .collect();
    let notched = footprint(&notched, 9);
    landscapes.push(Landscape::new(
        "the hatch fixture, low bands, on a notch from one boundary to the other, in a tile turned",
        low.clone(),
        vec![Affine2::IDENTITY, turned],
        Region::Inside(notched.clone()),
        walled(&notched, -0.5),
        None,
    ));
    let reflections = vec![
        Affine2::IDENTITY,
        affine(-1.0, 0.0, 5.0, 0.0, 1.0, 0.0),
        affine(1.0, 0.0, 0.0, 0.0, -1.0, 3.0),
        affine(-1.0, 0.0, 5.0, 0.0, -1.0, 3.0),
    ];
    let across = footprint(&[(-1.5, -0.5), (5.5, -0.8), (6.6, 1.2), (6.2, 3.6), (1.0, 3.3), (-1.8, 2.0)], 9);
    landscapes.push(Landscape::new(
        "the hatch fixture, low bands, in four tiles, on a footprint across them",
        low.clone(),
        reflections.clone(),
        Region::Inside(across.clone()),
        walled(&across, -0.5),
        None,
    ));
    landscapes.push(Landscape::new(
        "the hatch fixture, uniform, in four tiles",
        Surface::new(grid.clone(), None, Truncation::Uniform(2.0), false),
        reflections,
        Region::Full,
        Walls::None,
        None,
    ));
    let unit = domain(0.0, 1.0, 0.0, 1.0);
    let parabola_grid = Grid2D::new(
        21,
        21,
        unit,
        (0..441)
            .map(|k| {
                let t = 2.0 * (k % 21) as f64 / 20.0 - 1.0;
                (3.0 * real::pow(t, 2.0) + 0.2) as f32
            })
            .collect(),
    );
    let parabola = Surface::new(parabola_grid, None, bands(&[(0.5, 1.0)], 2.0), false);
    for (tile_name, tile) in
        [("beside it", affine(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)), ("turned", affine(0.0, 1.0, 1.0, -1.0, 0.0, 1.0))]
    {
        for (side, corners) in [
            ("second", [(1.0, 0.2), (1.8, 0.2), (1.8, 0.8), (1.0, 0.8)]),
            ("first", [(0.2, 0.2), (1.0, 0.2), (1.0, 0.8), (0.2, 0.8)]),
        ] {
            let walls = footprint(&corners, 9);
            landscapes.push(Landscape::new(
                &format!("a wall on the tile boundary with a tile {tile_name}, the footprint in the {side}"),
                parabola.clone(),
                vec![Affine2::IDENTITY, tile],
                Region::Inside(walls.clone()),
                walled(&walls, 0.0),
                None,
            ));
        }
    }
    let flat = Surface::new(Grid2D::new(21, 21, unit, vec![10.0; 441]), None, bands(&[(0.5, 1.0)], 2.0), false);
    let cornered = footprint(&[(1.0, 0.5), (1.0, 0.1), (1.8, 0.1), (1.8, 0.9), (1.5, 0.9)], 9);
    landscapes.push(Landscape::new(
        "|f| = 10 at 1 | 2, a tile turned, a corner on the tile boundary where its riser ends",
        flat.clone(),
        vec![Affine2::IDENTITY, affine(0.0, 1.0, 1.0, -1.0, 0.0, 1.0)],
        Region::Inside(cornered.clone()),
        walled(&cornered, 0.0),
        None,
    ));
    let slope = 1e-8;
    let leaving = footprint(&[(0.5, 0.9), (0.5 - 0.8 * slope, 0.1), (0.9, 0.1), (0.9, 0.9)], 9);
    landscapes.push(Landscape::new(
        "|f| = 10 at 1 | 2, a wall leaving the boundary at a slope of 1e-8",
        flat,
        id(),
        Region::Inside(leaving.clone()),
        walled(&leaving, 0.0),
        None,
    ));
    let gamma_domain = domain(-4.5, 4.5, -2.5, 2.5);
    let gamma = build("gamma(z)", gamma_domain, 160, bands(&[(-3.5, 2.5), (-2.5, 3.0), (-1.5, 4.0), (0.5, 5.5)], 4.0));
    let w = domain(-0.5, 1.5, -1.0, 1.0);
    let tower = build("1/(z - 0.5)", w, 32, bands(&[(0.5, 4.0)], 2.0));
    let half_notched = footprint(&[(0.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-0.5, 1.0), (-0.5, 0.0), (0.5, 0.0)], 9);
    landscapes.push(Landscape::new(
        "a tower banded at 4 | 2, on a footprint holding one side below Im z = 0, on f",
        tower.surface.clone(),
        id(),
        Region::Inside(half_notched.clone()),
        walled(&half_notched, -0.5),
        Some(tower.refine.clone()),
    ));
    for (name, tile, corners) in [
        ("reflected", affine(-1.0, 0.0, 3.0, 0.0, 1.0, 0.0), vec![(-0.3, -0.2), (2.9, -0.2), (3.2, 0.35), (0.1, 0.3)]),
        (
            "turned",
            affine(0.0, -1.0, 2.5, 1.0, 0.0, 0.0),
            vec![(-0.3, -0.3), (2.2, -0.3), (2.7, -0.25), (2.7, 1.2), (2.25, 1.25), (1.6, 0.3), (0.1, 0.3)],
        ),
    ] {
        let across = footprint(&corners, 9);
        landscapes.push(Landscape::new(
            &format!(
                "a tower banded at 4 | 2 in two tiles, the second {name}, walls across the boundary in both, on f"
            ),
            tower.surface.clone(),
            vec![Affine2::IDENTITY, tile],
            Region::Inside(across.clone()),
            walled(&across, -0.5),
            Some(tower.refine.clone()),
        ));
    }
    for (degrees, tx, x) in [(35.0f64, 13.0, 13.21), (20.0, 7.1, 7.37)] {
        let (c, s) = (real::cos(degrees * pi / 180.0), real::sin(degrees * pi / 180.0));
        let walls = footprint(&[(x - 0.4, -0.5), (x, -0.5), (x, 1.0), (x - 0.4, 1.0)], 9);
        landscapes.push(Landscape::new(
            &format!(
                "a tower banded at 4 | 2 in a tile turned by {}°, a vertical wall across the boundary, on f",
                degrees as i64
            ),
            tower.surface.clone(),
            vec![Affine2::IDENTITY, affine(c, -s, tx, s, c, 0.0)],
            Region::Inside(walls.clone()),
            walled(&walls, -0.5),
            Some(tower.refine.clone()),
        ));
    }
    let low_tower = build("1/(z - 0.5)", w, 32, bands(&[(0.5, 0.5)], 2.0));
    let across_the_boundary = footprint(&[(0.95, 0.0), (1.95, 1.0), (1.3, 0.95), (0.6, 0.2)], 8);
    landscapes.push(Landscape::new(
        "a tower banded at 0.5 | 2 in a tile turned a quarter, a wall across the tile boundary, on f",
        low_tower.surface.clone(),
        vec![Affine2::IDENTITY, affine(0.0, -1.0, 2.5, 1.0, 0.0, 0.0)],
        Region::Inside(across_the_boundary.clone()),
        walled(&across_the_boundary, -0.5),
        Some(low_tower.refine.clone()),
    ));
    let turned_beside = affine(0.0, 1.0, 1.0, -1.0, 0.0, 1.0);
    let across_the_tile_boundary = footprint(&[(0.2, 0.2), (1.8, 0.2), (1.8, 0.8), (0.2, 0.8)], 9);
    let mut boundary_drops = vec![
        Landscape::new(
            "copies meeting with different tops, a tile turned, on a footprint across the tile boundary, on the grid",
            parabola.clone(),
            vec![Affine2::IDENTITY, turned_beside],
            Region::Inside(across_the_tile_boundary.clone()),
            walled(&across_the_tile_boundary, 0.0),
            None,
        ),
        Landscape::new(
            "copies meeting with different tops, a tile turned, on the grid",
            parabola.clone(),
            vec![Affine2::IDENTITY, turned_beside],
            Region::Full,
            Walls::None,
            None,
        ),
    ];
    let reviewers_tower = build("1/(z - 1.4)", w, 32, bands(&[(1.45, 0.5)], 1.0));
    let turned_at = affine(0.0, 1.0, 2.5, -1.0, 0.0, 0.5);
    for (name, corners) in [
        ("across the tile boundary", [(0.2, -0.5), (2.8, -0.5), (2.8, 0.5), (0.2, 0.5)]),
        ("in the second tile, a wall down the tile boundary", [(1.5, -0.5), (2.8, -0.5), (2.8, 0.5), (1.5, 0.5)]),
    ] {
        let walls = footprint(&corners, 17);
        boundary_drops.push(Landscape::new(
            &format!("the reviewer's tower at 0.5 | 1, a tile turned, on a footprint {name}, on f"),
            reviewers_tower.surface.clone(),
            vec![Affine2::IDENTITY, turned_at],
            Region::Inside(walls.clone()),
            walled(&walls, -0.5),
            Some(reviewers_tower.refine.clone()),
        ));
    }
    for l in &mut boundary_drops {
        if !l.name.contains("in the second tile") {
            l.drops_at_a_tile_boundary = true;
        }
    }
    landscapes.extend(boundary_drops);
    let stepped =
        Surface::new(Grid2D::new(21, 21, unit, vec![10.0; 441]), None, bands(&[(0.5, 1.0), (0.9, 2.0)], 0.5), false);
    let mut more = vec![Landscape::new(
        "|f| = 10 at 1 | 2 | 1/2, a tile turned, the higher copy stepping along the tile boundary at a grid node",
        stepped,
        vec![Affine2::IDENTITY, turned_beside],
        Region::Full,
        Walls::None,
        None,
    )];
    let beside = |x: f64| affine(1.0, 0.0, x, 0.0, 1.0, 0.0);
    for (lo, hi) in [(0.2, 0.8), (0.23, 0.77)] {
        let row = footprint(&[(0.2, lo), (2.8, lo), (2.8, hi), (0.2, hi)], 9);
        more.push(Landscape::new(
            &format!("three tiles in a row, the middle turned, on a footprint from Im z = {lo} to {hi}, on the grid"),
            parabola.clone(),
            vec![Affine2::IDENTITY, turned_beside, beside(2.0)],
            Region::Inside(row.clone()),
            walled(&row, 0.0),
            None,
        ));
    }
    let from_the_crossing = footprint(&[(0.2, 0.12), (1.8, 0.12), (1.8, 0.8), (0.2, 0.8)], 9);
    more.push(Landscape::new(
        "copies meeting with different tops, a tile turned, a footprint from Im z = 0.12 where the tops cross, on the grid",
        parabola.clone(),
        vec![Affine2::IDENTITY, turned_beside],
        Region::Inside(from_the_crossing.clone()),
        walled(&from_the_crossing, 0.0),
        None,
    ));
    for l in &mut more {
        l.drops_at_a_tile_boundary = true;
    }
    let square = footprint(&[(0.2, 0.2), (1.8, 0.2), (1.8, 1.8), (0.2, 1.8)], 9);
    let translations = vec![
        Affine2::IDENTITY,
        beside(1.0),
        affine(1.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        affine(1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
    ];
    let mut four = Landscape::new(
        "four tiles by translation, on a footprint over all four, on the grid",
        parabola.clone(),
        translations.clone(),
        Region::Inside(square.clone()),
        walled(&square, 0.0),
        None,
    );
    four.drops_at_a_tile_boundary = true;
    more.push(four);
    let mut reversed = Landscape::new(
        "four tiles by translation, listed the other way, on the grid",
        parabola.clone(),
        translations.into_iter().rev().collect(),
        Region::Full,
        Walls::None,
        None,
    );
    reversed.drops_at_a_tile_boundary = true;
    more.push(reversed);
    let round = Surface::new(
        Grid2D::new(
            21,
            21,
            unit,
            (0..441)
                .map(|k| {
                    let (x, y) = ((k % 21) as f64 / 20.0, (k / 21) as f64 / 20.0);
                    (1.0 / ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) + 0.01).sqrt()) as f32
                })
                .collect(),
        ),
        None,
        Truncation::Uniform(100.0),
        false,
    );
    let (c4, s4, c2, s2) = (real::cos(pi / 2.0), real::sin(pi / 2.0), real::cos(pi), real::sin(pi));
    for (name, tile) in [
        ("a quarter", affine(c4, s4, 1.0, -s4, c4, 1.0)),
        ("a half", affine(c2, -s2, 1.0 - c2 + 0.5 * s2, s2, c2, 0.5 - s2 - 0.5 * c2)),
    ] {
        more.push(Landscape::new(
            &format!("|f| round about the centre, a tile turned {name} from cos and sin, on the grid"),
            round.clone(),
            vec![Affine2::IDENTITY, tile],
            Region::Full,
            Walls::None,
            None,
        ));
    }
    let millionth = Surface::new(
        Grid2D::new(
            21,
            21,
            unit,
            (0..441)
                .map(|k| {
                    let x = (k % 21) as f64 / 20.0;
                    (1.0 + x * (1.0 - x) + if k % 21 == 0 { 1e-6 } else { 0.0 }) as f32
                })
                .collect(),
        ),
        None,
        Truncation::Uniform(100.0),
        false,
    );
    let mut differing = Landscape::new(
        "copies whose tops differ by a millionth at a tile boundary, the second beside the first, on the grid",
        millionth,
        vec![Affine2::IDENTITY, beside(1.0)],
        Region::Full,
        Walls::None,
        None,
    );
    differing.drops_at_a_tile_boundary = true;
    more.push(differing);
    let pole = build("1/(z - 1.6)", w, 32, Truncation::Uniform(20.0));
    let mut peaked = Landscape::new(
        "a pole a tenth outside the tile boundary, a tile turned, on f",
        pole.surface.clone(),
        vec![Affine2::IDENTITY, affine(0.0, 1.0, 2.5, -1.0, 0.0, 0.5)],
        Region::Full,
        Walls::None,
        Some(pole.refine.clone()),
    );
    peaked.drops_at_a_tile_boundary = true;
    more.push(peaked);
    landscapes.extend(more);
    landscapes.push(built_landscape("Γ, banded, on f", &gamma, gamma_domain));
    landscapes
}
