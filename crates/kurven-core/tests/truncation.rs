//! `Sources/kurven-test/Truncation.swift`: the truncation edge, the pieces
//! `Mesh::truncation_edge_mesh` adds to the depth pass, and the vertex
//! `Surface::derive` carries a contour to. Swift's tolerances.

use kurven_core::grid::Grid2D;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, ContourField, Domain, HeightPolicy, Interval, LayerSource, PerimeterEdge, Region,
    Restriction, Truncation,
};
use kurven_core::mesh::Mesh;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{P2, WorldSpace};
use kurven_core::splitmix::SplitMix;
use kurven_core::surface::{ContourRefine, Surface};

const N: usize = 9;

fn domain() -> Domain {
    Domain { real: Interval::new(-1.0, 1.0), imag: Interval::new(-1.0, 1.0) }
}

/// Sample `k` of an `m x m` grid over [-1, 1]², rows indexing imag.
fn at(k: usize, m: usize) -> (f64, f64) {
    (-1.0 + 2.0 * (k % m) as f64 / (m - 1) as f64, -1.0 + 2.0 * (k / m) as f64 / (m - 1) as f64)
}

/// A spire on a coarse grid: `h = 3 - 4 r²`, so the truncation at 2 cuts it
/// in a disc of radius one half, two cells across. Every cell round the
/// truncation edge straddles it.
fn spire() -> Grid2D<f32> {
    Grid2D::new(
        N,
        N,
        domain(),
        (0..N * N)
            .map(|k| {
                let (x, y) = at(k, N);
                (3.0 - 4.0 * (x * x + y * y)) as f32
            })
            .collect(),
    )
}

const TRUNCATION_HEIGHT: f64 = 2.0;

fn surface() -> Surface {
    Surface::new(spire(), None, Truncation::Uniform(TRUNCATION_HEIGHT), false)
}

/// The height of the plane through `a`, `b`, `c` over `p`, where `p` is in
/// their triangle.
fn plane(a: Vec3, b: Vec3, c: Vec3, p: Vec2) -> Option<f64> {
    let d = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    let u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / d;
    let v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / d;
    let w = 1.0 - u - v;
    let eps = 1e-9;
    if !(u >= -eps && v >= -eps && w >= -eps) {
        return None;
    }
    Some(u * a.z + v * b.z + w * c.z)
}

#[test]
fn the_pieces_are_the_clamped_surface_cell_for_cell() {
    let grid = spire();
    let truncation_edge = Mesh::truncation_edge_mesh(&surface(), 1, &Region::Full, &[Affine2::IDENTITY]);
    assert!(!truncation_edge.is_empty(), "a truncated spire has a truncation edge");
    assert!(
        Mesh::truncation_edge_mesh(
            &Surface::new(grid.clone(), None, Truncation::None, false),
            1,
            &Region::Full,
            &[Affine2::IDENTITY]
        )
        .is_empty(),
        "and an untruncated one has none"
    );

    // The rasterizer's own triangle through a point: corners clamped, then
    // interpolated. And what the samples describe: interpolated, then
    // clamped. Drawn under a MAX blend with the truncation edge pieces, the
    // first must become the second.
    let mut rng = SplitMix::new(11);
    let (mut worst, mut tested, mut straddled) = (0.0f64, 0, 0);
    let cell = 2.0 / (N - 1) as f64;
    for _ in 0..4000 {
        let p = Vec2::new(rng.next(-1.0, 1.0), rng.next(-1.0, 1.0));
        let i = (((p.x + 1.0) / cell) as usize).min(N - 2);
        let j = (((p.y + 1.0) / cell) as usize).min(N - 2);
        let (x0, y0) = (-1.0 + i as f64 * cell, -1.0 + j as f64 * cell);
        let corner = |dx: usize, dy: usize| -> Vec3 {
            Vec3::new(x0 + dx as f64 * cell, y0 + dy as f64 * cell, f64::from(grid.at(i + dx, j + dy)))
        };
        let (c00, c10, c01, c11) = (corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1));
        let corners = [c00, c10, c01, c11];
        let straddles =
            corners.iter().any(|c| c.z > TRUNCATION_HEIGHT) && corners.iter().any(|c| c.z < TRUNCATION_HEIGHT);
        // The rasterizer's diagonal: (00, 10, 01) and (10, 11, 01).
        let tri = if plane(c00, c10, c01, p).is_some() { [c00, c10, c01] } else { [c10, c11, c01] };
        let expected = plane(tri[0], tri[1], tri[2], p).unwrap().min(TRUNCATION_HEIGHT);
        let clamp = |v: Vec3| Vec3::new(v.x, v.y, v.z.min(TRUNCATION_HEIGHT));
        let mut drawn = plane(clamp(tri[0]), clamp(tri[1]), clamp(tri[2]), p).unwrap();
        for t in &truncation_edge.triangles {
            let v = |k: i32| truncation_edge.vertices[k as usize].v;
            if let Some(z) = plane(v(t[0]), v(t[1]), v(t[2]), p) {
                drawn = drawn.max(z);
            }
        }
        worst = worst.max((drawn - expected).abs());
        tested += 1;
        if straddles {
            straddled += 1;
        }
    }
    assert!(straddled > 200, "the sample covers the truncation edge cells: {straddled} of {tested}");
    assert!(
        worst < 1e-9,
        "MAX(rasterized cell, truncation edge pieces) is min(interpolated, truncation) everywhere: worst {worst:.1e} over {tested} points"
    );

    // Every piece lies on the clamped surface: on the plateau, or on its
    // cell's interpolated height, and never above the truncation.
    let over = truncation_edge.vertices.iter().filter(|v| v.z() > TRUNCATION_HEIGHT + 1e-12).count();
    assert_eq!(over, 0, "no piece rises above the truncation: {over} vertices do");
}

#[test]
fn the_grid_is_the_rasterizers() {
    let surface = surface();
    // Per tile, once; outside the region, not at all.
    let one = Mesh::truncation_edge_mesh(&surface, 1, &Region::Full, &[Affine2::IDENTITY]);
    let shifted = Affine2 { a: 1.0, b: 0.0, tx: 5.0, c: 0.0, d: 1.0, ty: 0.0 };
    let two = Mesh::truncation_edge_mesh(&surface, 1, &Region::Full, &[Affine2::IDENTITY, shifted]);
    assert_eq!(two.triangles.len(), 2 * one.triangles.len(), "two tiles draw the truncation edge twice");
    assert!(
        two.vertices.iter().map(|v| v.x()).fold(f64::NEG_INFINITY, f64::max) > 4.0,
        "the second copy is where its tile puts it"
    );
    let edge =
        |a: (f64, f64), b: (f64, f64)| PerimeterEdge { start: P2::new(a.0, a.1), end: P2::new(b.0, b.1), density: 2 };
    let far = BoundaryPerimeter::new(vec![
        edge((3.0, 3.0), (4.0, 3.0)),
        edge((4.0, 3.0), (4.0, 4.0)),
        edge((4.0, 4.0), (3.0, 4.0)),
        edge((3.0, 4.0), (3.0, 3.0)),
    ]);
    assert!(
        Mesh::truncation_edge_mesh(&surface, 1, &Region::Inside(far), &[Affine2::IDENTITY]).is_empty(),
        "a region that excludes the spire has no truncation edge in it"
    );
    // Decimated by two, the grid is 5 x 5 over the same domain, and the
    // truncation edge is cut on those cells: coarser, and still below the
    // truncation.
    let coarse = Mesh::truncation_edge_mesh(&surface, 2, &Region::Full, &[Affine2::IDENTITY]);
    let grid: Vec<f64> = (0..5).map(|k| -1.0 + k as f64 * 0.5).collect();
    let on_grid = coarse
        .vertices
        .iter()
        .filter(|v| grid.iter().any(|g| (g - v.x()).abs() < 1e-9) || grid.iter().any(|g| (g - v.y()).abs() < 1e-9))
        .count();
    assert!(
        !coarse.is_empty() && on_grid > 0 && coarse.triangles.len() < one.triangles.len(),
        "step 2 cuts the truncation edge on the decimated grid: {} triangles",
        coarse.triangles.len()
    );
}

#[test]
fn a_contour_is_carried_to_the_truncation_and_only_by_a_refiner() {
    // A ramp `2x + 3` under a truncation of 3.5, so the truncation is crossed
    // at x = 1/4; and a phase grid equal to y, whose zero contour is the line
    // y = 0 straight up the ramp.
    let m = 5;
    let ramp = Grid2D::new(m, m, domain(), (0..m * m).map(|k| (2.0 * at(k, m).0 + 3.0) as f32).collect());
    let phase = Grid2D::new(m, m, domain(), (0..m * m).map(|k| at(k, m).1 as f32).collect());
    let ramp_truncation = 3.5;
    let s = Surface::new(ramp, Some(phase), Truncation::Uniform(ramp_truncation), false);
    let source = LayerSource::Contour {
        field: ContourField::Phase,
        levels: vec![0.0],
        restriction: Restriction::BelowTruncation,
        faces: FaceSelector::Graph { tiled: false },
        light: None,
    };
    let identity = ContourRefine::new(std::sync::Arc::new(|_, _, paths, _, _| paths.to_vec()), None);
    let tiles = [Affine2::IDENTITY];

    let plain = s.derive(&source, HeightPolicy::Magnitude, &Region::Full, &tiles, None);
    let carried = s.derive(&source, HeightPolicy::Magnitude, &Region::Full, &tiles, Some(&identity));
    assert!(plain.count() == 1 && carried.count() == 1, "one run each: {} and {}", plain.count(), carried.count());
    let top = |set: &kurven_core::polyline::PolylineSet<WorldSpace>| {
        set.vertices.iter().map(|v| v.z()).fold(f64::NEG_INFINITY, f64::max)
    };
    let (plain_top, carried_top) = (top(&plain), top(&carried));
    assert!(
        (plain_top - 3.0).abs() < 1e-6,
        "from the grid alone the run ends at its last sample below the truncation: {plain_top}"
    );
    assert!(
        (carried_top - ramp_truncation).abs() < 1e-12,
        "with a refiner it ends on the plateau exactly: {carried_top}"
    );
    let vertex = carried.vertices.iter().find(|v| (v.z() - ramp_truncation).abs() < 1e-12);
    assert!(
        vertex.is_some_and(|v| (v.x() - 0.25).abs() < 1e-9 && v.y().abs() < 1e-9),
        "and that vertex is on the truncation edge, where the interpolated height is the truncation: {vertex:?}"
    );
    assert_eq!(carried.vertices.len(), plain.vertices.len() + 1, "one vertex was added and none moved");
    assert!(carried.vertices.iter().all(|v| v.z() <= ramp_truncation + 1e-12), "nothing over the truncation survives");

    // A layer that does not keep below the truncation is not touched.
    let free = LayerSource::Contour {
        field: ContourField::Phase,
        levels: vec![0.0],
        restriction: Restriction::All,
        faces: FaceSelector::Graph { tiled: false },
        light: None,
    };
    let a = s.derive(&free, HeightPolicy::Magnitude, &Region::Full, &tiles, None);
    let b = s.derive(&free, HeightPolicy::Magnitude, &Region::Full, &tiles, Some(&identity));
    assert!(
        a.vertices.len() == b.vertices.len() && top(&a) == top(&b),
        "a layer that keeps everything gains no truncation vertex"
    );
}
