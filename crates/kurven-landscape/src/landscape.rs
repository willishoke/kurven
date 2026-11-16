//! A landscape sampled here: `kurven.landscape`, natively. The port of
//! `Sources/KurvenLandscape/Landscape.swift`.
//!
//! The Python module's contract is that a landscape is *derived* -- an
//! expression, a rectangle, a resolution and a truncation are the whole of
//! what it is, and every layer of its bundle is a description of curves rather
//! than the curves. That is what let the frontend restyle without asking Python.
//! This is the other half of the same idea: with the evaluator in
//! `kurven-math`, the sampling does not need Python either, so a landscape is a
//! value this process builds from a `LandscapeRequest` in the time it takes to
//! evaluate f on a grid.
//!
//! The rules were ported line for line, and when the Python builder was
//! retired the bundles in `tests/fixtures/expr` were byte-identical from both
//! (but for the git sha); they are golden now. Where Python rounds (`round()`
//! is half-to-even), this rounds the same way; where numpy's `linspace`
//! computes `start + i * step`, so does this.

use std::cmp::Ordering;
use std::fmt;
use std::path::PathBuf;

use kurven_core::bundle::KurvenBundle;
use kurven_core::grid::Grid2D;
use kurven_core::hatching::{FaceSelector, Hatching};
use kurven_core::json::{JsonObject, JsonValue};
use kurven_core::manifest::{
    Affine2, BoundaryPerimeter, CameraPreset, ContourField, Domain, DomainAxis, GridRef, HeightPolicy, LayerRole,
    LayerSource, LayerSpec, Manifest, ObliqueProjection, Occluder, PerimeterEdge, Provenance, Region, Restriction,
    Truncation, Walls,
};
use kurven_core::npy::NpyDType;
use kurven_core::spaces::{P2, WorldSpace};
use kurven_core::surface::Surface;
use kurven_math::{Compiled, Complex, Expression, ExpressionError, real, swift};

use crate::catalog::LandscapeRequest;
use crate::refine::ContourRefiner;

/// `(nReal, nImag)`: samples along Re z and along Im z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridShape {
    pub n_real: usize,
    pub n_imag: usize,
}

/// The landscape's numbers and rules (`NativeLandscape`, a namespace).
pub enum NativeLandscape {}

/// Why a landscape cannot be built (`NativeLandscape::build`).
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// The expression does not parse or compile (Swift's thrown error).
    Expression(ExpressionError),
    /// A domain whose side is not a number. Swift's rounding (`pyRound`)
    /// traps on it; the port refuses it.
    Domain(Domain),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Expression(e) => write!(f, "{e}"),
            BuildError::Domain(d) => write!(f, "a domain whose side is not a number: {d:?}"),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<ExpressionError> for BuildError {
    fn from(e: ExpressionError) -> BuildError {
        BuildError::Expression(e)
    }
}

/// The samples: |f| and arg f on the grid, with every non-finite value
/// pushed to `huge` and every value past it scaled back to it, as
/// `kurven.landscape.evaluator` does. Rows index imag, columns real.
#[derive(Debug, Clone, PartialEq)]
pub struct Samples {
    pub magnitude: Vec<f64>,
    pub phase: Vec<f64>,
    pub shape: GridShape,
}

impl NativeLandscape {
    /// Samples along the longer side of the domain.
    pub const DEFAULT_RESOLUTION: i64 = 600;
    /// The occluder's heightfield is decimated to about this many samples a side.
    pub const OCCLUDER_RESOLUTION: i64 = 800;
    /// Where a non-finite sample is put instead: past any truncation, and still a number.
    pub const HUGE: f64 = 1e12;

    pub const PROJECTION: ObliqueProjection =
        ObliqueProjection { shear: 0.5, x_angle: -55.0, z_angle: -90.0, flip_x: true, y_scale: None };
    pub const DEFAULT_DEPTH_TOLERANCE: f64 = 0.02;
    pub const DEFAULT_BUFFER: i64 = 4000;

    /// Phase contours: every 90 degrees major.
    pub fn phase_major() -> Vec<f64> {
        NativeLandscape::linspace(-core::f64::consts::PI, core::f64::consts::PI, 5)
    }

    /// Every 30 degrees minor, less the majors.
    pub fn phase_minor() -> Vec<f64> {
        let major = NativeLandscape::phase_major();
        NativeLandscape::linspace(-core::f64::consts::PI, core::f64::consts::PI, 13)
            .into_iter()
            .filter(|v| !major.iter().any(|m| (v - m).abs() < 1e-12))
            .collect()
    }

    /// numpy's `linspace`: `start + i * step`, with the last sample exactly `stop`.
    pub fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        if n <= 1 {
            return if n == 1 { vec![lo] } else { Vec::new() };
        }
        let step = (hi - lo) / (n - 1) as f64;
        (0..n).map(|i| if i == n - 1 { hi } else { lo + i as f64 * step }).collect()
    }

    /// Python's `round()`: half to even. None where Swift's `Int(_:)` traps
    /// (a value that is not finite or does not fit).
    pub fn py_round(x: f64) -> Option<i64> {
        swift::int(x.round_ties_even())
    }

    // MARK: the derived numbers

    /// `(nReal, nImag)` for a resolution given along the longer side, with
    /// square cells. None for a domain whose sides are not numbers, where
    /// Swift's rounding traps.
    pub fn grid_shape(domain: Domain, resolution: i64) -> Option<GridShape> {
        let dr = (domain.real.hi - domain.real.lo).abs();
        let di = (domain.imag.hi - domain.imag.lo).abs();
        let longest = swift::max(dr, di);
        let shape = |a: i64, b: i64| GridShape { n_real: a as usize, n_imag: b as usize };
        if longest <= 0.0 {
            return Some(shape(resolution.max(2), resolution.max(2)));
        }
        let n_real = NativeLandscape::py_round(resolution as f64 * dr / longest)?.max(8);
        let n_imag = NativeLandscape::py_round(resolution as f64 * di / longest)?.max(8);
        Some(shape(n_real, n_imag))
    }

    pub fn cell_size(domain: Domain, shape: GridShape) -> (f64, f64) {
        (
            (domain.real.hi - domain.real.lo).abs() / (shape.n_real.max(2) - 1) as f64,
            (domain.imag.hi - domain.imag.lo).abs() / (shape.n_imag.max(2) - 1) as f64,
        )
    }

    /// numpy's default quantile: linear interpolation between order statistics.
    pub fn quantile(sorted: &[f64], q: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let position = q * (sorted.len() - 1) as f64;
        // q is in [0, 1], so the floor is a valid index; Swift's `Int` of it
        // cannot trap.
        let lo = swift::int(position.floor()).unwrap_or(0).max(0) as usize;
        let hi = (lo + 1).min(sorted.len() - 1);
        let t = position - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * t
    }

    /// How far up to cut, read off the samples: a spire is a top more than
    /// three times the 99th percentile, cut at a round number near it.
    pub fn default_truncation(magnitudes: &[f64]) -> Truncation {
        let finite = sorted_finite(magnitudes);
        let Some(&top) = finite.last() else { return Truncation::None };
        let q = NativeLandscape::quantile(&finite, 0.99);
        if q <= 0.0 || top <= 3.0 * q {
            return Truncation::None;
        }
        Truncation::Uniform(LandscapeStyle::nice(q, false))
    }

    /// The highest |f| worth a contour: the truncation when there is one, else a
    /// robust maximum of the samples.
    pub fn ceiling(truncation: &Truncation, magnitudes: &[f64]) -> f64 {
        match truncation {
            Truncation::Uniform(z) => *z,
            Truncation::RealBands { bands, beyond } => {
                let mut tops: Vec<f64> = bands.iter().map(|b| b.height).collect();
                if beyond.is_finite() {
                    tops.push(*beyond);
                }
                sequence_max(&tops).unwrap_or(1.0)
            }
            Truncation::None => {
                let finite = sorted_finite(magnitudes);
                if finite.is_empty() { 1.0 } else { NativeLandscape::quantile(&finite, 0.999) }
            }
        }
    }

    /// The domain's boundary as a closed traversal: front, right, back, left,
    /// so edge 0 is the real axis at the near side.
    pub fn rectangle(domain: Domain, density: &[i64]) -> BoundaryPerimeter {
        let (r0, r1) = (domain.real.lo, domain.real.hi);
        let (i0, i1) = (domain.imag.lo, domain.imag.hi);
        let p = |x: f64, y: f64| P2::<WorldSpace>::new(x, y);
        let corners = [(p(r0, i0), p(r1, i0)), (p(r1, i0), p(r1, i1)), (p(r1, i1), p(r0, i1)), (p(r0, i1), p(r0, i0))];
        BoundaryPerimeter::new(
            corners
                .iter()
                .zip(density)
                .map(|(&(start, end), &density)| PerimeterEdge { start, end, density })
                .collect(),
        )
    }

    /// Curtain samples per edge: one per grid cell along that edge. None
    /// where Swift's rounding traps (a domain side that is not a number).
    pub fn wall_density(domain: Domain, shape: GridShape) -> Option<Vec<i64>> {
        let (dx, dy) = NativeLandscape::cell_size(domain, shape);
        let along_real =
            (NativeLandscape::py_round((domain.real.hi - domain.real.lo).abs() / swift::max(dx, 1e-12))? + 1).max(2);
        let along_imag =
            (NativeLandscape::py_round((domain.imag.hi - domain.imag.lo).abs() / swift::max(dy, 1e-12))? + 1).max(2);
        Some(vec![along_real.min(4096), along_imag.min(4096), along_real.min(4096), along_imag.min(4096)])
    }

    /// The whole drawing, as descriptions: `kurven.landscape.default_layers`.
    /// Swift's `phase` defaults to true.
    pub fn default_layers(
        domain: Domain,
        truncation: &Truncation,
        ceiling: f64,
        spacing: Option<f64>,
        phase: bool,
    ) -> Vec<LayerSpec> {
        let spacing = spacing.unwrap_or_else(|| LandscapeStyle::hatch_spacing(domain));
        let (mut major, mut minor) = LandscapeStyle::magnitude_levels(ceiling);
        if *truncation != Truncation::None {
            major.retain(|&m| m < ceiling - 1e-9);
            minor.retain(|&m| m < ceiling - 1e-9);
        }
        let edges = vec![0, 1, 2, 3];
        let contour = |field: ContourField, levels: Vec<f64>| LayerSource::Contour {
            field,
            levels,
            restriction: Restriction::BelowTruncation,
            faces: FaceSelector::Graph { tiled: false },
            light: None,
        };
        let mut layers = vec![LayerSpec::new(
            "mag_major",
            LayerRole::Magnitude,
            contour(ContourField::Magnitude, major),
            0.4,
            HeightPolicy::Level,
        )];
        if phase {
            layers.push(LayerSpec::new(
                "ang_major",
                LayerRole::Phase,
                contour(ContourField::Phase, NativeLandscape::phase_major()),
                0.4,
                HeightPolicy::Magnitude,
            ));
        }
        layers.push(LayerSpec::new(
            "mag_minor",
            LayerRole::Magnitude,
            contour(ContourField::Magnitude, minor),
            0.15,
            HeightPolicy::Level,
        ));
        if phase {
            layers.push(LayerSpec::new(
                "ang_minor",
                LayerRole::Phase,
                contour(ContourField::Phase, NativeLandscape::phase_minor()),
                0.15,
                HeightPolicy::Magnitude,
            ));
        }
        layers.extend([
            LayerSpec::new(
                "truncation_edge",
                LayerRole::Scaffold,
                LayerSource::TruncationEdge { tiled: false },
                0.4,
                HeightPolicy::Level,
            ),
            LayerSpec::new(
                "plateau_hatch",
                LayerRole::Scaffold,
                LayerSource::plateau_hatching(DomainAxis::Real, spacing, false),
                0.2,
                HeightPolicy::Level,
            ),
            LayerSpec::new(
                "wall_edges",
                LayerRole::Scaffold,
                LayerSource::WallEdges { edges: edges.clone(), base: 0.0 },
                0.3,
                HeightPolicy::Surface,
            ),
            LayerSpec::new(
                "wall_hatch",
                LayerRole::Scaffold,
                LayerSource::wall_hatching(edges.clone(), spacing, false, 0.0, 0.0),
                0.25,
                HeightPolicy::Surface,
            ),
            // The outline and inner fold curves, derived per camera: the
            // bounding curves along the curved edges, as the walls' edges are
            // along the straight ones.
            LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.4, HeightPolicy::Surface),
        ]);
        NativeLandscape::riser_hatching(layers, Some(edges.len()), truncation)
    }

    /// The layers under `truncation`, a hatching on the risers added or
    /// taken away. A banded truncation steps at its band boundaries, and the
    /// risers there are hatched as the walls are: by `riser_hatch`, after the
    /// first hatching on every wall, at its spacing and in its style, unless
    /// a layer hatches the risers already. Under any other truncation there
    /// are no band risers, and a `riser_hatch` in place goes. `walls` is how
    /// many walls the perimeter has, None when they are not a perimeter, and
    /// then no wall hatching is on every wall.
    pub fn riser_hatching(layers: Vec<LayerSpec>, walls: Option<usize>, truncation: &Truncation) -> Vec<LayerSpec> {
        if !matches!(truncation, Truncation::RealBands { .. }) {
            return layers
                .into_iter()
                .filter(|spec| {
                    !(spec.name == "riser_hatch"
                        && matches!(
                            spec.source,
                            LayerSource::Hatching { faces: FaceSelector::Risers { tiled: false, .. }, .. }
                        ))
                })
                .collect();
        }
        let Some(walls) = walls else { return layers };
        if layers.iter().any(|l| matches!(l.source, LayerSource::Hatching { faces: FaceSelector::Risers { .. }, .. })) {
            return layers;
        }
        let found = layers.iter().position(|spec| match &spec.source {
            LayerSource::Hatching { faces: FaceSelector::Walls { edges, .. }, .. } => {
                (0..walls as i64).all(|w| edges.contains(&w))
            }
            _ => false,
        });
        let Some(i) = found else { return layers };
        let LayerSource::Hatching { hatching: Hatching { spacing, .. }, faces: FaceSelector::Walls { end_lines, .. } } =
            &layers[i].source
        else {
            return layers;
        };
        let wall = &layers[i];
        let riser = LayerSpec {
            name: "riser_hatch".to_string(),
            role: wall.role,
            source: LayerSource::riser_hatching(*spacing, false, *end_lines),
            width: wall.width,
            height_policy: wall.height_policy,
            color: wall.color.clone(),
            clipped: wall.clipped,
        };
        let mut out = layers.clone();
        out.insert(i + 1, riser);
        out
    }

    // MARK: sampling

    /// The samples at a resolution along the longer side; None where
    /// [`NativeLandscape::grid_shape`] is.
    pub fn sample(compiled: &Compiled, domain: Domain, resolution: i64) -> Option<Samples> {
        Some(NativeLandscape::sample_shape(compiled, domain, NativeLandscape::grid_shape(domain, resolution)?))
    }

    /// The same at a shape given outright, for a window that must line up
    /// sample for sample with another: `PeriodicTorus.drawing` samples the
    /// plane a few cells past a period rectangle on every side, on the
    /// rectangle's own grid extended. Swift splits the rows over the cores
    /// (`concurrentPerform`); each sample is its own, so a plain loop gives
    /// the same values.
    pub fn sample_shape(compiled: &Compiled, domain: Domain, shape: GridShape) -> Samples {
        let real = NativeLandscape::linspace(domain.real.lo, domain.real.hi, shape.n_real);
        let imag = NativeLandscape::linspace(domain.imag.lo, domain.imag.hi, shape.n_imag);
        let count = shape.n_real * shape.n_imag;
        let mut magnitude = vec![0.0; count];
        let mut phase = vec![0.0; count];
        for (y, &im) in imag.iter().enumerate() {
            let row = y * shape.n_real;
            for (x, &re) in real.iter().enumerate() {
                let mut v = compiled.call(Complex::new(re, im));
                if !v.is_finite() {
                    v = Complex::real(NativeLandscape::HUGE);
                } else if v.magnitude() > NativeLandscape::HUGE {
                    let a = v.argument();
                    v = Complex::new(NativeLandscape::HUGE * real::cos(a), NativeLandscape::HUGE * real::sin(a));
                }
                magnitude[row + x] = v.magnitude();
                phase[row + x] = v.argument();
            }
        }
        Samples { magnitude, phase, shape }
    }

    /// `kurven.export.quantize`: to float32, clamped into the float64 range so
    /// rounding never invents a value the samples did not have -- a phase grid
    /// with a value above π would contour at π where the drawing has nothing.
    pub fn quantize(values: &[f64]) -> Vec<f32> {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for &v in values.iter().filter(|v| !v.is_nan()) {
            lo = swift::min(lo, v);
            hi = swift::max(hi, v);
        }
        if !(lo <= hi) {
            return values.iter().map(|&v| v as f32).collect();
        }
        let mut lo32 = lo as f32;
        if f64::from(lo32) < lo {
            lo32 = lo32.next_up();
        }
        let mut hi32 = hi as f32;
        if f64::from(hi32) > hi {
            hi32 = hi32.next_down();
        }
        // Swift's generic `min(max(x, lo), hi)`, which keeps a NaN.
        let max32 = |x: f32, y: f32| if y >= x { y } else { x };
        let min32 = |x: f32, y: f32| if y < x { y } else { x };
        values.iter().map(|&v| min32(max32(v as f32, lo32), hi32)).collect()
    }
}

impl NativeLandscape {
    // MARK: the landscape

    /// The bundle for a request: `build_scene` with the geometry left described
    /// and `export --derived`, in memory.
    ///
    /// With `refine`, the contour layers are placed by the function rather
    /// than by the grid (`ContourRefiner`), and the bundle carries the refiner
    /// so every restyle keeps doing so. Off, the curves are exactly what the
    /// Python side derives from the same grids, which is what the fixtures
    /// compare.
    ///
    /// Swift's defaults are `git_sha` "native" and `refine` true.
    pub fn build(request: &LandscapeRequest, git_sha: &str, refine: bool) -> Result<KurvenBundle, BuildError> {
        let compiled = Expression::compile(&request.expression)?;
        let canonical = compiled.expression();
        let domain = request.domain;
        let samples =
            NativeLandscape::sample(&compiled, domain, request.resolution).ok_or(BuildError::Domain(domain))?;
        let shape = samples.shape;
        let truncation =
            request.truncation.clone().unwrap_or_else(|| NativeLandscape::default_truncation(&samples.magnitude));
        let density = NativeLandscape::wall_density(domain, shape).ok_or(BuildError::Domain(domain))?;
        let perimeter = NativeLandscape::rectangle(domain, &density);
        let layers = request.layers.clone().unwrap_or_else(|| {
            NativeLandscape::default_layers(
                domain,
                &truncation,
                NativeLandscape::ceiling(&truncation, &samples.magnitude),
                request.spacing,
                true,
            )
        });
        let step = 1.max(shape.n_real.max(shape.n_imag) as i64 / NativeLandscape::OCCLUDER_RESOLUTION);
        let grid_shape = (shape.n_imag as i64, shape.n_real as i64);
        let params: JsonObject = [
            ("expression", JsonValue::String(canonical.clone())),
            ("name", JsonValue::String(request.name.clone())),
            ("resolution", JsonValue::Int(request.resolution)),
            ("rMin", JsonValue::Double(domain.real.lo)),
            ("rMax", JsonValue::Double(domain.real.hi)),
            ("iMin", JsonValue::Double(domain.imag.lo)),
            ("iMax", JsonValue::Double(domain.imag.hi)),
            ("nReal", JsonValue::Int(shape.n_real as i64)),
            ("nImag", JsonValue::Int(shape.n_imag as i64)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let manifest = Manifest::new(
            domain,
            GridRef { file: "height.npy".to_string(), shape: grid_shape, dtype: NpyDType::Float32 },
            Some(GridRef { file: "phase.npy".to_string(), shape: grid_shape, dtype: NpyDType::Float32 }),
            truncation.clone(),
            Occluder {
                step,
                tiles: vec![Affine2::IDENTITY],
                walls: Walls::Perimeter { perimeter, base: 0.0 },
                region: Region::Full,
                base: 0.0,
            },
            layers,
            vec![CameraPreset {
                name: "oblique".to_string(),
                projection: NativeLandscape::PROJECTION,
                depth_tolerance: NativeLandscape::DEFAULT_DEPTH_TOLERANCE,
                buffer: NativeLandscape::DEFAULT_BUFFER,
            }],
            Provenance {
                function: canonical,
                example: "function".to_string(),
                params,
                cpu_count: 1,
                git_sha: git_sha.to_string(),
            },
        );
        let height = Grid2D::new(shape.n_real, shape.n_imag, domain, NativeLandscape::quantize(&samples.magnitude));
        let phase = Grid2D::new(shape.n_real, shape.n_imag, domain, NativeLandscape::quantize(&samples.phase));
        let surface = Surface::new(height, Some(phase), truncation, false);
        let name = if request.name.is_empty() { "landscape" } else { request.name.as_str() };
        let refiner = refine.then(|| ContourRefiner::new(compiled, domain, shape.n_real, shape.n_imag));
        Ok(KurvenBundle::described(
            PathBuf::from(format!("/native/{name}.kurven")),
            manifest,
            surface,
            refiner.map(|r| r.refine_hook()),
        ))
    }
}

/// The finite values, sorted ascending by `<` with Swift's stable sort, so
/// that 0 and -0 keep their order as Swift's `sorted()` keeps it.
fn sorted_finite(values: &[f64]) -> Vec<f64> {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    finite
}

/// Swift's `Sequence.max()` on doubles: `max(by: <)`, which replaces the
/// running maximum only when it is less than the next element, so a NaN
/// first is kept and a NaN later is passed over.
fn sequence_max(values: &[f64]) -> Option<f64> {
    let mut it = values.iter().copied();
    let mut best = it.next()?;
    for v in it {
        if best < v {
            best = v;
        }
    }
    Some(best)
}

/// Swift's `Sequence.min()` on doubles: `min(by: <)`.
fn sequence_min(values: &[f64]) -> Option<f64> {
    let mut it = values.iter().copied();
    let mut best = it.next()?;
    for v in it {
        if v < best {
            best = v;
        }
    }
    Some(best)
}

/// The rules `kurven.landscape` derives a landscape's styling by, as far as
/// the consumer needs them when the *truncation* moves: pure arithmetic over one
/// number, pinned against the Python answers in `kurven-test`.
pub enum LandscapeStyle {}

impl LandscapeStyle {
    /// `x` rounded to a number a person would have chosen: 1, 2, 2.5 or 5 times
    /// a power of ten. Down, for a spacing, where rounding up thins the
    /// hatching. Swift's `down` defaults to false.
    pub fn nice(x: f64, down: bool) -> f64 {
        if !(x.is_finite() && x > 0.0) {
            return 1.0;
        }
        let scale = real::pow(10.0, real::log10(x).floor());
        let m = x / scale;
        let steps = [1.0, 2.0, 2.5, 5.0, 10.0];
        if down {
            let below: Vec<f64> = steps.iter().copied().filter(|&s| s <= m * (1.0 + 1e-12)).collect();
            return scale * sequence_max(&below).unwrap_or(1.0);
        }
        let above: Vec<f64> = steps.iter().copied().filter(|&s| s >= m * (1.0 - 1e-12)).collect();
        scale * sequence_min(&above).unwrap_or(10.0)
    }

    /// Major and minor |f| levels under a ceiling: about ten majors, five
    /// minors to a major, as multiples of the step.
    pub fn magnitude_levels(ceiling: f64) -> (Vec<f64>, Vec<f64>) {
        if !(ceiling.is_finite() && ceiling > 0.0) {
            return (Vec::new(), Vec::new());
        }
        let step = LandscapeStyle::nice(ceiling / 10.0, false);
        let minor_step = step / 5.0;
        // A finite positive ceiling over a step near a tenth of it is about
        // ten: Swift's `Int` of it cannot trap, and the fallback is never taken.
        let count = |x: f64| swift::int(x).unwrap_or(1).max(1);
        let major: Vec<f64> = (1..=count(ceiling / step)).map(|k| (step * k as f64 * 1e10).round() / 1e10).collect();
        let minor: Vec<f64> = (1..=count(ceiling / minor_step))
            .map(|k| (minor_step * k as f64 * 1e10).round() / 1e10)
            .filter(|m| !major.iter().any(|j| (j - m).abs() < 1e-9))
            .collect();
        (major, minor)
    }

    /// The levels a landscape is actually given under these truncations: the rule
    /// above, less any level sitting exactly on the plateau, which is the truncation edge.
    pub fn levels(truncation: &Truncation) -> (Vec<f64>, Vec<f64>) {
        let Some(ceiling) = LandscapeStyle::ceiling(truncation) else { return (Vec::new(), Vec::new()) };
        let (major, minor) = LandscapeStyle::magnitude_levels(ceiling);
        (
            major.into_iter().filter(|&m| m < ceiling - 1e-9).collect(),
            minor.into_iter().filter(|&m| m < ceiling - 1e-9).collect(),
        )
    }

    /// The highest |f| worth a contour: the truncation, when there is one.
    pub fn ceiling(truncation: &Truncation) -> Option<f64> {
        match truncation {
            Truncation::None => None,
            Truncation::Uniform(z) => Some(*z),
            Truncation::RealBands { bands, beyond } => {
                let mut tops: Vec<f64> = bands.iter().map(|b| b.height).collect();
                if beyond.is_finite() {
                    tops.push(*beyond);
                }
                sequence_max(&tops)
            }
        }
    }

    /// About seventy strokes along the longest side, rounded to a ruler.
    pub fn hatch_spacing(domain: Domain) -> f64 {
        LandscapeStyle::nice(swift::max(domain.real.length().abs(), domain.imag.length().abs()) / 70.0, true)
    }
}
