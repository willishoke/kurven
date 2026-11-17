//! A doubly periodic function drawn on the torus its periods glue into: the
//! port of `Sources/KurvenLandscape/Torus.swift`.

use std::fmt;
use std::sync::Arc;

use kurven_core::cover::PeriodLattice;
use kurven_core::hatching::FaceSelector;
use kurven_core::manifest::{
    Affine2, ContourField, Domain, HeightPolicy, Interval, LayerRole, LayerSource, LayerSpec, Region, Restriction,
    Truncation,
};
use kurven_core::parametric::{ParamAxis, ParamScalar, ParametricError, ParametricSurface};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::spaces::{DomainSpace, P2, P3, ParamSpace, WorldSpace};
use kurven_core::surface::Surface;
use kurven_math::{Complex, Expression, ExpressionError, Jacobi, Root, format, real, swift};

use crate::landscape::{GridShape, LandscapeStyle, NativeLandscape};
use crate::refine::ContourRefiner;

/// Contours of |f| and arg f for other layers (`Drawing.contours`).
pub type ContoursFn = Arc<dyn Fn(&[LayerSpec]) -> Vec<Layer> + Send + Sync>;
/// A layer carried onto another surface.
type Carry = Arc<dyn Fn(&Layer) -> Layer + Send + Sync>;

/// Why a periodic drawing cannot be built: the expression (Swift's thrown
/// error), and what Swift's preconditions trap on.
#[derive(Debug, Clone, PartialEq)]
pub enum PeriodicError {
    Expression(ExpressionError),
    Surface(ParametricError),
    /// A period rectangle whose sides are not numbers, where Swift's rounding traps.
    Domain(Domain),
}

impl fmt::Display for PeriodicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeriodicError::Expression(e) => write!(f, "{e}"),
            PeriodicError::Surface(e) => write!(f, "{e}"),
            PeriodicError::Domain(d) => write!(f, "a period rectangle whose sides are not numbers: {d:?}"),
        }
    }
}

impl std::error::Error for PeriodicError {}

impl From<ExpressionError> for PeriodicError {
    fn from(e: ExpressionError) -> PeriodicError {
        PeriodicError::Expression(e)
    }
}

impl From<ParametricError> for PeriodicError {
    fn from(e: ParametricError) -> PeriodicError {
        PeriodicError::Surface(e)
    }
}

/// A doubly periodic function drawn on the torus its periods glue into.
///
/// A function with periods ω₁ and ω₂ lives on the rectangle they span with
/// opposite edges identified, and that rectangle, rolled up both ways, is a
/// torus. The elliptic plate reflects one tile across a grid to show the
/// same thing flat; here the contours close up on themselves instead.
///
/// The contours are the landscape's own: the function sampled over the
/// plane, marching squares on |f| and arg f, the phase wraps dropped, every
/// vertex placed on the function's level set by `ContourRefiner` and every
/// phase line carried to the zero or pole it ends at. The window sampled is
/// the period rectangle and a padding of cells past it on every side: the
/// torus is the plane modulo the periods (`PeriodLattice`), the rectangle one
/// fundamental domain of it, and a zero on the rectangle's edge -- sn has two
/// on the real axis and a pole on the imaginary one -- is a point of the
/// plane's interior like any other, with the whole fan of lines round it to
/// be placed. Each curve is then cut to the rectangle and carried into it by
/// whole periods, and that point of the rectangle is each vertex's surface
/// coordinate: where it lies on the torus and what its visibility is tested
/// by. Where a curve crosses the rectangle's edge it ends on the edge, and
/// its continuation, a copy of the same curve one period over, begins on the
/// opposite edge at the same point of the torus.
pub enum PeriodicTorus {}

/// A periodic drawing: its surface, its layers, and what a relief, a
/// repeat or a re-placing needs (`PeriodicTorus.Drawing`).
#[derive(Clone)]
pub struct Drawing {
    pub surface: ParametricSurface,
    pub layers: Vec<Layer>,
    /// The landscape's height at a point of the rectangle, |f| truncated
    /// at the ceiling, for `relief`; None for a drawing built without f.
    pub height: Option<ParamScalar>,
    pub ceiling: f64,
    /// The torus's radii, for a drawing to be re-placed on the same torus.
    pub radii: (f64, f64),
    /// Whether the torus is parametrized isothermally round its tube
    /// (`ParametricSurface::torus_over`).
    pub isothermal: bool,
    /// Contours of |f| and arg f for other layers, drawn as this
    /// drawing's are: from the function as it was sampled once, carried
    /// through whatever the drawing was repeated, re-placed or displaced
    /// by, onto its surface. None for a drawing built without f.
    pub contours: Option<ContoursFn>,
}

impl Drawing {
    /// Swift's `init(surface:layers:)` with its defaults: no height, an
    /// infinite ceiling, radii (2, 1), in the tube's angle, no contours.
    pub fn new(surface: ParametricSurface, layers: Vec<Layer>) -> Drawing {
        Drawing {
            surface,
            layers,
            height: None,
            ceiling: f64::INFINITY,
            radii: (2.0, 1.0),
            isothermal: false,
            contours: None,
        }
    }

    /// The same curves, each layer carried by `carry`, and the contours
    /// drawn later carried by it too.
    fn carrying(
        &self,
        carry: Carry,
        surface: ParametricSurface,
        radii: (f64, f64),
        isothermal: Option<bool>,
    ) -> Drawing {
        let carried: Option<ContoursFn> = self.contours.clone().map(|contours| {
            let carry = carry.clone();
            Arc::new(move |specs: &[LayerSpec]| contours(specs).iter().map(|l| carry(l)).collect()) as ContoursFn
        });
        Drawing {
            surface,
            layers: self.layers.iter().map(|l| carry(l)).collect(),
            height: self.height.clone(),
            ceiling: self.ceiling,
            radii,
            isothermal: isothermal.unwrap_or(self.isothermal),
            contours: carried,
        }
    }

    /// The same function over several periods: a torus whose rectangle
    /// is `real` periods long and `imag` periods high, carrying `real`
    /// × `imag` copies of the curves. The curves are periodic, so each copy is
    /// a translation of the one computed; the bigger torus is the
    /// quotient of the plane by the sublattice, and this is the
    /// pullback along the covering. The grid keeps the base's
    /// samples per period around the ring, up to `grid` in all.
    pub fn repeated(&self, real: i64, imag: i64, grid: i64) -> Result<Drawing, PeriodicError> {
        if !(real >= 1 && imag >= 1 && real * imag > 1) {
            return Ok(self.clone());
        }
        let (pu, pv) = (self.surface.u.range.length(), self.surface.v.range.length());
        let (ulo, vlo) = (self.surface.u.range.lo, self.surface.v.range.lo);
        let domain =
            Domain::new(Interval::new(ulo, ulo + real as f64 * pu), Interval::new(vlo, vlo + imag as f64 * pv));
        let samples = (
            (self.surface.u.samples as i64 * real).min(grid) as usize,
            (self.surface.v.samples as i64 * imag).max(32) as usize,
        );
        let torus = PeriodicTorus::torus_samples(domain, self.radii.0, self.radii.1, samples, self.isothermal)?;
        let placed_on = torus.clone();
        let carry: Carry = Arc::new(move |layer: &Layer| {
            let Some(coords) = &layer.paths.coords else { return layer.clone() };
            let mut all: Vec<Vec<P2<ParamSpace>>> = Vec::new();
            for i in 0..real {
                for j in 0..imag {
                    let shift = (i as f64 * pu, j as f64 * pv);
                    for k in 0..layer.paths.count() {
                        all.push(
                            coords[layer.paths.offsets[k]..layer.paths.offsets[k + 1]]
                                .iter()
                                .map(|c| P2::new(c.x() + shift.0, c.y() + shift.1))
                                .collect(),
                        );
                    }
                }
            }
            let paths: Vec<Vec<P3<WorldSpace>>> =
                all.iter().map(|r| r.iter().map(|&c| P3::from_vec((placed_on.map)(c).position)).collect()).collect();
            Layer { spec: layer.spec.clone(), paths: PolylineSet::from_paths_with_coords(&paths, &all) }
        });
        Ok(self.carrying(carry, torus, self.radii, None))
    }

    /// The landscape in relief on the torus: the surface displaced along
    /// its normal by `amplitude` times the truncated |f|, and every
    /// layer placed again through the displaced map, since the curves'
    /// coordinates are the rectangle's whatever the embedding. The
    /// relief bounds a solid while the displacement stays under the
    /// torus's reach, `reach`; past it the surface passes through itself
    /// and its back faces are kept.
    pub fn relief(&self, amplitude: f64, reach: f64) -> Result<Drawing, PeriodicError> {
        let Some(height) = self.height.clone() else { return Ok(self.clone()) };
        if amplitude == 0.0 {
            return Ok(self.clone());
        }
        // The displacement lies within ±|amplitude| · ceiling only for a
        // height in [0, ceiling], so a ceiling below 0 encloses nothing.
        let encloses = self.surface.outward.is_some() && self.ceiling >= 0.0 && amplitude.abs() * self.ceiling < reach;
        let displaced = self.surface.displaced(height, amplitude, self.ceiling, encloses)?;
        let on = displaced.clone();
        Ok(self.carrying(Arc::new(move |l: &Layer| placed(l, &on)), displaced, self.radii, None))
    }

    /// The same function on a torus of other radii, or parametrized
    /// otherwise round its tube. The contours are the function's, which
    /// neither touches: only where each vertex sits in space moves, read
    /// from its coordinate on the new torus.
    pub fn placed(&self, major: f64, minor: f64, grid: i64, isothermal: bool) -> Result<Drawing, PeriodicError> {
        let domain = Domain::new(self.surface.u.range, self.surface.v.range);
        let torus = PeriodicTorus::torus_grid(domain, major, minor, grid, isothermal)?;
        let on = torus.clone();
        Ok(self.carrying(Arc::new(move |l: &Layer| placed(l, &on)), torus, (major, minor), Some(isothermal)))
    }
}

/// The same curves, each vertex placed by its coordinate on `surface`.
pub fn placed(layer: &Layer, surface: &ParametricSurface) -> Layer {
    let Some(coords) = &layer.paths.coords else { return layer.clone() };
    Layer {
        spec: layer.spec.clone(),
        paths: PolylineSet::new(
            coords.iter().map(|&c| P3::from_vec((surface.map)(c).position)).collect(),
            layer.paths.offsets.clone(),
            Some(coords.clone()),
        ),
    }
}

impl PeriodicTorus {
    /// How far past the rectangle the plane is sampled, in cells. More than
    /// the refiner looks for a zero or pole from a run's end
    /// (`ContourRefiner.reach`) and the two cells it carries one over, so a
    /// singular point on the rectangle's edge wants nothing outside the window.
    pub const PADDING: usize = 8;

    /// Only rectangular period lattices: a real period along `real` and an
    /// imaginary one along `imag`, which is what sn, and cn and dn on
    /// suitable rectangles, have.
    /// `truncated`: the phase lines are cut at the ceiling, as a landscape's
    /// are at its truncation, and the truncation edge |f| = ceiling is drawn; for a drawing to be
    /// put in relief, where the truncation is a plateau. Otherwise every phase line
    /// runs through its pole. `contours`: the contours of |f| and arg f to
    /// draw, each kept below the ceiling where its restriction says so;
    /// None for the four families and the truncation edge (`contours`).
    /// `isothermal`: the torus parametrized isothermally round its tube
    /// (`ParametricSurface::torus_over`).
    ///
    /// Swift's defaults: origin 0, radii (2, 1), resolution 600, ceiling 3,
    /// grid 1024, neither truncated nor isothermal, the default contours
    /// ([`PeriodicOptions::default`]).
    pub fn drawing(expression: &str, periods: (f64, f64), o: &PeriodicOptions) -> Result<Drawing, PeriodicError> {
        let compiled = Expression::compile(expression)?;
        let domain = Domain::new(
            Interval::new(o.origin.re, o.origin.re + periods.0),
            Interval::new(o.origin.im, o.origin.im + periods.1),
        );
        let surface = PeriodicTorus::torus_grid(domain, o.major, o.minor, o.grid, o.isothermal)?;
        let group = PeriodLattice::new(domain);

        // The window: the rectangle's grid, extended `PADDING` cells each
        // way and shifted by half a cell on both axes. The samples past the
        // edge are the function's own, since it is periodic, and a curve and
        // its copy one period over are computed from the same sample
        // positions, so they agree to rounding where the clip joins them.
        // The half cell keeps every sample off the rectangle's edges and off
        // its half-period lines: sn is real on the real axis and imaginary
        // on the imaginary one, so its lines of phase 0, π and ±π/2 lie
        // along them, and a level set lying along a row of samples is the
        // one configuration marching squares cannot resolve, each sample a
        // tie decided by rounding and the two copies of a curve chained
        // differently. Off the row, the same lines cross the cells.
        let shape = NativeLandscape::grid_shape(domain, o.resolution).ok_or(PeriodicError::Domain(domain))?;
        let dx = domain.real.length() / (shape.n_real - 1) as f64;
        let dy = domain.imag.length() / (shape.n_imag - 1) as f64;
        let m = PeriodicTorus::PADDING as f64;
        let cover = Domain::new(
            Interval::new(domain.real.lo - (m + 0.5) * dx, domain.real.hi + (m - 0.5) * dx),
            Interval::new(domain.imag.lo - (m + 0.5) * dy, domain.imag.hi + (m - 0.5) * dy),
        );
        let cover_shape = GridShape {
            n_real: shape.n_real + 2 * PeriodicTorus::PADDING,
            n_imag: shape.n_imag + 2 * PeriodicTorus::PADDING,
        };
        let samples = NativeLandscape::sample_shape(&compiled, cover, cover_shape);
        let height = kurven_core::grid::Grid2D::new(
            cover_shape.n_real,
            cover_shape.n_imag,
            cover,
            NativeLandscape::quantize(&samples.magnitude),
        );
        let phase = kurven_core::grid::Grid2D::new(
            cover_shape.n_real,
            cover_shape.n_imag,
            cover,
            NativeLandscape::quantize(&samples.phase),
        );
        let ceiling = o.ceiling;
        let field = Surface::new(height, Some(phase), Truncation::Uniform(ceiling), false);
        // Not lifted: the curves lie on the torus, placed by its coordinate,
        // and |f| is no height of it.
        let refiner =
            ContourRefiner::with(compiled.clone(), cover, cover_shape.n_real, cover_shape.n_imag, 0.02, 5, false);
        // Swift asks `refiner.refine`, the hook over the singular points, for
        // every layer; the points are the same each time, so the port finds
        // them once, on the first layer that asks.
        let hook = Arc::new(std::sync::OnceLock::new());
        // The rounding the clip allows at the rectangle's edges: wider than
        // the error the function is evaluated to, so that a zero found on
        // the far edge by Newton is on one side of it, not sometimes on
        // either, which would leave a stub of the line along that edge. A
        // crossing is put on the edge itself whatever the slack.
        let slack = 1e-6 * swift::max(dx, dy);

        // One layer's contours, from the samples and the refiner, which a
        // later restyle draws with too (`Drawing.contours`).
        let draw = {
            let (surface, refiner) = (surface.clone(), refiner.clone());
            Arc::new(move |spec: &LayerSpec| -> Option<Layer> {
                let LayerSource::Contour { field: kind, levels, restriction, .. } = &spec.source else { return None };
                let kind = *kind;
                if !kind.is_field_of_f() {
                    return None;
                }
                // Kept below the truncation, a phase line is lifted to |f| and
                // kept under the ceiling, ending on the truncation edge where it
                // crosses it. Contoured on the plane's graph, and drawn on the torus.
                let cut = *restriction == Restriction::BelowTruncation;
                let refine = hook.get_or_init(|| refiner.refine_hook());
                let flat = field.derive(
                    &LayerSource::Contour {
                        field: kind,
                        levels: levels.clone(),
                        restriction: restriction.clone(),
                        faces: FaceSelector::Graph { tiled: false },
                        light: None,
                    },
                    if cut { HeightPolicy::Magnitude } else { HeightPolicy::Surface },
                    &Region::Full,
                    &[Affine2::IDENTITY],
                    Some(refine),
                );
                // Where a curve crosses the rectangle's edge, the crossing is
                // solved on f along the edge, so that it lies on the level set
                // like every other vertex: the level is the one the inside
                // vertex is on, and the crossing is looked for within half a
                // cell of the chord's, which is within the refiner's tolerance
                // of it. A phase near ±π is solved on the turned phase, as the
                // cut is placed.
                let on_edge = |a: P2<DomainSpace>, _: P2<DomainSpace>, c: P2<DomainSpace>| -> P2<DomainSpace> {
                    let g = |z: Complex| -> f64 {
                        match kind {
                            ContourField::Magnitude => refiner.magnitude(z),
                            _ => {
                                if refiner.phase(Complex::new(a.x(), a.y())).abs() > core::f64::consts::FRAC_PI_2 {
                                    refiner.turned_phase(z)
                                } else {
                                    refiner.phase(z)
                                }
                            }
                        }
                    };
                    let level = g(Complex::new(a.x(), a.y()));
                    if !level.is_finite() {
                        return c;
                    }
                    let vertical =
                        swift::min((c.x() - domain.real.lo).abs(), (c.x() - domain.real.hi).abs()) <= 2.0 * slack;
                    let at =
                        |t: f64| -> P2<DomainSpace> { if vertical { P2::new(c.x(), t) } else { P2::new(t, c.y()) } };
                    let residual = |t: f64| -> f64 {
                        let p = at(t);
                        let r = g(Complex::new(p.x(), p.y())) - level;
                        if kind == ContourField::Phase { ContourRefiner::wrapped(r) } else { r }
                    };
                    // No bracket, no solve; and none either where the level set
                    // lies along the edge itself, as sn's lines of phase 0 and
                    // π do along the real axis: the residual is then zero the
                    // whole way and any point would do. There the chord's
                    // crossing, a vertex of a curve crossing the line, is right.
                    let t0 = if vertical { c.y() } else { c.x() };
                    let h = 0.5 * refiner.cell();
                    let (ra, rb) = (residual(t0 - h), residual(t0 + h));
                    let flat = 1e-9 * if kind == ContourField::Phase { 1.0 } else { swift::max(level.abs(), 1.0) };
                    if !(ra.is_finite()
                        && rb.is_finite()
                        && ra.abs() > flat
                        && rb.abs() > flat
                        && real::sign_minus(ra) != real::sign_minus(rb))
                    {
                        return c;
                    }
                    match Root::brent_default(residual, t0 - h, t0 + h, ra, rb, 1e-12 * refiner.cell()) {
                        Some(t) => at(t),
                        None => c,
                    }
                };
                // Each curve of the plane, cut to the rectangle and carried into
                // it: (Re z, Im z) brought in by whole periods is the surface
                // coordinate, and the torus says where that is.
                let mut placed: Vec<Vec<P3<WorldSpace>>> = Vec::new();
                let mut coords: Vec<Vec<P2<ParamSpace>>> = Vec::new();
                for i in 0..flat.count() {
                    let path: Vec<P2<DomainSpace>> = flat.vertices[flat.offsets[i]..flat.offsets[i + 1]]
                        .iter()
                        .map(|v| P2::new(v.x(), v.y()))
                        .collect();
                    for run in group.clipped(&path, slack, on_edge) {
                        let c = group.canonical(&run);
                        placed.push(c.iter().map(|&p| P3::from_vec((surface.map)(p).position)).collect());
                        coords.push(c);
                    }
                }
                Some(Layer { spec: spec.clone(), paths: PolylineSet::from_paths_with_coords(&placed, &coords) })
            })
        };
        let specs = o.contours.clone().unwrap_or_else(|| PeriodicTorus::contours(ceiling, o.truncated));
        let mut layers: Vec<Layer> = specs.iter().filter_map(|s| draw(s)).collect();
        layers.push(Layer {
            spec: LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, 0.6, HeightPolicy::Surface),
            paths: PolylineSet::empty(),
        });
        // The landscape's height over the rectangle, for a relief.
        let f = Arc::new(compiled);
        let relief: ParamScalar = Arc::new(move |c: P2<ParamSpace>| {
            let v = f.call(Complex::new(c.x(), c.y()));
            if v.is_finite() { swift::min(v.magnitude(), ceiling) } else { ceiling }
        });
        let contours: ContoursFn = Arc::new(move |specs: &[LayerSpec]| specs.iter().filter_map(|s| draw(s)).collect());
        Ok(Drawing {
            surface,
            layers,
            height: Some(relief),
            ceiling,
            radii: (o.major, o.minor),
            isothermal: o.isothermal,
            contours: Some(contours),
        })
    }

    /// The contours a drawing draws unless told otherwise: |f| and arg f, each
    /// at major and minor levels, the moduli under the ceiling; truncated, the
    /// phase lines kept below it and the truncation edge |f| = ceiling.
    pub fn contours(ceiling: f64, truncated: bool) -> Vec<LayerSpec> {
        let (major, minor) = LandscapeStyle::magnitude_levels(ceiling);
        let mut families: Vec<(&str, LayerRole, ContourField, Vec<f64>, f64)> = vec![
            ("mag_major", LayerRole::Magnitude, ContourField::Magnitude, major, 0.35),
            ("mag_minor", LayerRole::Magnitude, ContourField::Magnitude, minor, 0.15),
            ("ang_major", LayerRole::Phase, ContourField::Phase, NativeLandscape::phase_major(), 0.35),
            ("ang_minor", LayerRole::Phase, ContourField::Phase, NativeLandscape::phase_minor(), 0.15),
        ];
        if truncated {
            families.push(("truncation edge", LayerRole::Outline, ContourField::Magnitude, vec![ceiling], 0.6));
        }
        families
            .into_iter()
            .map(|(name, role, kind, levels, width)| {
                let restriction = if truncated && kind == ContourField::Phase {
                    Restriction::BelowTruncation
                } else {
                    Restriction::All
                };
                LayerSpec::new(
                    name,
                    role,
                    LayerSource::Contour {
                        field: kind,
                        levels,
                        restriction,
                        faces: FaceSelector::Surface,
                        light: None,
                    },
                    width,
                    HeightPolicy::Surface,
                )
            })
            .collect()
    }

    /// The torus a rectangle of periods glues into, its grid `grid`
    /// points around the real period and enough around the imaginary one to
    /// keep the cells square on the tube.
    pub fn torus_grid(
        domain: Domain,
        major: f64,
        minor: f64,
        grid: i64,
        isothermal: bool,
    ) -> Result<ParametricSurface, ParametricError> {
        let aspect = domain.imag.length() / domain.real.length();
        let v = swift::int(grid as f64 * swift::min(aspect, 1.0) * minor / major).unwrap_or(0).max(32);
        PeriodicTorus::torus_samples(domain, major, minor, (grid as usize, v as usize), isothermal)
    }

    /// The same torus with the grid's samples given outright.
    pub fn torus_samples(
        domain: Domain,
        major: f64,
        minor: f64,
        samples: (usize, usize),
        isothermal: bool,
    ) -> Result<ParametricSurface, ParametricError> {
        ParametricSurface::torus_over(
            major,
            minor,
            ParamAxis::new(domain.real, true, samples.0),
            ParamAxis::new(domain.imag, true, samples.1),
            isothermal,
        )
    }

    /// sn(z, m) on its fundamental rectangle, [0, 4K) × [0, 2K').
    /// `turn`: the rectangle begins `turn` imaginary periods up the plane,
    /// which turns the function that fraction round the tube: at 0 the
    /// real axis, with sn's zeros, lies on the outer equator and the poles
    /// on the inner; at 1/4 the poles sit on top of the tube; at 1/2 they
    /// are on the outer equator. The real period goes round the ring and the
    /// imaginary round the tube, so the rectangle's aspect, tube over ring,
    /// is K′ / (2K); `isothermal` on the torus whose minor radius
    /// `Shape::conformal_minor` pins, the map is conformal. `o.origin` is
    /// set from `turn`.
    pub fn jacobi_sn(modulus: f64, turn: f64, o: &PeriodicOptions) -> Result<Drawing, PeriodicError> {
        let k = Jacobi::quarter_period(modulus);
        let kp = Jacobi::quarter_period(1.0 - modulus);
        let o = PeriodicOptions { origin: Complex::new(0.0, turn * 2.0 * kp), ..o.clone() };
        PeriodicTorus::drawing(&format!("sn(z, {})", format::description(modulus)), (4.0 * k, 2.0 * kp), &o)
    }
}

/// `PeriodicTorus.drawing`'s defaulted arguments, at Swift's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicOptions {
    pub origin: Complex,
    pub major: f64,
    pub minor: f64,
    pub resolution: i64,
    pub ceiling: f64,
    pub grid: i64,
    pub truncated: bool,
    pub isothermal: bool,
    pub contours: Option<Vec<LayerSpec>>,
}

impl Default for PeriodicOptions {
    fn default() -> PeriodicOptions {
        PeriodicOptions {
            origin: Complex::new(0.0, 0.0),
            major: 2.0,
            minor: 1.0,
            resolution: 600,
            ceiling: 3.0,
            grid: 1024,
            truncated: false,
            isothermal: false,
            contours: None,
        }
    }
}
