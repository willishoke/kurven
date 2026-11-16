//! A surface drawing, built: the port of
//! `Sources/KurvenDynamics/SurfaceDrawing.swift`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use kurven_core::camera::Camera;
use kurven_core::hatching::{FaceSelector, Tone};
use kurven_core::integral_curves::Limits;
use kurven_core::manifest::{
    ContourField, DomainAxis, FieldClass, HeightPolicy, LayerRole, LayerSource, LayerSpec, ManifestError,
    SurfaceManifest,
};
use kurven_core::parametric::{ParamScalar, ParametricError, ParametricSurface, PortRefusal};
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scalar_field::ScalarField;
use kurven_core::scene::{ContentID, PreviewMode, Scene};
use kurven_core::spaces::{P3, WorldSpace};
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_landscape::torus::{ContoursFn, Drawing, PeriodicError, PeriodicOptions, PeriodicTorus};
use kurven_math::{Expression, Jacobi, Node, Simd3, real, swift};

use crate::forced::{DynamicsError, ForcedSystem, InvariantTorus, Revolution};

/// Why a drawing cannot be built or restyled: a layer the surface cannot
/// draw, a function that does not compile, a forced system with no torus,
/// and what the port has not brought across yet.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawingError {
    Manifest(ManifestError),
    Periodic(PeriodicError),
    Dynamics(DynamicsError),
    Port(PortRefusal),
}

impl fmt::Display for DrawingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrawingError::Manifest(e) => write!(f, "{e}"),
            DrawingError::Periodic(e) => write!(f, "{e}"),
            DrawingError::Dynamics(e) => write!(f, "{e}"),
            DrawingError::Port(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DrawingError {}

impl From<ManifestError> for DrawingError {
    fn from(e: ManifestError) -> DrawingError {
        DrawingError::Manifest(e)
    }
}
impl From<PeriodicError> for DrawingError {
    fn from(e: PeriodicError) -> DrawingError {
        DrawingError::Periodic(e)
    }
}
impl From<DynamicsError> for DrawingError {
    fn from(e: DynamicsError) -> DrawingError {
        DrawingError::Dynamics(e)
    }
}
impl From<ParametricError> for DrawingError {
    fn from(e: ParametricError) -> DrawingError {
        DrawingError::Periodic(PeriodicError::Surface(e))
    }
}
impl From<PortRefusal> for DrawingError {
    fn from(e: PortRefusal) -> DrawingError {
        DrawingError::Port(e)
    }
}

/// What a forced drawing's fit is made from.
#[derive(Debug, Clone, PartialEq)]
pub struct FitKey {
    pub system: String,
    pub fit: f64,
    pub harmonics: i64,
}

/// A fitted torus and what has been worked out from it, in state space:
/// its values on each grid asked for.
#[derive(Debug, Clone)]
pub struct Forced {
    pub fit: FitKey,
    pub torus: InvariantTorus,
    pub values: BTreeMap<usize, Vec<Simd3>>,
}

/// A surface drawing, built: the surface, its curves, and what is worth keeping
/// between edits.
///
/// What an edit costs is decided here, by comparing the new request with the
/// drawing it replaces, and only what the edit touches is made again:
///
///   - **the curves alone** -- line counts, a linear curve's slope, stroke widths, a
///     forced trajectory's turns, sn's sampling grid -- keeps the surface,
///     and with
///     it `content`, so the window redraws the curves and keeps the surface's
///     textures;
///   - **the radii of a torus** re-place what is drawn on it: a periodic
///     function's contours are the function's, and move with the torus from
///     their coordinates;
///   - **a forced torus's revolution** re-places its values on the grid,
///     kept in state space, and the trajectory's states, without evaluating
///     the fitted series or integrating again;
///   - **its fit** -- system, fit length, harmonics -- fits again.
///
/// A layer painted on the surface, which the request does not draw, goes by
/// its field's class (`FieldClass`, `keeping`): a coordinate's
/// contours are carried while the parameter rectangle keeps its samples, the
/// height's while the surface is kept, and |f| and arg f's with the function.
///
/// Everything reused is exactly what building from nothing would produce, so
/// a drawing depends on its request alone -- the window's is the CLI's. The one
/// `draft` names a request made while a control is dragged; nothing is
/// approximated for it now, and it is kept for the drag to say so.
///
/// A drawing restyled (`restyled`) draws other layers on the same
/// surface, as a bundle that kept them asks; what an edit reuses is still
/// only what its request drew.
#[derive(Clone)]
pub struct SurfaceDrawing {
    pub request: SurfaceRequest,
    pub surface: ParametricSurface,
    pub layers: Vec<Layer>,
    /// The layers the request draws, which are all an edit reuses, whatever
    /// a restyle has drawn instead.
    own: Vec<Layer>,
    /// Contours of the request's function at other levels, from its samples
    /// as they were taken once (`Drawing.contours`); None for a surface that
    /// paints none.
    contours: Option<ContoursFn>,
    /// The surface's identity: kept by an edit that leaves the surface alone,
    /// new whenever it moves.
    pub content: ContentID,
    pub revolution: Option<Revolution>,
    /// False when a forced torus's revolution passes through itself, so its
    /// back faces are kept rather than hidden.
    pub embedded: bool,
    /// What a forced drawing keeps between edits.
    forced: Option<Forced>,
}

impl fmt::Debug for SurfaceDrawing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SurfaceDrawing")
            .field("request", &self.request)
            .field("layers", &self.layers.iter().map(|l| &l.spec.name).collect::<Vec<_>>())
            .field("embedded", &self.embedded)
            .finish_non_exhaustive()
    }
}

fn scaffold(name: &str, source: LayerSource, width: f64, paths: PolylineSet<WorldSpace>) -> Layer {
    Layer { spec: LayerSpec::new(name, LayerRole::Scaffold, source, width, HeightPolicy::Surface), paths }
}

/// Whether a layer is a contour: what a periodic drawing keeps of its own.
fn is_contour(layer: &Layer) -> bool {
    matches!(layer.spec.source, LayerSource::Contour { .. })
}

impl SurfaceDrawing {
    /// A forced drawing's fitted torus.
    pub fn torus(&self) -> Option<&InvariantTorus> {
        self.forced.as_ref().map(|f| &f.torus)
    }

    /// Whether building `request` from this drawing fits a torus again -- the
    /// one edit that takes a noticeable time.
    pub fn refits(&self, request: &SurfaceRequest) -> bool {
        let Shape::Forced { system, fit, harmonics, .. } = &request.shape else { return false };
        self.forced.as_ref().map(|f| &f.fit)
            != Some(&FitKey { system: system.clone(), fit: *fit, harmonics: *harmonics })
    }

    /// The scene this drawing draws, from `camera`.
    pub fn scene(&self, camera: Camera) -> Scene {
        Scene::parametric_scene(self.surface.clone(), self.layers.clone(), camera, PreviewMode::Drawing, 0.0)
    }

    /// Everything the surface itself -- not what is drawn on it -- is built
    /// from: two requests with the same geometry have the same surface.
    fn geometry(r: &SurfaceRequest) -> SurfaceRequest {
        let shape = match &r.shape {
            Shape::Torus { .. } => r.shape.clone(),
            Shape::Sn { modulus, major, minor, conformal, .. } => {
                Shape::Sn { modulus: *modulus, major: *major, minor: *minor, grid: 0, conformal: *conformal }
            }
            Shape::Periodic { periods, major, minor, .. } => {
                Shape::Periodic { expression: String::new(), periods: *periods, major: *major, minor: *minor, grid: 0 }
            }
            Shape::Forced { system, fit, harmonics, radial, axial, radius, .. } => Shape::Forced {
                system: system.clone(),
                fit: *fit,
                harmonics: *harmonics,
                radial: *radial,
                axial: *axial,
                radius: *radius,
                turns: 0,
            },
            // Its contours are placed through the relief, so the whole
            // shape is its geometry, and a kept surface keeps them too.
            Shape::Relief { .. } => r.shape.clone(),
        };
        SurfaceRequest { grid: r.grid, ..SurfaceRequest::new(shape) }
    }

    /// Everything a periodic drawing's contours are built from: the function,
    /// its periods and its sampling, and not the torus they are drawn on, nor
    /// how it is parametrized round its tube.
    fn contours_key(r: &SurfaceRequest) -> Option<Shape> {
        match &r.shape {
            Shape::Sn { modulus, grid, .. } => {
                Some(Shape::Sn { modulus: *modulus, major: 0.0, minor: 0.0, grid: *grid, conformal: false })
            }
            Shape::Periodic { expression, periods, grid, .. } => Some(Shape::Periodic {
                expression: expression.clone(),
                periods: *periods,
                major: 0.0,
                minor: 0.0,
                grid: *grid,
            }),
            _ => None,
        }
    }

    /// Build a drawing. `previous`, when given, lends whatever of it the new
    /// request has not changed. `draft` says the request is a drag's.
    pub fn build(
        request: &SurfaceRequest,
        previous: Option<&SurfaceDrawing>,
        _draft: bool,
    ) -> Result<SurfaceDrawing, DrawingError> {
        let style = request.style;
        let kept = previous.filter(|p| SurfaceDrawing::geometry(&p.request) == SurfaceDrawing::geometry(request));
        // The lines, reused when neither the surface nor their counts moved.
        let lines = |surface: &ParametricSurface, resolution: i64| -> Option<Layer> {
            request.lines.map(|n| {
                let old = kept
                    .filter(|k| k.request.lines == Some(n))
                    .and_then(|k| k.own.iter().find(|l| l.spec.name == "lines"));
                scaffold(
                    "lines",
                    LayerSource::CoordinateCurves { u: n.0, v: n.1 },
                    style.lines,
                    old.map(|l| l.paths.clone()).unwrap_or_else(|| {
                        surface.coordinate_curves((n.0 as usize, n.1 as usize), resolution as usize)
                    }),
                )
            })
        };
        // The linear curve, reused when neither the surface nor it moved. Placed
        // from the grid, so a new slope costs its own vertices and
        // nothing of the surface's.
        let linear_curves = |surface: &ParametricSurface| -> Option<Layer> {
            request.linear_curves.map(|w| {
                let old = kept
                    .filter(|k| k.request.linear_curves == Some(w))
                    .and_then(|k| k.own.iter().find(|l| l.spec.name == "linearCurves"));
                scaffold(
                    "linearCurves",
                    LayerSource::LinearCurves { slope: w.slope, turns: w.turns, count: w.count },
                    style.linear_curves,
                    old.map(|l| l.paths.clone())
                        .unwrap_or_else(|| surface.linear_curves(w.slope, w.turns as usize, w.count as usize, None)),
                )
            })
        };

        let surface: ParametricSurface;
        let mut layers: Vec<Layer>;
        let (mut revolution, mut embedded, mut forced) = (None, true, None);
        let mut contours: Option<ContoursFn> = None;
        match &request.shape {
            Shape::Torus { major, minor } => {
                let n = request.grid as usize;
                surface = match kept {
                    Some(k) => k.surface.clone(),
                    None => ParametricSurface::torus(*major, *minor, (n, (n / 2).max(8)))?,
                };
                layers = lines(&surface, request.line_resolution()).into_iter().collect();
            }
            Shape::Sn { major, minor, .. } | Shape::Periodic { major, minor, .. } => {
                let drawing = match previous {
                    Some(previous)
                        if SurfaceDrawing::contours_key(request).is_some()
                            && SurfaceDrawing::contours_key(&previous.request)
                                == SurfaceDrawing::contours_key(request) =>
                    {
                        // The same contours: as they are, or re-placed on new radii,
                        // or on the same torus parametrized conformally or not.
                        let old = Drawing {
                            contours: previous.contours.clone(),
                            ..Drawing::new(previous.surface.clone(), previous.own.clone())
                        };
                        if kept.is_some() {
                            old
                        } else {
                            old.placed(*major, *minor, request.grid, request.conformal())?
                        }
                    }
                    _ => SurfaceDrawing::periodic(request)?,
                };
                surface = drawing.surface;
                contours = drawing.contours;
                // Their own folds come with them; the style's replace them below.
                layers = drawing.layers.into_iter().filter(is_contour).collect();
                layers.extend(lines(&surface, request.line_resolution()));
            }
            Shape::Relief { .. } => {
                // The same relief keeps its contours as they are; any other
                // change is a new relief.
                if let Some(k) = kept {
                    surface = k.surface.clone();
                    layers = k.own.iter().filter(|l| is_contour(l)).cloned().collect();
                    contours = k.contours.clone();
                } else {
                    let drawing = SurfaceDrawing::periodic(request)?;
                    surface = drawing.surface;
                    layers = drawing.layers.into_iter().filter(is_contour).collect();
                    contours = drawing.contours;
                }
                layers.extend(lines(&surface, request.line_resolution()));
            }
            Shape::Forced { system: name, fit, harmonics, radial, axial, radius, turns } => {
                let Some(system) = ForcedSystem::catalog().get(name) else {
                    return Err(DynamicsError::UnknownSystem {
                        name: name.clone(),
                        known: ForcedSystem::catalog().keys().cloned().collect(),
                    }
                    .into());
                };
                let key = FitKey { system: name.clone(), fit: *fit, harmonics: *harmonics };
                let mut state = match previous.and_then(|p| p.forced.clone()).filter(|old| old.fit == key) {
                    Some(old) => old,
                    None => Forced {
                        fit: key,
                        torus: InvariantTorus::find_default(system, *fit, *harmonics as usize)?,
                        values: BTreeMap::new(),
                    },
                };
                let torus = state.torus.clone();
                let e = Revolution::fitting(&torus, *radial as usize, *axial as usize, *radius);
                revolution = Some(e.clone());
                let grid = (request.grid as usize, (request.grid / 2) as usize);
                if let Some(k) = kept {
                    surface = k.surface.clone();
                    embedded = k.embedded;
                } else {
                    let values = match state.values.get(&grid.0) {
                        Some(v) => v.clone(),
                        None => torus.values(grid),
                    };
                    // The grid asked for now, and the largest before it: the
                    // one a drag's drafts return to.
                    let largest = state.values.keys().max().map_or(grid.0, |&m| m.max(grid.0));
                    state.values.retain(|&k, _| k == largest);
                    state.values.insert(grid.0, values.clone());
                    let (s, emb) = torus.surface_from(&e, grid, &values)?;
                    surface = s;
                    embedded = emb;
                }
                forced = Some(state);

                // The trajectory is the linear curve at the rotation number, placed
                // from the grid: kept when the surface and its turns are.
                let same_curves = kept.is_some()
                    && matches!(previous.map(|p| &p.request.shape), Some(Shape::Forced { turns: t, .. }) if t == turns);
                let old = if same_curves {
                    kept.and_then(|k| k.own.iter().find(|l| l.spec.name == "trajectory"))
                } else {
                    None
                };
                layers = vec![scaffold(
                    "trajectory",
                    LayerSource::Trajectory,
                    style.trajectory,
                    old.map(|l| l.paths.clone()).unwrap_or_else(|| torus.trajectory(&surface, *turns as usize)),
                )];
                layers.extend(lines(&surface, request.line_resolution()));
            }
        }
        layers.extend(linear_curves(&surface));
        // The outline and inner fold curves, derived per camera.
        if style.folds > 0.0 {
            layers.push(Layer {
                spec: LayerSpec::new(
                    "folds",
                    LayerRole::Outline,
                    LayerSource::FoldCurves,
                    style.folds,
                    HeightPolicy::Surface,
                ),
                paths: PolylineSet::empty(),
            });
        }
        Ok(SurfaceDrawing {
            request: request.clone(),
            surface,
            own: layers.clone(),
            layers,
            contours,
            content: kept.map_or_else(ContentID::new, |k| k.content),
            revolution,
            embedded,
            forced,
        })
    }

    /// The drawing a request's function makes on its torus: sn, any periodic
    /// expression, or sn in relief. A torus of revolution or a forced torus
    /// paints no function; Swift's precondition is an assertion here.
    fn periodic(request: &SurfaceRequest) -> Result<Drawing, DrawingError> {
        match &request.shape {
            Shape::Torus { .. } | Shape::Forced { .. } => {
                panic!("a torus of revolution or a forced torus paints no function")
            }
            Shape::Sn { modulus, major, minor, grid, conformal } => Ok(PeriodicTorus::jacobi_sn(
                *modulus,
                0.0,
                &PeriodicOptions {
                    major: *major,
                    minor: *minor,
                    resolution: *grid,
                    grid: request.grid,
                    isothermal: *conformal,
                    ..PeriodicOptions::default()
                },
            )?),
            Shape::Periodic { expression, periods, major, minor, grid } => Ok(PeriodicTorus::drawing(
                expression,
                *periods,
                &PeriodicOptions {
                    major: *major,
                    minor: *minor,
                    resolution: *grid,
                    grid: request.grid,
                    ..PeriodicOptions::default()
                },
            )?),
            Shape::Relief { modulus, major, minor, amplitude, repeats, grid, turn, floor, soft } => {
                // `repeats` periods round the ring and the tube, the curves
                // computed once and carried round by the periods; the grid round
                // the ring in all, 1024 a period at most.
                let mut base = PeriodicTorus::jacobi_sn(
                    *modulus,
                    *turn,
                    &PeriodicOptions {
                        major: *major,
                        minor: *minor,
                        resolution: *grid,
                        grid: request.grid.min(1024),
                        ceiling: SurfaceRequest::RELIEF_CEILING,
                        truncated: true,
                        ..PeriodicOptions::default()
                    },
                )?
                .repeated(repeats.0, repeats.1, request.grid)?;
                // The truncation as a smooth saturation rather than a crease,
                // c · tanh(|f| / c), an experiment on what the crease costs.
                if *soft && let Some(height) = base.height.clone() {
                    let c = base.ceiling;
                    let soft: ParamScalar = Arc::new(move |p| c * real::tanh(height(p) / c));
                    base = Drawing { height: Some(soft), ..base };
                }
                // Only |f| above the floor displaces, so each pole stands alone
                // on the bare torus instead of on a raised ground.
                if let Some(floor) = *floor
                    && let Some(height) = base.height.clone()
                {
                    let raised: ParamScalar = Arc::new(move |p| swift::max(height(p) - floor, 0.0));
                    base = Drawing { height: Some(raised), ceiling: base.ceiling - floor, ..base };
                }
                Ok(base.relief(*amplitude, swift::min(*minor, major - minor))?)
            }
        }
    }

    /// The same surface drawn with `specs`, as a bundle that kept them asks.
    ///
    /// A layer whose source the drawing draws already keeps its curves, under
    /// the new name and style. Any other is drawn on the surface: a contour of
    /// a coordinate or the height by the plain routine (`Face::contours`),
    /// of |f| or arg f from the request's function as it was sampled once,
    /// all of them in one pass (`contours`); coordinate curves, linear
    /// curves, the trajectory, lines of curvature. The surface and its
    /// identity are kept, so a restyle redraws curves alone.
    pub fn restyled(&self, specs: &[LayerSpec]) -> Result<SurfaceDrawing, DrawingError> {
        self.restyled_carrying(specs, None)
    }

    /// `restyled`, with the contours `previous` drew carried over from
    /// it by their field's class (`FieldClass`): a coordinate's where this
    /// drawing's parameter rectangle and its samples are the same -- found
    /// in parameter coordinates, they are the same curves there, and only
    /// their place in space is new -- and the height's where the surface is
    /// the same, since it is read off the map. |f| and arg f come with the
    /// drawing, from its function.
    fn restyled_carrying(
        &self,
        specs: &[LayerSpec],
        previous: Option<&SurfaceDrawing>,
    ) -> Result<SurfaceDrawing, DrawingError> {
        let drawn = |spec: &LayerSpec| self.layers.iter().find(|l| l.spec.source == spec.source);
        let same_rectangle = previous.is_some_and(|p| p.surface.u == self.surface.u && p.surface.v == self.surface.v);
        let same_surface = previous.is_some_and(|p| p.content == self.content);
        let carried = |spec: &LayerSpec| -> Option<PolylineSet<WorldSpace>> {
            let LayerSource::Contour { field, .. } = &spec.source else { return None };
            let class = field.field_class();
            let carries = if class == FieldClass::Coordinate {
                same_rectangle
            } else {
                class == FieldClass::Embedding && same_surface
            };
            if !carries {
                return None;
            }
            let old = previous?.layers.iter().find(|l| l.spec.source == spec.source)?;
            let coords = old.paths.coords.as_ref()?;
            Some(PolylineSet::new(
                coords.iter().map(|&c| P3::from_vec((self.surface.map)(c).position)).collect(),
                old.paths.offsets.clone(),
                Some(coords.clone()),
            ))
        };
        let of_f: Vec<LayerSpec> = specs
            .iter()
            .filter(|spec| {
                drawn(spec).is_none()
                    && matches!(&spec.source, LayerSource::Contour { field, .. } if field.is_field_of_f())
            })
            .cloned()
            .collect();
        // On a surface with no function they are refused below, one by one.
        let placed: Vec<Layer> =
            if of_f.is_empty() { Vec::new() } else { self.contours.as_ref().map(|c| c(&of_f)).unwrap_or_default() };
        let mut restyled = Vec::with_capacity(specs.len());
        for spec in specs {
            if let Some(layer) = drawn(spec).or_else(|| placed.iter().find(|l| l.spec.source == spec.source)) {
                restyled.push(Layer { spec: spec.clone(), paths: layer.paths.clone() });
                continue;
            }
            let paths = match carried(spec) {
                Some(paths) => paths,
                None => self.curves(spec)?,
            };
            restyled.push(Layer { spec: spec.clone(), paths });
        }
        Ok(SurfaceDrawing { layers: restyled, ..self.clone() })
    }

    /// This drawing, of an edited request, with the layers the drawing
    /// before it kept: each stays as it was styled -- its levels, its width,
    /// a contour the request does not draw -- but for the curves the request
    /// itself describes, its coordinate curves, linear curve, trajectory and
    /// folds, which are this request's: where it still draws them, and
    /// added where it now does. `previous`, the drawing the edit was made
    /// from, lends the contours of a coordinate it drew on the same
    /// parameter rectangle (`restyled_carrying`).
    pub fn keeping(
        &self,
        kept: &[LayerSpec],
        previous: Option<&SurfaceDrawing>,
    ) -> Result<SurfaceDrawing, DrawingError> {
        // The request's own curves as it draws them, whatever a restyle drew.
        let built: Vec<LayerSpec> = self.own.iter().map(|l| l.spec.clone()).collect();
        let mut specs: Vec<LayerSpec> = kept
            .iter()
            .filter_map(|spec| {
                if SurfaceDrawing::describes_request(spec) {
                    built.iter().find(|b| b.name == spec.name && SurfaceDrawing::describes_request(b)).cloned()
                } else {
                    Some(spec.clone())
                }
            })
            .collect();
        let added: Vec<LayerSpec> = built
            .iter()
            .filter(|spec| SurfaceDrawing::describes_request(spec) && !specs.iter().any(|s| s.name == spec.name))
            .cloned()
            .collect();
        specs.extend(added);
        self.restyled_carrying(&specs, previous)
    }

    /// Whether a layer is one of the curves a request describes in itself
    /// rather than as a layer: its coordinate curves, linear curve,
    /// trajectory or folds, as `build` names them.
    pub fn describes_request(spec: &LayerSpec) -> bool {
        matches!(
            (spec.name.as_str(), &spec.source),
            ("lines", LayerSource::CoordinateCurves { .. })
                | ("linearCurves", LayerSource::LinearCurves { .. })
                | ("trajectory", LayerSource::Trajectory)
                | ("folds", LayerSource::FoldCurves)
        )
    }

    /// The curves of a layer the request does not draw, on this surface.
    fn curves(&self, spec: &LayerSpec) -> Result<PolylineSet<WorldSpace>, DrawingError> {
        if let Some(why) = self.request.unsupported(&spec.source) {
            return Err(ManifestError::Unsupported(format!("layer '{}': {why}", spec.name)).into());
        }
        let surface = &self.surface;
        match &spec.source {
            LayerSource::CoordinateCurves { u, v } => {
                Ok(surface.coordinate_curves((*u as usize, *v as usize), self.request.line_resolution() as usize))
            }
            LayerSource::LinearCurves { slope, turns, count } => {
                Ok(surface.linear_curves(*slope, *turns as usize, *count as usize, None))
            }
            LayerSource::FoldCurves => Ok(PolylineSet::empty()),
            LayerSource::Trajectory => {
                let (Some(torus), Shape::Forced { turns, .. }) = (self.torus(), &self.request.shape) else {
                    return Ok(PolylineSet::empty());
                };
                Ok(torus.trajectory(surface, *turns as usize))
            }
            LayerSource::Contour { field: ContourField::Isophote, light: None, .. } => {
                // Refused above (`SurfaceManifest::unsupported`), and here again
                // in so many words: never drawn as another field's contours.
                Err(ManifestError::Unsupported(format!(
                    "layer '{}': an isophote with no light to read n · l by",
                    spec.name
                ))
                .into())
            }
            LayerSource::Contour {
                field: ContourField::Isophote,
                levels,
                faces: FaceSelector::Surface,
                light: Some(light),
                ..
            } => {
                // A forced torus bounds a solid only where its revolution is
                // embedded, which its build decides.
                if surface.outward.is_none() {
                    return Err(ManifestError::Unsupported(format!(
                        "layer '{}': isophotes on a surface that bounds no solid, which has no outward normal",
                        spec.name
                    ))
                    .into());
                }
                Ok(surface.isophotes(*light, levels))
            }
            LayerSource::Hatching { hatching, faces: FaceSelector::Surface } => {
                if matches!(hatching.tone, Tone::HalfLambert { .. }) && surface.outward.is_none() {
                    return Err(ManifestError::Unsupported(format!(
                        "layer '{}': half-Lambert tone on a surface that bounds no solid, which has no outward normal",
                        spec.name
                    ))
                    .into());
                }
                // Every width step's lines in one set, with their coordinates.
                let mut paths: Vec<Vec<P3<WorldSpace>>> = Vec::new();
                let mut coords = Vec::new();
                for (_, step) in surface.coordinate_hatching(hatching) {
                    for k in 0..step.count() {
                        paths.push(step.path(k).to_vec());
                        coords.push(step.coords_of_path(k).expect("a surface's hatching carries coordinates").to_vec());
                    }
                }
                Ok(PolylineSet::from_paths_with_coords(&paths, &coords))
            }
            LayerSource::IntegralCurves { field, seeds, faces: FaceSelector::Surface } => {
                // Lines of curvature, read off the map's second derivatives,
                // which sn in relief has not (refused above).
                Ok(surface.integral_curves(field, seeds, &Limits::default())?)
            }
            LayerSource::Contour { field, levels, faces: FaceSelector::Surface, .. } if !field.is_field_of_f() => {
                let Some(named) = surface.field(*field) else {
                    return Err(ManifestError::Unsupported(format!(
                        "layer '{}': a contour of {} on a parametric surface",
                        spec.name,
                        field.raw()
                    ))
                    .into());
                };
                let face = surface.face();
                let value = named.value.clone();
                let coords: Vec<Vec<_>> = face
                    .contours(&ScalarField::new(named.period, move |p| value(p)), levels, &face.copies[0])
                    .into_iter()
                    .flat_map(|(_, paths)| paths)
                    .collect();
                let paths: Vec<Vec<P3<WorldSpace>>> = coords
                    .iter()
                    .map(|c| c.iter().map(|&p| P3::from_vec((surface.map)(p).position)).collect())
                    .collect();
                Ok(PolylineSet::from_paths_with_coords(&paths, &coords))
            }
            other => Err(ManifestError::Unsupported(format!(
                "layer '{}': a {} on a parametric surface",
                spec.name,
                other.kind()
            ))
            .into()),
        }
    }
}

/// A surface the gallery offers: a name to ask for it by, a label, a formula
/// and a note to show, and the request it opens as.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePreset {
    pub name: String,
    pub label: String,
    pub formula: String,
    pub notes: String,
    pub request: SurfaceRequest,
}

impl SurfacePreset {
    pub fn new(name: &str, label: &str, formula: &str, notes: &str, request: SurfaceRequest) -> SurfacePreset {
        let mut request = request;
        request.name = name.to_string();
        SurfacePreset {
            name: name.to_string(),
            label: label.to_string(),
            formula: formula.to_string(),
            notes: notes.to_string(),
            request,
        }
    }

    /// Named as `kurven-cli surface` names them, and built with its defaults,
    /// so a preset and the CLI's drawing of the same name are the same drawing.
    pub fn catalog() -> Vec<SurfacePreset> {
        vec![
            SurfacePreset::new(
                "torus",
                "Torus",
                "R = 2, r = 1",
                "The torus of revolution, ruled by its two families of circles.",
                SurfaceRequest {
                    lines: Some((36, 18)),
                    ..SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 })
                },
            ),
            SurfacePreset::new(
                "sn",
                "sn on its torus",
                "sn(z, 0.64)",
                "Jacobi's sn is doubly periodic: its rectangle of periods, rolled up both ways, is a torus, and its \
                 contours close on it.",
                SurfaceRequest::new(Shape::Sn { modulus: 0.64, major: 2.0, minor: 1.0, grid: 600, conformal: false }),
            ),
            SurfacePreset::new(
                "forced",
                "Forced oscillator",
                "x‴ = −αx′x″ − (ω₀² + 3βx²)x′ + A sin ωt",
                "A forced jerk oscillator's invariant torus, found in the spectrum of one long run and fitted, with \
                 the trajectory linear curve round it.",
                // 344 turns: a convergent denominator of the rotation
                // number, where the trajectory cuts the torus evenly.
                SurfaceRequest::new(Shape::Forced {
                    system: "jerk".to_string(),
                    fit: 8000.0,
                    harmonics: 24,
                    radial: 0,
                    axial: 1,
                    radius: 2.5,
                    turns: 344,
                }),
            ),
        ]
    }

    pub fn named(name: &str) -> Option<SurfacePreset> {
        SurfacePreset::catalog().into_iter().find(|p| p.name == name)
    }
}

/// Why a function cannot be painted on a surface, in words a field can show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionRefusal(pub String);

impl fmt::Display for FunctionRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FunctionRefusal {}

/// KurvenDynamics's extension of `SurfaceRequest`: what it knows of the
/// surface the request builds.
pub trait SurfaceRequestExt {
    /// The sides of the surface's parameter rectangle.
    fn rectangle(&self) -> (f64, f64);
    /// Why a layer cannot be drawn on the surface this request builds.
    fn unsupported(&self, source: &LayerSource) -> Option<String>;
    /// Whether a function can be painted on the surface.
    fn takes_function(&self) -> bool;
    /// The function's period rectangle, or the plain torus's turn each way.
    fn periods(&self) -> Option<(f64, f64)>;
    /// This request with `expression` painted on its torus.
    fn painting(&self, expression: &str, periods: (f64, f64)) -> Result<SurfaceRequest, FunctionRefusal>;
}

/// sn's rectangle of periods, [0, 4K) × [0, 2K').
fn sn_periods(m: f64) -> (f64, f64) {
    (4.0 * Jacobi::quarter_period(m), 2.0 * Jacobi::quarter_period(1.0 - m))
}

/// The rectangle of periods of an expression whose periods the catalog
/// knows: sn(z, m) for a number 0 < m < 1, whose rectangle is
/// [0, 4K) × [0, 2K'), as `surface sn` draws it. None for any other.
pub fn periods_of(expression: &str) -> Option<(f64, f64)> {
    let Ok(Node::Call(name, arguments)) = Expression::parse(expression) else { return None };
    match (name.as_str(), &arguments[..]) {
        ("sn", [Node::Variable, Node::Number(m)]) if *m > 0.0 && *m < 1.0 => Some(sn_periods(*m)),
        _ => None,
    }
}

impl SurfaceRequestExt for SurfaceRequest {
    /// A whole turn each way on a torus of revolution and a forced torus,
    /// the rectangle of periods of sn or a periodic function, and in relief
    /// sn's rectangle times its repeats.
    fn rectangle(&self) -> (f64, f64) {
        let two_pi = 2.0 * core::f64::consts::PI;
        match &self.shape {
            Shape::Torus { .. } | Shape::Forced { .. } => (two_pi, two_pi),
            Shape::Sn { modulus, .. } => sn_periods(*modulus),
            Shape::Periodic { periods, .. } => *periods,
            Shape::Relief { modulus, repeats, .. } => {
                let (a, b) = sn_periods(*modulus);
                (a * repeats.0 as f64, b * repeats.1 as f64)
            }
        }
    }

    /// What its manifest refuses (`SurfaceManifest::unsupported`), and
    /// hatching whose lines would hold more vertices than a layer does, a
    /// line a vertex at every sample along it (`sample_bound`) and a line
    /// every spacing across. Asked before an edit is taken, so the window
    /// refuses what the build would, and by the build itself.
    fn unsupported(&self, source: &LayerSource) -> Option<String> {
        if let Some(why) = SurfaceManifest::unsupported(source, self) {
            return Some(why);
        }
        let LayerSource::Hatching { hatching, faces: FaceSelector::Surface } = source else { return None };
        let axis = DomainAxis::from_angle(hatching.angle)?;
        let along_u = axis == DomainAxis::Real;
        let rectangle = self.rectangle();
        let lines =
            swift::max(1.0, ((if along_u { rectangle.1 } else { rectangle.0 }).abs() / hatching.spacing).round());
        let bound = self.sample_bound();
        let vertices = lines * ((if along_u { bound.0 } else { bound.1 }) + 1) as f64;
        if !(vertices > SurfaceRequest::MAX_VERTICES as f64) {
            return None;
        }
        Some(format!(
            "hatching every {} would take {} vertices, more than the {} a layer holds",
            kurven_math::format::description(hatching.spacing),
            crate::text::g(vertices, 3),
            SurfaceRequest::MAX_VERTICES
        ))
    }

    /// A round torus, plain or carrying one already. A forced torus is
    /// fitted to a flow, and sn in relief is built from sn.
    fn takes_function(&self) -> bool {
        matches!(self.shape, Shape::Torus { .. } | Shape::Sn { .. } | Shape::Periodic { .. })
    }

    fn periods(&self) -> Option<(f64, f64)> {
        let two_pi = 2.0 * core::f64::consts::PI;
        match &self.shape {
            Shape::Torus { .. } => Some((two_pi, two_pi)),
            Shape::Sn { modulus, .. } => Some(sn_periods(*modulus)),
            Shape::Periodic { periods, .. } => Some(*periods),
            Shape::Forced { .. } | Shape::Relief { .. } => None,
        }
    }

    /// This request with `expression` painted on its torus over the period
    /// rectangle [0, periods.0) × [0, periods.1), as `kurven-cli surface
    /// periodic --expression E --periods P1,P2` asks for it: the radii and
    /// the grid kept, f sampled as the catalog's sn is, unless a function
    /// was sampled before, and then as that one was. The surface then has
    /// |f| and arg f to contour. A conformal sn's torus is kept as a round
    /// torus of its radii, in its angles: the pin was sn's rectangle's, and
    /// another function's has another aspect. Refused, saying why: a surface that takes
    /// no function, an expression the language does not parse, and what
    /// `check` refuses, a period that is not positive among it.
    fn painting(&self, expression: &str, periods: (f64, f64)) -> Result<SurfaceRequest, FunctionRefusal> {
        let catalog = match SurfacePreset::named("sn").map(|p| p.request.shape) {
            Some(Shape::Sn { grid, .. }) => grid,
            _ => 600,
        };
        let (major, minor, samples) = match &self.shape {
            Shape::Torus { major, minor } => (*major, *minor, catalog),
            Shape::Sn { major, minor, grid, .. } | Shape::Periodic { major, minor, grid, .. } => {
                (*major, *minor, *grid)
            }
            Shape::Forced { .. } => {
                return Err(FunctionRefusal(
                    "a function is painted on a round torus; a forced torus is fitted to its flow".to_string(),
                ));
            }
            Shape::Relief { .. } => {
                return Err(FunctionRefusal(
                    "a function is painted on a round torus; sn in relief is built from sn".to_string(),
                ));
            }
        };
        let text = expression.trim();
        if text.is_empty() {
            return Err(FunctionRefusal("the expression is empty".to_string()));
        }
        if let Err(error) = Expression::compile(text) {
            return Err(FunctionRefusal(format!("{text}: {error}")));
        }
        let mut next = self.clone();
        next.shape = Shape::Periodic { expression: text.to_string(), periods, major, minor, grid: samples };
        next.name = String::new();
        let words = |key: &str| -> String {
            match key {
                "shape.periods" => "a period",
                "shape.expression" => "the expression",
                "shape.grid" => "the function's samples",
                "grid" => "the grid",
                other => other,
            }
            .to_string()
        };
        match next.check(&words) {
            Ok(()) => Ok(next),
            Err(ManifestError::Invalid(why)) => Err(FunctionRefusal(why)),
            Err(other) => Err(FunctionRefusal(other.to_string())),
        }
    }
}
