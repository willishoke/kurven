//! Painting curves on the picked face, the "Curves on" section's model: the
//! port of `Sources/KurvenApp/Paint.swift`.
//!
//! A layer names the faces it lies on by a face selector (`FaceSelector`),
//! which names a kind of face, not one face: the graph, every plateau, listed
//! walls, every riser, the surface, every face of a solid. So a layer painted
//! on one plateau lies on all of them, and one painted on a wall lies on that
//! wall, which a wall's selector can name alone. Every edit goes through
//! `edit` or a drag, so each is one undo step.

use std::fmt;

use kurven_core::face::{Face, FaceID};
use kurven_core::hatching::FaceKind;
use kurven_core::hatching::{FaceSelector, Hatching, Tone};
use kurven_core::manifest::{
    Affine2, ContourField, DirectionField, DomainAxis, HeightPolicy, LayerRole, LayerSource, LayerSpec,
    PrincipalCurvature, Restriction, Seeds, Spacing,
};
use kurven_core::pick::FacePick;
use kurven_core::scene::SceneGeometry;
use kurven_core::simd::Vec3;
use kurven_core::spaces::P2;
use kurven_core::surface_request::{Style, SurfaceRequest};
use kurven_landscape::landscape::{LandscapeStyle, NativeLandscape};
use kurven_landscape::torus::PeriodicTorus;
use kurven_surfaces::drawing::{SurfaceDrawing, SurfaceRequestExt, periods_of};

use crate::document::{Document, Opened, State};

/// Why a paint edit was refused.
#[derive(Debug, Clone, PartialEq)]
pub struct PaintRefusal(pub String);

impl fmt::Display for PaintRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PaintRefusal {}

/// A curve the Add menu offers on a face: a layer as the catalog would draw
/// it there, at the catalog's levels or spacing.
#[derive(Debug, Clone, PartialEq)]
pub struct PaintChoice {
    pub label: String,
    pub spec: LayerSpec,
}

impl PaintChoice {
    /// How `--add` names it: the layer's name.
    pub fn id(&self) -> &str {
        &self.spec.name
    }
}

/// The levels an isophote layer is added at: n · l at every quarter strictly
/// between −1 and 1.
pub const ISOPHOTE_LEVELS: [f64; 7] = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];

/// The light an isophote or a half-Lambert tone is added under: the one
/// `kurven-cli surface solid --hatching half-lambert` draws a solid under.
pub const LIGHT: Vec3 = Vec3 { x: -0.4, y: -0.6, z: 0.7 };

/// The angles of arg f slope lines are added at: every 30°, from −π.
pub fn slope_angles() -> Vec<f64> {
    let pi = core::f64::consts::PI;
    (0..12).map(|k| -pi + k as f64 * pi / 6.0).collect()
}

impl Document {
    // MARK: - the layers on a face

    /// The indices into the edited layers of the layers that lie on the
    /// picked face. A layer whose selector is not tiled lies only on the
    /// copy in place, on the identity tile; one on listed walls, only on
    /// those.
    pub fn painted_layers(&self, pick: &FacePick) -> Vec<usize> {
        let (Some(paint), Some(scene)) = (self.edited(), &self.scene) else { return Vec::new() };
        (0..paint.layers.len()).filter(|&i| self.lies(&paint.layers[i], pick, scene.geometry())).collect()
    }

    fn lies(&self, spec: &LayerSpec, pick: &FacePick, geometry: &SceneGeometry) -> bool {
        // The fold curves are the graph's or the surface's, though their
        // image is the outline and no face holds them.
        if matches!(spec.source, LayerSource::FoldCurves) {
            return matches!(pick.face.kind(), FaceKind::Graph | FaceKind::Surface);
        }
        // A contour whose selector names the surface lies on it, the
        // relief's truncation edge among them, though its role is an
        // outline's and visibility treats it as one.
        if let LayerSource::Contour { faces: FaceSelector::Surface, .. } = spec.source {
            return pick.face == FaceID::Surface;
        }
        if geometry.face_of(spec) != Some(pick.face.kind()) {
            return false;
        }
        match &spec.source {
            LayerSource::Contour { faces: FaceSelector::Walls { edges, .. }, .. }
            | LayerSource::Hatching { faces: FaceSelector::Walls { edges, .. }, .. }
            | LayerSource::WallEdges { edges, .. } => {
                let FaceID::Wall { edge } = pick.face else { return false };
                edges.contains(&(edge as i64))
            }
            LayerSource::Contour { faces, .. }
            | LayerSource::Hatching { faces, .. }
            | LayerSource::IntegralCurves { faces, .. } => {
                // A riser at a tile boundary stands between two copies, which
                // only risers selected on every tile reach.
                if let FaceID::TileBoundaryRiser { .. } = pick.face {
                    return tiled(faces) == Some(true);
                }
                tiled(faces).unwrap_or(true) || self.on_identity_tile(pick)
            }
            _ => true,
        }
    }

    fn on_identity_tile(&self, pick: &FacePick) -> bool {
        let Some(tiles) = self.bundle().map(|b| &b.manifest.occluder.tiles) else { return true };
        match tiles.get(pick.copy) {
            Some(tile) => *tile == Affine2::IDENTITY,
            None => true,
        }
    }

    /// Whether what is painted on a landscape lands on every tile.
    fn paints_every_tile(&self) -> bool {
        self.bundle().map(|b| b.manifest.occluder.tiles.len()).unwrap_or(1) > 1
    }

    /// "Curves on" what: the faces a layer added here lies on.
    pub fn paint_title(&self, pick: &FacePick) -> String {
        match pick.face {
            FaceID::Graph => "the graph".to_string(),
            FaceID::Plateau { .. } => "the plateaus".to_string(),
            FaceID::Riser { .. } | FaceID::TileBoundaryRiser { .. } => "the risers".to_string(),
            FaceID::Wall { edge } => format!("wall {edge}"),
            FaceID::Surface => "the surface".to_string(),
            FaceID::Solid { .. } => "the solid's faces".to_string(),
        }
    }

    /// What the picked face is, and what a layer added there lies on
    /// (`paintCaption`).
    pub fn paint_caption(&mut self, pick: &FacePick) -> String {
        let count =
            self.faces_now().map_or(0, |faces| faces.iter().filter(|f| f.id.kind() == pick.face.kind()).count());
        let tiles = if self.paints_every_tile() { ", on every tile" } else { "" };
        match pick.face {
            FaceID::Graph => format!("Picked the graph. A layer added here lies on the graph{tiles}."),
            FaceID::Plateau { band } => format!(
                "Picked plateau {band}. A layer added here lies on {}{tiles}.",
                if count == 1 {
                    "the whole plateau, wherever |f| reaches the truncation".to_string()
                } else {
                    format!("all {count} plateaus")
                }
            ),
            FaceID::Riser { .. } | FaceID::TileBoundaryRiser { .. } => format!(
                "Picked {pick}. A layer added here lies on {}{tiles}.",
                if count == 1 { "it".to_string() } else { format!("all {count} risers") }
            ),
            FaceID::Wall { edge } => format!(
                "Picked wall {edge}. A layer added here lies on wall {edge} alone; a layer listed may lie on other \
                 walls too."
            ),
            FaceID::Surface => "Picked the surface, which is one face.".to_string(),
            FaceID::Solid { face } => {
                format!("Picked face {face}. Hatching added here lies on all {count} faces of the solid.")
            }
        }
    }

    /// Whether "Function…" is offered on the pick: on a surface whose
    /// request takes a function painted on it.
    pub fn offers_function(&self, pick: &FacePick) -> bool {
        pick.face == FaceID::Surface && self.edited().and_then(|e| e.request).is_some_and(|r| r.takes_function())
    }

    // MARK: - adding

    /// What the Add menu offers on the picked face: a contour of each field
    /// the face exposes, hatching on a flat face, the fold curves where
    /// there are none, each as the catalog draws it -- |f| and arg f at the
    /// catalog's levels, hatching at its spacing -- and a coordinate or the
    /// height at seven levels across the face. Hatching is not laid on a
    /// curved face, and a polyhedron's faces take hatching alone.
    pub fn paint_choices(&mut self, pick: &FacePick) -> Vec<PaintChoice> {
        if !self.is_current() {
            return Vec::new();
        }
        let (Some(paint), Some(content)) = (self.edited(), self.scene.as_ref().map(|s| s.content())) else {
            return Vec::new();
        };
        let Some(faces) = self.faces_now() else { return Vec::new() };
        let Some(face) = faces.iter().find(|f| f.id == pick.face) else { return Vec::new() };
        let has = |source: &LayerSource| paint.layers.iter().any(|l| l.source == *source);
        let mut choices: Vec<PaintChoice> = Vec::new();
        let contours = |choices: &mut Vec<PaintChoice>, fields: &[ContourField], faces: FaceSelector| {
            for &field in fields {
                if faces.unsupported_field(field).is_some() {
                    continue;
                }
                let levels = levels_of(field, face);
                if levels.len() < 2 {
                    continue;
                }
                choices.push(PaintChoice {
                    label: format!("Contours of {}", words(field)),
                    spec: LayerSpec::new(
                        &format!("{}_{}", faces.name(), field.raw()),
                        LayerRole::Scaffold,
                        LayerSource::Contour {
                            field,
                            levels,
                            restriction: Restriction::All,
                            faces: faces.clone(),
                            light: None,
                        },
                        0.2,
                        HeightPolicy::Surface,
                    ),
                });
            }
        };
        let catalog = |choices: &mut Vec<PaintChoice>, spec: LayerSpec, label: &str| {
            choices.push(PaintChoice { label: label.to_string(), spec });
        };
        // Isophotes at seven levels of n · l, and half-Lambert tone along u
        // at `spacing`, under the light a half-Lambert solid is drawn under:
        // on a curved face with an outward side.
        let shading = |choices: &mut Vec<PaintChoice>, faces: FaceSelector, spacing: f64| {
            if face.is_flat()
                || face.copies.first().and_then(|c| c.outward).is_none()
                || !(spacing > 0.0)
                || !spacing.is_finite()
            {
                return;
            }
            catalog(
                choices,
                LayerSpec::new(
                    &format!("{}_isophote", faces.name()),
                    LayerRole::Scaffold,
                    LayerSource::Contour {
                        field: ContourField::Isophote,
                        levels: ISOPHOTE_LEVELS.to_vec(),
                        restriction: Restriction::All,
                        faces: faces.clone(),
                        light: Some(LIGHT),
                    },
                    0.2,
                    HeightPolicy::Surface,
                ),
                "Isophotes",
            );
            catalog(
                choices,
                LayerSpec::new(
                    &format!("{}_tone", faces.name()),
                    LayerRole::Scaffold,
                    LayerSource::Hatching {
                        hatching: Hatching::new(0.0, spacing, None, Tone::HalfLambert { light: LIGHT }, 4),
                        faces,
                    },
                    0.4,
                    HeightPolicy::Surface,
                ),
                "Half-Lambert tone",
            );
        };

        match &self.state {
            State::Ready(Opened::Landscape(bundle)) => {
                let m = bundle.manifest.clone();
                let tiled = self.paints_every_tile();
                // The ceiling the build chose the levels under: the
                // truncation's, or with none, read off the samples as the
                // build reads it, once.
                let ceiling = match LandscapeStyle::ceiling(&m.truncation) {
                    Some(c) => c,
                    None => match self.paint_ceiling {
                        Some((held, value)) if held == content => value,
                        _ => {
                            let magnitudes: Vec<f64> = bundle.surface.height.values.iter().map(|&v| v as f64).collect();
                            let value = NativeLandscape::ceiling(&m.truncation, &magnitudes);
                            self.paint_ceiling = Some((content, value));
                            value
                        }
                    },
                };
                let bundle = self.bundle().cloned().expect("a landscape");
                let defaults = NativeLandscape::default_layers(
                    m.domain,
                    &m.truncation,
                    ceiling,
                    None,
                    bundle.surface.phase.is_some(),
                );
                let template = |name: &str| defaults.iter().find(|l| l.name == name).cloned();
                match pick.face {
                    FaceID::Graph => {
                        for (name, label) in [
                            ("mag_major", "|f| at the major levels"),
                            ("mag_minor", "|f| at the minor levels"),
                            ("ang_major", "arg f at the major levels"),
                            ("ang_minor", "arg f at the minor levels"),
                        ] {
                            let Some(mut spec) = template(name) else { continue };
                            let LayerSource::Contour { field, levels, restriction, light, .. } = &spec.source else {
                                continue;
                            };
                            spec.source = LayerSource::Contour {
                                field: *field,
                                levels: levels.clone(),
                                restriction: restriction.clone(),
                                faces: FaceSelector::Graph { tiled },
                                light: *light,
                            };
                            catalog(&mut choices, spec, label);
                        }
                        contours(
                            &mut choices,
                            &[ContourField::U, ContourField::V, ContourField::Height],
                            FaceSelector::Graph { tiled },
                        );
                        // Hatched at the plateaus' spacing, so its lines
                        // continue theirs across the truncation edge.
                        let plateau = match template("plateau_hatch").map(|t| t.source) {
                            Some(LayerSource::Hatching { hatching, .. }) => Some(hatching.spacing),
                            _ => None,
                        };
                        let spacing = plateau.unwrap_or_else(|| {
                            kurven_math::swift::max(m.domain.real.length().abs(), m.domain.imag.length().abs()) / 70.0
                        });
                        shading(&mut choices, FaceSelector::Graph { tiled }, spacing);
                        // Slope lines of |f| from the contour |f| = 1, at
                        // every 30° of arg f, which only f itself can place.
                        if bundle.refine.as_ref().is_some_and(|r| r.phase.is_some()) {
                            catalog(
                                &mut choices,
                                LayerSpec::new(
                                    "graph_slope_lines",
                                    LayerRole::Scaffold,
                                    LayerSource::IntegralCurves {
                                        field: DirectionField::Slope(ContourField::Magnitude),
                                        seeds: Seeds {
                                            field: ContourField::Magnitude,
                                            level: 1.0,
                                            spacing: Spacing::Values(ContourField::Phase, slope_angles()),
                                        },
                                        faces: FaceSelector::Graph { tiled },
                                    },
                                    0.3,
                                    HeightPolicy::Surface,
                                ),
                                "Slope lines of |f|",
                            );
                        }
                        if !has(&LayerSource::FoldCurves)
                            && let Some(folds) = template("folds")
                        {
                            catalog(&mut choices, folds, "Fold curves");
                        }
                    }
                    FaceID::Plateau { .. } => {
                        if let Some(mut spec) = template("plateau_hatch")
                            && let LayerSource::Hatching { hatching, .. } = &spec.source
                        {
                            spec.source =
                                LayerSource::Hatching { hatching: *hatching, faces: FaceSelector::Plateaus { tiled } };
                            catalog(&mut choices, spec, "Hatching along Re");
                        }
                        contours(&mut choices, &[ContourField::U, ContourField::V], FaceSelector::Plateaus { tiled });
                    }
                    FaceID::Wall { edge } => {
                        let walls = FaceSelector::Walls {
                            edges: vec![edge as i64],
                            end_lines: false,
                            base: 0.0,
                            top_offset: 0.0,
                        };
                        if let Some(mut spec) = template("wall_hatch")
                            && let LayerSource::Hatching { hatching, .. } = &spec.source
                        {
                            spec.source = LayerSource::Hatching { hatching: *hatching, faces: walls.clone() };
                            catalog(&mut choices, spec, "Hatching");
                        }
                        contours(&mut choices, &[ContourField::U, ContourField::V, ContourField::Height], walls);
                    }
                    FaceID::Riser { .. } | FaceID::TileBoundaryRiser { .. } => {
                        if let Some(wall) = template("wall_hatch")
                            && let LayerSource::Hatching { hatching, .. } = &wall.source
                        {
                            let mut spec = wall.clone();
                            spec.name = "riser_hatch".to_string();
                            spec.source = LayerSource::riser_hatching(hatching.spacing, tiled, false);
                            catalog(&mut choices, spec, "Hatching");
                        }
                        contours(
                            &mut choices,
                            &[ContourField::U, ContourField::V, ContourField::Height],
                            FaceSelector::Risers { tiled, end_lines: false },
                        );
                    }
                    FaceID::Surface | FaceID::Solid { .. } => {}
                }
            }
            State::Ready(Opened::Surface(_)) => {
                if let Some(request) = &paint.request {
                    if request.paints_function() {
                        let labels = [
                            ("mag_major", "|f| at the major levels"),
                            ("mag_minor", "|f| at the minor levels"),
                            ("ang_major", "arg f at the major levels"),
                            ("ang_minor", "arg f at the minor levels"),
                            ("truncation edge", "The truncation edge"),
                        ];
                        for spec in PeriodicTorus::contours(3.0, request.truncates_function()) {
                            let label = labels.iter().find(|(n, _)| *n == spec.name).map(|(_, l)| l.to_string());
                            let label = label.unwrap_or_else(|| spec.name.clone());
                            catalog(&mut choices, spec, &label);
                        }
                    }
                    contours(
                        &mut choices,
                        &[ContourField::U, ContourField::V, ContourField::Height],
                        FaceSelector::Surface,
                    );
                    shading(&mut choices, FaceSelector::Surface, face.domain.v.range.length().abs() / 48.0);
                    // Lines of curvature, each family through 24 seeds round a
                    // coordinate curve, where the map has second derivatives.
                    if face.second_order_jet.is_some() {
                        for (curvature, label) in [
                            (PrincipalCurvature::Greater, "Lines of curvature, greater"),
                            (PrincipalCurvature::Lesser, "Lines of curvature, lesser"),
                        ] {
                            catalog(
                                &mut choices,
                                LayerSpec::lines_of_curvature(
                                    curvature,
                                    (face.domain.u, face.domain.v),
                                    24,
                                    request.style.lines,
                                ),
                                label,
                            );
                        }
                    }
                    if request.lines.is_none() {
                        catalog(
                            &mut choices,
                            LayerSpec::new(
                                "lines",
                                LayerRole::Scaffold,
                                LayerSource::CoordinateCurves { u: 36, v: 18 },
                                request.style.lines,
                                HeightPolicy::Surface,
                            ),
                            "Coordinate curves",
                        );
                    }
                    if request.style.folds == 0.0 {
                        catalog(&mut choices, folds(Style::default().folds), "Fold curves");
                    }
                }
            }
            State::Ready(Opened::Polyhedron(bundle)) => {
                let width =
                    paint.layers.iter().find(|l| l.source == LayerSource::Edges).map(|l| l.width / 2.0).unwrap_or(0.3);
                let reach = bundle.solid().frames.iter().map(|f| f.extent).fold(None, |m: Option<f64>, e| {
                    Some(match m {
                        Some(m) => kurven_math::swift::max(m, e),
                        None => e,
                    })
                });
                catalog(
                    &mut choices,
                    LayerSpec::new(
                        "hatching",
                        LayerRole::Scaffold,
                        LayerSource::Hatching {
                            hatching: Hatching::plain(core::f64::consts::PI / 4.0, reach.unwrap_or(1.0) / 25.0),
                            faces: FaceSelector::Solid,
                        },
                        width,
                        HeightPolicy::Surface,
                    ),
                    "Hatching",
                );
            }
            State::Empty | State::Failed(..) => {}
        }
        // Nothing the face has already: an addition the same as a layer
        // present would draw the same curves twice.
        choices.into_iter().filter(|c| !paint.layers.iter().any(|l| l.source == c.spec.source)).collect()
    }

    /// Add `choice` to the picked face's layers: before the outline drawn
    /// last, under a name of its own. A surface's coordinate curves and fold
    /// curves are its request's, which the edit sets too.
    pub fn add(&mut self, choice: &PaintChoice) {
        self.paint_status = None;
        self.edit_infallible(&format!("Add {}", choice.label), |paint| {
            let mut spec = choice.spec.clone();
            let names: std::collections::HashSet<String> = paint.layers.iter().map(|l| l.name.clone()).collect();
            if names.contains(&spec.name) && (!SurfaceDrawing::describes_request(&spec) || paint.request.is_none()) {
                let base = spec.name.clone();
                spec.name = (2..).map(|k| format!("{base} {k}")).find(|n| !names.contains(n)).expect("a free name");
            }
            if let Some(request) = &mut paint.request
                && SurfaceDrawing::describes_request(&spec)
            {
                match &spec.source {
                    LayerSource::CoordinateCurves { u, v } => request.lines = Some((*u, *v)),
                    LayerSource::FoldCurves => request.style.folds = spec.width,
                    _ => {}
                }
            }
            let outline = paint.layers.iter().rev().take_while(|l| l.role == LayerRole::Outline).count();
            let at = if spec.role == LayerRole::Outline { paint.layers.len() } else { paint.layers.len() - outline };
            paint.layers.insert(at, spec);
        });
    }

    /// Take a layer off: one of the face's, by its index in the edited
    /// layers. A surface's coordinate curves, linear curve and fold curves
    /// go from its request with it; a forced torus's trajectory is the
    /// torus's own and stays.
    pub fn remove_layer(&mut self, index: usize) -> Result<(), PaintRefusal> {
        let Some(spec) = self.edited().and_then(|e| e.layers.get(index).cloned()) else { return Ok(()) };
        if spec.source == LayerSource::Trajectory {
            return Err(PaintRefusal("the trajectory is the forced torus's own: set its turns instead".to_string()));
        }
        self.paint_status = None;
        self.edit_infallible(&format!("Remove {}", spec.name), |paint| {
            if let Some(request) = &mut paint.request
                && SurfaceDrawing::describes_request(&spec)
            {
                match &spec.source {
                    LayerSource::CoordinateCurves { .. } => request.lines = None,
                    LayerSource::LinearCurves { .. } => request.linear_curves = None,
                    LayerSource::FoldCurves => request.style.folds = 0.0,
                    _ => {}
                }
            }
            paint.layers.remove(index);
            paint.level_counts.remove(&spec.name);
        });
        Ok(())
    }

    /// Paint a function on the surface (`SurfaceRequestExt::painting`): its
    /// |f| and arg f, whose contours can then be added. Refused, saying why,
    /// and nothing changes.
    pub fn add_function(&mut self, expression: &str, periods: (f64, f64)) -> Result<(), PaintRefusal> {
        let Some(request) = self.edited().and_then(|e| e.request) else {
            return Err(PaintRefusal("a function is painted on a parametric surface".to_string()));
        };
        let next = request.painting(expression, periods).map_err(|e| PaintRefusal(e.to_string()))?;
        self.paint_status = None;
        self.edit_infallible(&format!("Function {expression}"), |e| e.request = Some(next));
        Ok(())
    }

    /// The periods "Function…" offers for `expression`: its rectangle where
    /// the catalog knows it (`periods_of`), else the surface's own.
    pub fn function_periods(&self, expression: &str) -> (f64, f64) {
        let request = self.edited().and_then(|e| e.request);
        // The function the surface carries keeps the rectangle it is painted
        // over; the catalog's is offered for a new one.
        if let Some(SurfaceRequest {
            shape: kurven_core::surface_request::Shape::Periodic { expression: painted, periods, .. },
            ..
        }) = &request
            && painted.trim() == expression.trim()
        {
            return *periods;
        }
        let two_pi = 2.0 * core::f64::consts::PI;
        periods_of(expression).or_else(|| request.as_ref().and_then(|r| r.periods())).unwrap_or((two_pi, two_pi))
    }

    /// Whether what is on screen is what is asked for: nothing building,
    /// sampling, restyling or waiting to. The Add menu is offered only then,
    /// since its levels are read off the drawing.
    pub fn is_current(&self) -> bool {
        !self.building
            && !self.sampling
            && !self.drawing
            && self.surface_wanted.is_none()
            && self.wanted.is_none()
            && self.styled.is_none()
            && self.surface_styled.is_none()
            && self.polyhedron_styled.is_none()
            && self.curves_wanted.is_none()
    }

    // MARK: - sliders

    /// Set a layer's level count as its slider does: a step of a drag, or,
    /// with none under way, an edit of its own.
    pub fn paint_levels(&mut self, count: i64, name: &str, draft: bool) {
        let Some(index) = self.layer_specs().iter().position(|s| s.name == name) else { return };
        if self.edited_before_drag.is_none() {
            let name = name.to_string();
            self.edit_infallible(&format!("Levels of {name}"), |e| {
                e.level_counts.insert(name.clone(), count.max(2));
            });
        } else {
            self.set_level_count(count, index, draft);
        }
    }

    /// Set a hatching's strokes as its slider does.
    pub fn paint_strokes(&mut self, count: i64, name: &str) {
        let Some(index) = self.edited().and_then(|e| e.layers.iter().position(|l| l.name == name)) else { return };
        self.editing(&format!("Strokes of {name}"), |d| d.set_hatch_count(count, index));
    }

    /// Put a layer's levels back to its own.
    pub fn paint_reset_levels(&mut self, name: &str) {
        let name = name.to_string();
        self.edit_infallible(&format!("Reset Levels of {name}"), |e| {
            e.level_counts.remove(&name);
        });
    }

    /// Set the light a layer is read under (`LayerSource::lit`) as its
    /// sliders do: a step of a drag, or, with none under way, an edit of its
    /// own.
    pub fn paint_light(&mut self, light: Vec3, name: &str) {
        let name = name.to_string();
        self.editing(&format!("Light of {name}"), |d| {
            let Some(mut next) = d.edited() else { return };
            let Some(index) = next.layers.iter().position(|l| l.name == name) else { return };
            let Some(source) = next.layers[index].source.lit(light) else { return };
            next.layers[index].source = source;
            d.set_edited(next);
        });
    }

    /// Run the plateaus' hatching along an axis.
    pub fn paint_axis(&mut self, axis: DomainAxis) {
        let name = format!("Hatching along {}", if axis == DomainAxis::Real { "Re" } else { "Im" });
        self.editing(&name, |d| d.set_plateau_hatch_axis(axis));
    }
}

fn tiled(faces: &FaceSelector) -> Option<bool> {
    match faces {
        FaceSelector::Graph { tiled } | FaceSelector::Plateaus { tiled } | FaceSelector::Risers { tiled, .. } => {
            Some(*tiled)
        }
        FaceSelector::Walls { .. } | FaceSelector::Surface | FaceSelector::Solid => None,
    }
}

fn folds(width: f64) -> LayerSpec {
    LayerSpec::new("folds", LayerRole::Outline, LayerSource::FoldCurves, width, HeightPolicy::Surface)
}

fn words(field: ContourField) -> &'static str {
    match field {
        ContourField::Magnitude => "|f|",
        ContourField::Phase => "arg f",
        ContourField::U => "u",
        ContourField::V => "v",
        ContourField::Height => "the height",
        ContourField::Isophote => "n · l",
    }
}

/// Seven levels evenly across a field's range on a face, its ends left out:
/// a coordinate's from the face's parameter rectangle, the height's from the
/// face sampled where its first copy holds it.
pub fn levels_of(field: ContourField, face: &Face) -> Vec<f64> {
    let (u, v) = (face.domain.u, face.domain.v);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    match field {
        ContourField::Isophote => return ISOPHOTE_LEVELS.to_vec(),
        ContourField::U => (lo, hi) = (u.range.lo, u.range.hi),
        ContourField::V => (lo, hi) = (v.range.lo, v.range.hi),
        ContourField::Height | ContourField::Magnitude | ContourField::Phase => {
            let n = 32;
            for j in 0..=n {
                for i in 0..=n {
                    let p = P2::new(
                        u.range.lo + u.range.length() * i as f64 / n as f64,
                        v.range.lo + v.range.length() * j as f64 / n as f64,
                    );
                    if let Some(copy) = face.copies.first()
                        && !(copy.contains)(p)
                    {
                        continue;
                    }
                    let z = (face.map)(p).position.z;
                    if !z.is_finite() {
                        continue;
                    }
                    lo = kurven_math::swift::min(lo, z);
                    hi = kurven_math::swift::max(hi, z);
                }
            }
        }
    }
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Vec::new();
    }
    (1..=7).map(|k| lo + (hi - lo) * k as f64 / 8.0).collect()
}
