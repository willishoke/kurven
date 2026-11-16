//! Hatching, the faces a layer lies on, and the tones: the values of
//! `Sources/KurvenCore/Hatching.swift` that the bundle schema names.
//!
//! Laying a hatching on a solid (`Solid.hatchLines`, `visibleHatching`) is
//! drawing, and comes with the polyhedra (Phase 5).

use std::collections::{BTreeMap, HashSet};

use kurven_math::{real, swift};

use crate::json::{JsonError, JsonObject, JsonValue, Members, object, string};
use crate::manifest::{ContourField, DomainAxis, ManifestError, SchemaError};
use crate::polyline::PolylineSet;
use crate::simd::{Vec2, Vec3};
use crate::solid::{FaceFrame, Solid};
use crate::spaces::{P3, Transform, ViewSpace, WorldSpace};
use crate::text;

/// The kind of a face: what a selector picks and what a face is
/// (`Face.swift`'s `FaceKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceKind {
    Graph,
    Plateau,
    Riser,
    Wall,
    Surface,
    Solid,
}

/// Hatching: parallel lines at an angle and a spacing, each drawn at a width
/// set by a tone read along it.
///
/// kurven draws strokes only, never areas, so a face is shaded by hatching
/// it. The tone decides two things about each hatch line: where it is drawn,
/// and how wide. It is read along the line every `step`, and wherever the
/// width changes between two readings the change is bisected on the tone, so
/// a line ends on the tone's own level set to the last bit rather than at
/// the nearest reading. The lines are then tested for visibility like any
/// curve on a face (`Solid.visibleCurves`): the drawing depends on the
/// solid, the tone and the camera, and not on a resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hatching {
    /// The direction of the lines in the face's plane, in radians from its
    /// `u` axis (`Solid.FaceFrame`).
    pub angle: f64,
    /// Between neighbouring lines, in world units.
    pub spacing: f64,
    /// The distance between successive vertices along a line, or readings
    /// of its tone, in world units. On a solid the tone is read every
    /// `step`, and a change of width narrower than this can fall between two
    /// readings. On a wall or a riser a line is its two ends, and on a
    /// plateau, the graph or a parametric surface it has a vertex wherever
    /// it crosses a sample of the face's parameter rectangle, so the step is
    /// not read there, and a manifest writes none.
    /// The spacing by default, on every face, so the tone is read as finely
    /// as it is drawn, and no finer than `FINEST_STEP` of it.
    pub step: f64,
    pub tone: Tone,
    /// How many widths a line can have. One: a line is drawn at the layer's
    /// width where the tone is positive and nowhere else. More: the tone in
    /// `(0, 1]` is cut into that many steps and the width follows it,
    /// `k / widthSteps` of the layer's width on step `k`.
    pub width_steps: i64,
}

impl Hatching {
    /// The finest `step`, as a fraction of the spacing: a line is read at
    /// most a million times per spacing of its length.
    pub const FINEST_STEP: f64 = 1e-6;

    /// How far, in radians, an angle a face requires may be from the one
    /// written: a thousandth, so π/2 written as 1.571 is π/2.
    pub const ANGLE_TOLERANCE: f64 = 1e-3;

    /// - Precondition: a positive spacing, at least one width, a step no
    ///   finer than `FINEST_STEP` of the spacing, a positive Turing scale and
    ///   a light with a direction.
    pub fn new(angle: f64, spacing: f64, step: Option<f64>, tone: Tone, width_steps: i64) -> Hatching {
        assert!(spacing > 0.0 && spacing.is_finite(), "hatching has a positive spacing");
        assert!(width_steps >= 1, "hatching has at least one width");
        let step = step.unwrap_or(spacing);
        assert!(
            step >= Hatching::FINEST_STEP * spacing && step.is_finite(),
            "hatching reads its tone at most a million times a spacing"
        );
        if let Tone::Turing { scale, .. } = tone {
            assert!(scale > 0.0 && scale.is_finite(), "Turing's tone has a positive scale");
        }
        if let Tone::HalfLambert { light } = tone {
            assert!(Tone::has_direction(light), "a light has a direction");
        }
        Hatching { angle, spacing, step, tone, width_steps }
    }

    /// Uniform, one width, read every spacing: Swift's defaults.
    pub fn plain(angle: f64, spacing: f64) -> Hatching {
        Hatching::new(angle, spacing, None, Tone::Uniform, 1)
    }

    /// The width step a tone falls on: 0 for none, `widthSteps` for the
    /// full width.
    pub fn width_step(&self, tone: f64) -> i64 {
        if tone <= 0.0 || tone.is_nan() {
            return 0;
        }
        self.width_steps.min((tone * self.width_steps as f64).ceil() as i64)
    }

    /// Read from its fields: `angle`, `spacing`, `step` (the spacing when
    /// absent), `widthSteps` and `tone`. A `hatching` layer source holds them
    /// beside its `kind` and `faces` (`LayerSource`), so anything else in `o`
    /// is ignored.
    pub fn from_fields(o: &JsonObject) -> Result<Hatching, SchemaError> {
        let what = "LayerSource.hatching.tone";
        let t = o.value("tone", "LayerSource.hatching")?.object(what)?;
        let tone = match t.string("kind", what)?.as_str() {
            "uniform" => Tone::Uniform,
            "turing" => {
                let scale = t.double("scale", what)?;
                if !(scale > 0.0 && scale.is_finite()) {
                    return Err(JsonError::wrong_type("scale", "a positive number", what).into());
                }
                Tone::Turing { scale, seed: t.int("seed", what)? as u64 }
            }
            "halfLambert" => {
                let l = t.doubles("light", what)?;
                if l.len() != 3 || !Tone::has_direction(Vec3::new(l[0], l[1], l[2])) {
                    return Err(JsonError::wrong_type("light", "three numbers, not all zero", what).into());
                }
                Tone::HalfLambert { light: Vec3::new(l[0], l[1], l[2]) }
            }
            other => {
                return Err(ManifestError::unknown_kind(other, "Tone", &["uniform", "turing", "halfLambert"]).into());
            }
        };
        let spacing = o.double("spacing", "LayerSource.hatching")?;
        let width_steps = o.int("widthSteps", "LayerSource.hatching")?;
        if !(spacing > 0.0 && spacing.is_finite()) {
            return Err(JsonError::wrong_type("spacing", "a positive number", "LayerSource.hatching").into());
        }
        if width_steps < 1 {
            return Err(JsonError::wrong_type("widthSteps", "a positive integer", "LayerSource.hatching").into());
        }
        let step = o.optional_double("step");
        if let Some(step) = step
            && !(step >= Hatching::FINEST_STEP * spacing && step.is_finite())
        {
            return Err(JsonError::wrong_type(
                "step",
                "a number no finer than a millionth of the spacing",
                "LayerSource.hatching",
            )
            .into());
        }
        Ok(Hatching::new(o.double("angle", "LayerSource.hatching")?, spacing, step, tone, width_steps))
    }

    /// The fields `from_fields` reads.
    pub fn fields(&self) -> JsonObject {
        let tone = match self.tone {
            Tone::Uniform => object([("kind", string("uniform"))]),
            Tone::Turing { scale, seed } => object([
                ("kind", string("turing")),
                ("scale", JsonValue::Double(scale)),
                ("seed", JsonValue::Int(seed as i64)),
            ]),
            Tone::HalfLambert { light } => {
                object([("kind", string("halfLambert")), ("light", crate::json::doubles(&light.to_array()))])
            }
        };
        [
            ("angle", JsonValue::Double(self.angle)),
            ("spacing", JsonValue::Double(self.spacing)),
            ("step", JsonValue::Double(self.step)),
            ("widthSteps", JsonValue::Int(self.width_steps)),
            ("tone", tone),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
}

/// The faces a contour or a hatching lies on: a layer's face selector, for
/// the two kinds of layer whose curves can lie on more than one kind of face.
/// Every other kind implies its face (`SceneGeometry.face(of:)`).
///
/// For a hatching it also says how the lines are placed. One `Hatching`
/// serves every face; what differs is the face's coordinates and where the
/// first line falls. On a landscape's walls, risers and plateaus the spacing
/// is rounded to a whole number of gaps, so that a plateau's lines meet the
/// walls' and the risers' at a corner; on a solid the lines run through the
/// multiples of the spacing in each face's own frame, so that a face's
/// hatching does not depend on how it is partitioned.
#[derive(Debug, Clone, PartialEq)]
pub enum FaceSelector {
    /// A landscape's graph, where z = |f|, copied to every tile when
    /// `tiled`. It is curved, so a hatching there runs along a coordinate,
    /// as on a parametric surface (`Face.coordinateHatching(_:on:)`).
    Graph { tiled: bool },
    /// A truncated landscape's plateaus, copied to every tile when `tiled`.
    /// A hatching there is the contours of a coordinate
    /// (`Surface.plateauHatching(axis:spacing:tiled:)`): lines along the real
    /// axis (angle 0) or the imaginary axis (angle π/2, `DomainAxis.angle`),
    /// with a vertex at every grid line they cross, each ending where it
    /// reaches the truncation edge, by bisection on f.
    Plateaus { tiled: bool },
    /// A landscape's walls along the listed perimeter edges. A hatching's
    /// lines are vertical in a wall's coordinates (distance along its edge,
    /// height), `WALL_ANGLE`, from `base` to the top edge plus `top_offset`,
    /// each its two ends (`Surface.vertical`), at whole gaps along each wall;
    /// the lines at a wall's two ends are drawn only with `end_lines`. The
    /// three place hatch lines, and a contour takes their defaults.
    Walls { edges: Vec<i64>, end_lines: bool, base: f64, top_offset: f64 },
    /// A landscape's risers, where the drawn top steps inside the solid: at
    /// the band boundaries, on every tile when `tiled`, and then also at the
    /// tile boundaries between the copies; in place otherwise. A hatching's
    /// lines are placed as on the walls, at whole gaps along the line each
    /// riser stands on, from its bottom edge to its top
    /// (`Surface.riserHatchLines`), the lines at a stretch's two ends drawn
    /// only with `end_lines`, which a contour leaves false.
    Risers { tiled: bool, end_lines: bool },
    /// A parametric surface's one face. A parametric surface's manifest
    /// names it (`SurfaceManifest`); a landscape's refuses it (`LayerSource`).
    /// A hatching there runs along a coordinate of its parameter rectangle
    /// (D8): lines of v along u at angle 0, lines of u along v at π/2,
    /// spread to whole gaps across the other coordinate, the spacing in its
    /// units (`Face.coordinateHatching(_:on:)`).
    Surface,
    /// Every face of a solid, at any angle. A polyhedron's manifest names
    /// it (`PolyhedronManifest`); a landscape's and a parametric surface's
    /// refuse it.
    Solid,
}

impl FaceSelector {
    /// The angle of hatching on a wall or a riser, in its coordinates: vertical.
    pub const WALL_ANGLE: f64 = core::f64::consts::FRAC_PI_2;

    /// Its name in a manifest.
    pub fn name(&self) -> &'static str {
        match self {
            FaceSelector::Graph { .. } => "graph",
            FaceSelector::Plateaus { .. } => "plateaus",
            FaceSelector::Walls { .. } => "walls",
            FaceSelector::Risers { .. } => "risers",
            FaceSelector::Surface => "surface",
            FaceSelector::Solid => "solid",
        }
    }

    /// The kind of face selected.
    pub fn kind(&self) -> FaceKind {
        match self {
            FaceSelector::Graph { .. } => FaceKind::Graph,
            FaceSelector::Plateaus { .. } => FaceKind::Plateau,
            FaceSelector::Walls { .. } => FaceKind::Wall,
            FaceSelector::Risers { .. } => FaceKind::Riser,
            FaceSelector::Surface => FaceKind::Surface,
            FaceSelector::Solid => FaceKind::Solid,
        }
    }

    /// Why `hatching` cannot be laid on these faces, or `None` when it can.
    /// On a flat face lines at a spacing are evenly spaced at any angle. A
    /// curved face is hatched along a coordinate of its parameter rectangle
    /// (D8), angle 0 or π/2, with a uniform or half-Lambert tone, the tone
    /// setting the widths, in at most `PolyhedronManifest::MAX_WIDTH_STEPS`. A
    /// tone other than uniform, or more than one width, is drawn on a solid
    /// and a curved face only: a landscape's flat faces are one width each.
    pub fn unsupported_hatching(&self, hatching: &Hatching) -> Option<String> {
        let landscape: &str = match self {
            FaceSelector::Solid => {
                return (!hatching.angle.is_finite()).then(|| {
                    format!(
                        "hatching on a solid's faces at an angle that is not a number: {}",
                        text::double(hatching.angle)
                    )
                });
            }
            FaceSelector::Graph { .. } | FaceSelector::Surface => {
                if DomainAxis::from_angle(hatching.angle).is_none() {
                    return Some(format!(
                        "hatching on a curved face runs along a coordinate, angle 0 or π/2, not {}",
                        text::double(hatching.angle)
                    ));
                }
                if let Tone::Turing { .. } = hatching.tone {
                    return Some("Turing's tone on a curved face: it is laid on a solid's faces".to_string());
                }
                let max = crate::manifest::PolyhedronManifest::MAX_WIDTH_STEPS;
                if hatching.width_steps > max {
                    return Some(format!(
                        "hatching in {} widths, more than the {max} a curved face takes",
                        hatching.width_steps
                    ));
                }
                return None;
            }
            FaceSelector::Walls { .. } | FaceSelector::Risers { .. } => {
                if (hatching.angle - FaceSelector::WALL_ANGLE).abs() > Hatching::ANGLE_TOLERANCE
                    || hatching.angle.is_nan()
                {
                    return Some(format!(
                        "hatching on {} is vertical, angle π/2, not {}",
                        self.name(),
                        text::double(hatching.angle)
                    ));
                }
                self.name()
            }
            FaceSelector::Plateaus { .. } => {
                if DomainAxis::from_angle(hatching.angle).is_none() {
                    return Some(format!(
                        "hatching on plateaus runs along an axis, angle 0 or π/2, not {}",
                        text::double(hatching.angle)
                    ));
                }
                "plateaus"
            }
        };
        if hatching.tone != Tone::Uniform || hatching.width_steps != 1 {
            return Some(format!("hatching on {landscape} has a uniform tone and one width"));
        }
        None
    }

    /// Why a contour of `field` cannot lie on these faces, or `None` when it
    /// can. |f| and arg f are the graph's, and a surface's that f is drawn
    /// on; the coordinates are every face's, and the height every face's but
    /// a plateau's, where it is constant and has no level set. An isophote
    /// lies on a curved face: on a flat one n · l is constant, and has no
    /// level set either. On walls and risers the hatching's placement is
    /// left at its defaults.
    pub fn unsupported_field(&self, field: ContourField) -> Option<String> {
        use ContourField as F;
        let curved = matches!(self, FaceSelector::Graph { .. } | FaceSelector::Surface);
        match field {
            F::Magnitude | F::Phase | F::Isophote if curved => {}
            F::U | F::V => {}
            F::Isophote => {
                return Some(format!(
                    "isophotes on the {}: n · l is constant on a flat face, and has no level set",
                    self.name()
                ));
            }
            F::Height if matches!(self, FaceSelector::Plateaus { .. }) => {
                return Some(
                    "a contour of the height on the plateaus: it is constant on each, and has no level set".to_string(),
                );
            }
            F::Height => {}
            F::Magnitude | F::Phase => {
                return Some(format!("a contour of {} lies on the graph, not on the {}", field.raw(), self.name()));
            }
        }
        let placed = match self {
            FaceSelector::Walls { end_lines, base, top_offset, .. } => *end_lines || *base != 0.0 || *top_offset != 0.0,
            FaceSelector::Risers { end_lines, .. } => *end_lines,
            _ => false,
        };
        placed.then(|| {
            "a contour on walls or risers is placed by its field: endLines, base and topOffset place hatch lines"
                .to_string()
        })
    }

    pub fn from_json(json: &JsonValue) -> Result<FaceSelector, SchemaError> {
        let o = json.object("FaceSelector")?;
        let kind = o.string("kind", "FaceSelector")?;
        Ok(match kind.as_str() {
            "graph" => FaceSelector::Graph { tiled: o.bool("tiled", "FaceSelector.graph", Some(false))? },
            "plateaus" => FaceSelector::Plateaus { tiled: o.bool("tiled", "FaceSelector.plateaus", Some(false))? },
            "walls" => {
                if o.contains_key("trim") {
                    return Err(ManifestError::Unsupported(
                        "trim on walls: renamed endLines, with the opposite sense (schema 3)".to_string(),
                    )
                    .into());
                }
                FaceSelector::Walls {
                    edges: o.ints("edges", "FaceSelector.walls")?,
                    end_lines: o.bool("endLines", "FaceSelector.walls", Some(false))?,
                    base: o.optional_double("base").unwrap_or(0.0),
                    top_offset: o.optional_double("topOffset").unwrap_or(0.0),
                }
            }
            "risers" => {
                if o.contains_key("topOffset") {
                    return Err(ManifestError::Unsupported(
                        "topOffset on risers: a riser's lines run from its bottom edge to its top".to_string(),
                    )
                    .into());
                }
                FaceSelector::Risers {
                    tiled: o.bool("tiled", "FaceSelector.risers", Some(false))?,
                    end_lines: o.bool("endLines", "FaceSelector.risers", Some(false))?,
                }
            }
            "surface" => FaceSelector::Surface,
            "solid" => FaceSelector::Solid,
            _ => {
                return Err(ManifestError::unknown_kind(
                    &kind,
                    "FaceSelector",
                    &["graph", "plateaus", "walls", "risers", "surface", "solid"],
                )
                .into());
            }
        })
    }

    pub fn json(&self) -> JsonValue {
        match self {
            FaceSelector::Graph { tiled } => object([("kind", string("graph")), ("tiled", JsonValue::Bool(*tiled))]),
            FaceSelector::Plateaus { tiled } => {
                object([("kind", string("plateaus")), ("tiled", JsonValue::Bool(*tiled))])
            }
            FaceSelector::Walls { edges, end_lines, base, top_offset } => object([
                ("kind", string("walls")),
                ("edges", crate::json::ints(edges)),
                ("endLines", JsonValue::Bool(*end_lines)),
                ("base", JsonValue::Double(*base)),
                ("topOffset", JsonValue::Double(*top_offset)),
            ]),
            FaceSelector::Risers { tiled, end_lines } => object([
                ("kind", string("risers")),
                ("tiled", JsonValue::Bool(*tiled)),
                ("endLines", JsonValue::Bool(*end_lines)),
            ]),
            FaceSelector::Surface => object([("kind", string("surface"))]),
            FaceSelector::Solid => object([("kind", string("solid"))]),
        }
    }
}

/// The tone hatching reads, as a function on a face's plane with values in
/// `[-1, 1]`. Lines are drawn where it is positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tone {
    /// One everywhere: the whole face, hatched.
    Uniform,
    /// Turing's dappled pattern (`Turing`), in face coordinates divided by
    /// `scale`, with signs drawn from `seed`.
    Turing { scale: f64, seed: u64 },
    /// Darkness under half-Lambert shading: `(1 - n·l) / 2` for the face's
    /// outward normal `n` and the unit direction `l` toward the light, the
    /// complement of half-Lambert (wrapped) shading `(1 + n·l) / 2`. Nothing
    /// on a face square to the light, full width on one turned away from it:
    /// one value per flat face, and on a curved face read at each point from
    /// its normal there (`Face.coordinateHatching(_:on:)`).
    HalfLambert { light: Vec3 },
}

impl Tone {
    /// Whether `light` has a direction to normalize: not zero, not infinite.
    pub fn has_direction(light: Vec3) -> bool {
        let l2 = light.length_squared();
        l2 > 0.0 && l2.is_finite()
    }

    /// The tone on one face, in the coordinates of its frame.
    pub fn value(&self, face: &FaceFrame) -> Box<dyn Fn(Vec2) -> f64 + Send + Sync> {
        match *self {
            Tone::Uniform => Box::new(|_| 1.0),
            Tone::Turing { scale, seed } => Box::new(move |c| Turing::value(c / scale, seed)),
            Tone::HalfLambert { light } => {
                let w = Tone::darkness(face.normal, light);
                Box::new(move |_| w)
            }
        }
    }

    /// Half-Lambert darkness `(1 - n·l) / 2` at a point whose outward unit
    /// normal is `normal`, `l` the light's direction normalized.
    pub fn darkness(normal: Vec3, light: Vec3) -> f64 {
        (1.0 - normal.dot(light.normalized())) / 2.0
    }
}

/// Turing's dappled pattern, in closed form: unit Gaussians centred on a
/// square lattice, each with a random sign,
///
/// ```text
///     f(x, y) = Σ U(r, s) exp(-((x - hr)² + (y - hs)²) / 2),  U(r, s) = ±1,
/// ```
///
/// with h = 0.7. This is the closed form behind the dappled figure of
/// Turing's "The chemical basis of morphogenesis" (Phil. Trans. R. Soc. B
/// 237, 1952), as reconstructed in "Turing Patterns" (willishoke.github.io,
/// 2022-06-01). It is not a reaction-diffusion simulation. The signs are a
/// hash of the lattice point and the seed, so the pattern is a function on
/// the whole plane with no edge and no array behind it.
pub enum Turing {}

impl Turing {
    /// The lattice spacing h.
    pub const SPACING: f64 = 0.7;
    /// How far a kernel reaches before it is dropped: six, where it is
    /// under 2e-8 of its peak.
    const REACH: f64 = 6.0;
    /// The steepest the pattern can be, per unit of its own coordinates:
    /// every kernel's slope at its steepest sign, `Σ r exp(-r² / 2)` over the
    /// lattice, is `(2π / h²) √(π / 2)` by the integral and at most 16.12 on
    /// the lattice itself, and the pattern is the sum over `2 deviation`, so
    /// `|∇ value| ≤ 3.184`. Rounded up.
    pub const SLOPE: f64 = 3.2;

    /// The sum's standard deviation: the kernels' squares summed over the
    /// lattice, `π / h²` by the integral, rooted.
    pub fn deviation() -> f64 {
        real::sqrt(core::f64::consts::PI) / Turing::SPACING
    }

    /// The sign at lattice point `(r, s)` under `seed`: a mix of the three,
    /// its lowest bit.
    pub fn sign(r: i64, s: i64, seed: u64) -> f64 {
        let z = crate::splitmix::mix(
            seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (s as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
        );
        if z & 1 == 0 { -1.0 } else { 1.0 }
    }

    /// The raw sum at `p`.
    pub fn sum(p: Vec2, seed: u64) -> f64 {
        let r0 = ((p.x - Turing::REACH) / Turing::SPACING).ceil() as i64;
        let r1 = ((p.x + Turing::REACH) / Turing::SPACING).floor() as i64;
        let s0 = ((p.y - Turing::REACH) / Turing::SPACING).ceil() as i64;
        let s1 = ((p.y + Turing::REACH) / Turing::SPACING).floor() as i64;
        let mut total = 0.0;
        if r0 > r1 || s0 > s1 {
            return 0.0;
        }
        for r in r0..=r1 {
            let dx = p.x - r as f64 * Turing::SPACING;
            for s in s0..=s1 {
                let dy = p.y - s as f64 * Turing::SPACING;
                total += Turing::sign(r, s, seed) * real::exp(-0.5 * (dx * dx + dy * dy));
            }
        }
        total
    }

    /// The pattern: the sum over twice its deviation, clamped to `[-1, 1]`.
    pub fn value(p: Vec2, seed: u64) -> f64 {
        (Turing::sum(p, seed) / (2.0 * Turing::deviation())).clamp(-1.0, 1.0)
    }
}

// MARK: - hatching a solid

/// One hatch line on a solid: the face it lies on, its width step, and its
/// path in world space (Swift's `Solid.HatchLine`).
#[derive(Debug, Clone, PartialEq)]
pub struct HatchLine {
    pub face: usize,
    pub width_step: i64,
    pub path: Vec<P3<WorldSpace>>,
}

/// The hatching of one width step before visibility and after
/// (`Solid.visibleHatching(_:view:)`'s answer).
pub type VisibleHatching = Vec<(i64, PolylineSet<WorldSpace>, PolylineSet<WorldSpace>)>;

/// Swift's `sort { $0.0 < $1.0 }` on intervals: stable, ties in order.
fn by_start(a: &(f64, f64), b: &(f64, f64)) -> std::cmp::Ordering {
    if a.0 < b.0 {
        std::cmp::Ordering::Less
    } else if b.0 < a.0 {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

impl Solid {
    /// The hatching laid over every face, before visibility.
    pub fn hatch_lines(&self, hatching: &Hatching) -> Vec<HatchLine> {
        (0..self.faces.len()).flat_map(|i| self.hatch_lines_on_face(hatching, i)).collect()
    }

    /// Every hatch line's path, before visibility: the curves a hatching
    /// layer carries before the camera is known.
    pub fn hatch_paths(&self, hatching: &Hatching) -> PolylineSet<WorldSpace> {
        let paths: Vec<Vec<P3<WorldSpace>>> = self.hatch_lines(hatching).into_iter().map(|l| l.path).collect();
        PolylineSet::from_paths(&paths)
    }

    /// The hatching laid over face `i`.
    ///
    /// The face's plane is covered by lines at the hatching's angle, one
    /// every spacing, through the frame's origin and its multiples, so a
    /// face's hatching is fixed by its own frame and not by which of its
    /// pieces is asked. Each line is clipped to every piece of the face -- a
    /// convex polygon, so an interval -- and the intervals are joined where
    /// they meet, which is across every diagonal. The two pieces compute the
    /// crossing from the same two vertices but not always to the same bit;
    /// intervals within `1e-12` of the face's extent of each other are joined.
    /// A line lying on a side of the face is not drawn: that is an edge, and
    /// the edge is drawn. One lying on a diagonal is. The tone is then read
    /// along each run every `step`, and where the width step changes between
    /// two readings the change is bisected on the tone.
    pub fn hatch_lines_on_face(&self, hatching: &Hatching, i: usize) -> Vec<HatchLine> {
        let face = &self.frames[i];
        let polygons: Vec<Vec<Vec2>> = self.faces[i]
            .iter()
            .map(|&p| self.pieces[p].iter().map(|&k| face.coordinates(self.vertices[k].v)).collect())
            .collect();
        // Per side of each piece, whether it is a diagonal of the face
        // rather than an edge of the solid.
        let diagonals: HashSet<(usize, usize)> = (0..self.edges.len())
            .filter(|&k| self.is_diagonal[k])
            .map(|k| (self.edges[k].a, self.edges[k].b))
            .collect();
        let on_diagonal: Vec<Vec<bool>> = self.faces[i]
            .iter()
            .map(|&p| {
                let piece = &self.pieces[p];
                (0..piece.len())
                    .map(|j| {
                        let (a, b) = (piece[j], piece[(j + 1) % piece.len()]);
                        diagonals.contains(&(a.min(b), a.max(b)))
                    })
                    .collect()
            })
            .collect();
        let dir = Vec2::new(real::cos(hatching.angle), real::sin(hatching.angle));
        let per = Vec2::new(-dir.y, dir.x);
        let tone = hatching.tone.value(face);
        let cross = |a: Vec2, b: Vec2| a.x * b.y - a.y * b.x;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for poly in &polygons {
            for &p in poly {
                let c = p.dot(per);
                lo = swift::min(lo, c);
                hi = swift::max(hi, c);
            }
        }
        if !(lo.is_finite() && hi.is_finite() && hi > lo) {
            return Vec::new();
        }
        let extent = hi - lo;
        let mut out = Vec::new();
        // Where Swift's `Int` traps, on a count past Int, no line is laid.
        let (Some(k_lo), Some(k_hi)) =
            (swift::int((lo / hatching.spacing).ceil()), swift::int((hi / hatching.spacing).floor()))
        else {
            return Vec::new();
        };
        if k_lo > k_hi {
            return Vec::new();
        }
        for k in k_lo..=k_hi {
            let c = k as f64 * hatching.spacing;
            let base = c * per;
            // The line's overlap with each piece: half-plane by half-plane.
            let mut intervals: Vec<(f64, f64)> = Vec::new();
            for (poly, diagonal) in polygons.iter().zip(&on_diagonal) {
                let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
                let mut empty = false;
                for j in 0..poly.len() {
                    let (p, q) = (poly[j], poly[(j + 1) % poly.len()]);
                    let e = q - p;
                    let (f0, f1) = (cross(e, base - p), cross(e, dir));
                    if f1.abs() <= 1e-15 * extent {
                        // Parallel to the side. A line on it is outside the
                        // piece when the side is an edge, so a line along
                        // the face's boundary is left to the edge, and
                        // inside when it is a diagonal, so a line along a
                        // diagonal is drawn, once, across the join.
                        let on = f0.abs() <= 1e-12 * extent * e.length();
                        if if on { !diagonal[j] } else { f0 < 0.0 } {
                            empty = true;
                            break;
                        }
                    } else if f1 > 0.0 {
                        t0 = swift::max(t0, -f0 / f1);
                    } else {
                        t1 = swift::min(t1, -f0 / f1);
                    }
                }
                if !empty && t0 < t1 && t0.is_finite() && t1.is_finite() {
                    intervals.push((t0, t1));
                }
            }
            intervals.sort_by(by_start);
            let mut runs: Vec<(f64, f64)> = Vec::new();
            for (a, b) in intervals {
                match runs.last_mut() {
                    Some(last) if a <= last.1 + 1e-12 * extent => last.1 = swift::max(last.1, b),
                    _ => runs.push((a, b)),
                }
            }
            for (a, b) in runs {
                if !(b - a > 1e-12 * extent) {
                    continue;
                }
                let width_step = |t: f64| hatching.width_step(tone(base + t * dir));
                for (ta, tb, w) in Solid::graded(a, b, hatching.step, &width_step) {
                    if w > 0 {
                        out.push(HatchLine {
                            face: i,
                            width_step: w,
                            path: vec![
                                P3::from_vec(face.point(base + ta * dir)),
                                P3::from_vec(face.point(base + tb * dir)),
                            ],
                        });
                    }
                }
            }
        }
        out
    }

    /// `[a, b]` cut into intervals of one width step each: the step read
    /// every `step`, and every change between two readings bisected on the
    /// width step itself, to the last bit of the parameter. A change that
    /// reverses between two readings is not seen; that is what the reading
    /// step means. (Where Swift's `Int` traps on a count of readings past
    /// Int, one reading.)
    pub fn graded(a: f64, b: f64, step: f64, width_step: &dyn Fn(f64) -> i64) -> Vec<(f64, f64, i64)> {
        let length = b - a;
        let readings: i64 =
            if step > 0.0 && step.is_finite() { swift::int((length / step).ceil()).map_or(1, |n| n.max(1)) } else { 1 };
        let mut intervals: Vec<(f64, f64, i64)> = Vec::new();
        let mut start = a;
        let mut current = width_step(a);
        // `known` is the last point whose width step is `current`.
        let mut known = a;
        for j in 1..=readings {
            let t = if j == readings { b } else { a + length * j as f64 / readings as f64 };
            // Between two readings of different steps: the first change,
            // then whatever follows it, until the reading's step is reached.
            // Each change is bisected until `lo` and `hi` are neighbouring
            // numbers, where the midpoint no longer moves.
            while width_step(t) != current {
                let (mut lo, mut hi) = (known, t);
                loop {
                    let mid = 0.5 * (lo + hi);
                    if !(mid > lo && mid < hi) {
                        break;
                    }
                    if width_step(mid) == current {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                intervals.push((start, hi, current));
                start = hi;
                known = hi;
                current = width_step(hi);
            }
            known = t;
        }
        intervals.push((start, b, current));
        // Neighbours differ in width step by construction; only the last
        // interval can be empty, when a change is bisected to `b` itself.
        intervals.retain(|x| x.1 > x.0);
        intervals
    }

    /// The hatch lines by width step, the steps that occur in increasing
    /// order, each step's lines in the order they were laid.
    pub fn hatch_lines_by_width(&self, hatching: &Hatching) -> Vec<(i64, Vec<HatchLine>)> {
        let mut by_width: BTreeMap<i64, Vec<HatchLine>> = BTreeMap::new();
        for line in self.hatch_lines(hatching) {
            by_width.entry(line.width_step).or_default().push(line);
        }
        by_width.into_iter().collect()
    }

    /// The hatching for a camera: per width step, the hatch lines before
    /// and after visibility, so a layer per width step can be drawn at its
    /// width.
    pub fn visible_hatching(&self, hatching: &Hatching, view: &Transform<WorldSpace, ViewSpace>) -> VisibleHatching {
        let occlusion = self.occlusion(view);
        self.hatch_lines_by_width(hatching)
            .into_iter()
            .map(|(w, lines)| {
                let mut by_face: BTreeMap<usize, Vec<Vec<P3<WorldSpace>>>> = BTreeMap::new();
                for line in &lines {
                    by_face.entry(line.face).or_default().push(line.path.clone());
                }
                let visible: Vec<Vec<P3<WorldSpace>>> =
                    by_face.iter().flat_map(|(&i, paths)| self.visible_curves(paths, i, &occlusion)).collect();
                let all: Vec<Vec<P3<WorldSpace>>> = lines.into_iter().map(|l| l.path).collect();
                (w, PolylineSet::from_paths(&all), PolylineSet::from_paths(&visible))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_steps() {
        let h = Hatching::new(0.0, 0.1, None, Tone::Uniform, 4);
        assert_eq!(
            [h.width_step(-0.1), h.width_step(0.0), h.width_step(0.1), h.width_step(0.26), h.width_step(1.0)],
            [0, 0, 1, 2, 4]
        );
        assert_eq!(h.step, 0.1, "the step is the spacing by default");
    }

    #[test]
    fn fields_round_trip() {
        for tone in [
            Tone::Uniform,
            Tone::Turing { scale: 0.18, seed: u64::MAX },
            Tone::HalfLambert { light: Vec3::new(1.0, -2.0, 0.5) },
        ] {
            let h = Hatching::new(0.5, 0.035, Some(0.01), tone, 3);
            assert_eq!(Hatching::from_fields(&h.fields()).unwrap(), h);
        }
    }

    #[test]
    fn turing_is_a_function_on_the_plane() {
        let a = Turing::value(Vec2::new(0.3, -1.7), 3);
        assert!((-1.0..=1.0).contains(&a));
        assert_eq!(a, Turing::value(Vec2::new(0.3, -1.7), 3));
        assert!(Turing::sign(1, 2, 3).abs() == 1.0);
    }
}
