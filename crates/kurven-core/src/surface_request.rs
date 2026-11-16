//! A surface drawing, described: the port of
//! `Sources/KurvenCore/SurfaceRequest.swift`.
//!
//! The value the CLI parses its flags into, the app keeps in its document,
//! and a parametric surface's bundle stores (`SurfaceManifest`), so each
//! builds a drawing with one function -- `SurfaceDrawing.build` -- as a bake
//! and a landscape are each built by one function called twice. The map is
//! code, so a bundle keeps the request that builds it rather than samples of
//! it, as a landscape's provenance keeps its request.
//!
//! A `periodic` expression is kept as its text: KurvenCore depends on no
//! expression parser, and what does not parse is refused when the drawing
//! is built (`SurfaceDrawing.build`, Phase 4), not here.

use crate::json::{JsonObject, JsonValue, Members, object, string};
use crate::manifest::{CameraPreset, ManifestError, ObliqueProjection, SchemaError};
use crate::text;

#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    /// The torus of revolution about the z axis.
    Torus { major: f64, minor: f64 },
    /// sn(z, m) on its fundamental rectangle, glued into a torus. Its own
    /// case rather than a `periodic` expression, because the modulus
    /// decides the periods as well as the function. `conformal`: drawn in
    /// isothermal coordinates on the one round torus whose flat rectangle
    /// has the aspect of sn's, so the modulus pins the minor radius
    /// (`conformal_minor`) and every |f| contour crosses every arg f
    /// contour at a right angle in space.
    Sn { modulus: f64, major: f64, minor: f64, grid: i64, conformal: bool },
    /// Any function with a rectangle of periods, glued into a torus.
    Periodic { expression: String, periods: (f64, f64), major: f64, minor: f64, grid: i64 },
    /// A forced system's invariant torus, fitted from `fit` time units of
    /// one run at `harmonics`, drawn by revolution on a ring of `radius`
    /// with state components `radial` and `axial`; its trajectory drawn
    /// on it for `turns` turns of the forcing -- as the linear curve at the
    /// rotation number, which is what the trajectory is on the fitted
    /// torus. The turns where it cuts the torus most evenly are the
    /// denominators of the rotation number's convergents.
    Forced { system: String, fit: f64, harmonics: i64, radial: i64, axial: i64, radius: f64, turns: i64 },
    /// sn's landscape in relief on its torus: the torus displaced along
    /// its normal by `amplitude` times |sn| truncated at its ceiling,
    /// over `repeats` periods round the ring and round the tube, the
    /// rectangle begun `turn` imaginary periods up the plane. Its phase
    /// lines end on the truncation edge. `floor`, when given, displaces
    /// only |sn| above it; `soft` saturates the truncation smoothly,
    /// c · tanh(|sn| / c), instead of creasing it.
    Relief {
        modulus: f64,
        major: f64,
        minor: f64,
        amplitude: f64,
        repeats: (i64, i64),
        grid: i64,
        turn: f64,
        floor: Option<f64>,
        soft: bool,
    },
}

impl Shape {
    /// The shape's name: its `kind` in a manifest.
    pub fn kind(&self) -> &'static str {
        match self {
            Shape::Torus { .. } => "torus",
            Shape::Sn { .. } => "sn",
            Shape::Periodic { .. } => "periodic",
            Shape::Forced { .. } => "forced",
            Shape::Relief { .. } => "relief",
        }
    }

    /// The minor radius on which sn(z, m) is drawn conformally round a ring
    /// of radius `major`. The round torus is isothermal in
    /// w(θ) = ∫₀^θ r dθ′ / (R + r cos θ′), whose whole turn is 2πr / √(R² − r²),
    /// so its flat rectangle has the aspect r / √(R² − r²), tube over ring.
    /// sn's rectangle is 4K round the ring and 2K′ round the tube
    /// (`PeriodicTorus.jacobiSN`), of aspect a = K′ / (2K), and the two agree
    /// where r / R = a / √(1 + a²). Not a number, or not under `major`,
    /// where the modulus is so near 0 that K′ is out of reach.
    pub fn conformal_minor(m: f64, major: f64) -> f64 {
        let a = Shape::quarter_period(1.0 - m) / (2.0 * Shape::quarter_period(m));
        major * a / (1.0 + a * a).sqrt()
    }

    /// K(m), the quarter period, as `Jacobi.quarterPeriod` computes it, to
    /// the bit (which the test suite holds): the pin is checked here, and
    /// KurvenCore depends on nothing. Only the four operations and square
    /// roots, so the libm does not enter.
    fn quarter_period(m: f64) -> f64 {
        if !(0.0..1.0).contains(&m) {
            return if m == 1.0 { f64::INFINITY } else { f64::NAN };
        }
        let (mut a, mut b) = (1.0f64, (1.0 - m).sqrt());
        for _ in 0..64 {
            if a == b {
                break;
            }
            (a, b) = (0.5 * (a + b), (a * b).sqrt());
            if (a - b).abs() <= ulp(a) {
                break;
            }
        }
        core::f64::consts::PI / (2.0 * a)
    }
}

/// Swift's `Double.ulp`: the distance from a finite value to the next one
/// away from zero.
fn ulp(x: f64) -> f64 {
    let x = x.abs();
    if !x.is_finite() {
        return f64::NAN;
    }
    f64::from_bits(x.to_bits() + 1) - x
}

/// Stroke widths. A fold width of zero leaves the folds out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style {
    pub lines: f64,
    pub trajectory: f64,
    pub folds: f64,
    pub linear_curves: f64,
}

impl Default for Style {
    fn default() -> Style {
        Style { lines: 0.3, trajectory: 0.08, folds: 0.6, linear_curves: 0.3 }
    }
}

/// A linear curve drawn on the torus: `count` straight lines through its
/// parameter rectangle at `slope` turns of `v` per turn of `u`, each
/// `turns` turns long -- or closed, when the slope is a fraction whose
/// denominator is within the turns. See `ParametricSurface.linearCurves`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCurves {
    pub slope: f64,
    pub turns: i64,
    pub count: i64,
}

impl LinearCurves {
    /// The turns after which a strand closes, when it does within
    /// `turns`: the denominator of a rational slope.
    pub fn closes(&self) -> Option<i64> {
        linear_curve_closes(self.slope, self.turns)
    }
}

/// The turns after which a linear curve of `slope` closes, when it does
/// within `turns`: `ParametricSurface.linearCurveCloses(slope:within:)`,
/// here because the request's bounds read it, and KurvenCore's request
/// cannot reach the surface.
pub fn linear_curve_closes(slope: f64, turns: i64) -> Option<i64> {
    if turns < 1 {
        return None;
    }
    (1..=turns).find(|&k| {
        let w = slope * k as f64;
        (w - w.round()).abs() <= 1e-9 * 1f64.max(w.abs())
    })
}

/// One number of a request, to edit by name: what a control moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Field {
    Major,
    Minor,
    Modulus,
    Grid,
    Radius,
    Harmonics,
    Fit,
    Radial,
    Axial,
    /// A forced trajectory's length, in turns of the forcing.
    TrajectoryTurns,
    /// The linear curve's, present when one is drawn: its slope, its length
    /// in turns, and how many strands.
    Slope,
    Turns,
    Strands,
    /// A relief's amplitude: how far along the normal the torus is
    /// displaced per unit of the truncated |f|. `--relief` on the CLI.
    Relief,
    /// Whether sn is drawn conformally, 1 or 0: a switch, which
    /// `setting` refuses at any other value. On, the
    /// modulus and the major radius pin the minor radius.
    Conformal,
}

impl Field {
    pub const ALL: [Field; 15] = [
        Field::Major,
        Field::Minor,
        Field::Modulus,
        Field::Grid,
        Field::Radius,
        Field::Harmonics,
        Field::Fit,
        Field::Radial,
        Field::Axial,
        Field::TrajectoryTurns,
        Field::Slope,
        Field::Turns,
        Field::Strands,
        Field::Relief,
        Field::Conformal,
    ];

    pub fn raw(self) -> &'static str {
        match self {
            Field::Major => "major",
            Field::Minor => "minor",
            Field::Modulus => "modulus",
            Field::Grid => "grid",
            Field::Radius => "radius",
            Field::Harmonics => "harmonics",
            Field::Fit => "fit",
            Field::Radial => "radial",
            Field::Axial => "axial",
            Field::TrajectoryTurns => "trajectoryTurns",
            Field::Slope => "slope",
            Field::Turns => "turns",
            Field::Strands => "strands",
            Field::Relief => "relief",
            Field::Conformal => "conformal",
        }
    }

    pub fn parse(raw: &str) -> Option<Field> {
        Field::ALL.into_iter().find(|f| f.raw() == raw)
    }
}

/// A surface drawing, described: which surface, at what grid, with what curves.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceRequest {
    pub shape: Shape,
    /// Grid points around `u`; `v` gets half as many (a periodic drawing
    /// sizes `v` to its rectangle instead).
    pub grid: i64,
    /// Lines of constant `u` and `v`, when drawn. A plain torus has nothing
    /// else to draw; a forced torus draws them only when asked.
    pub lines: Option<(i64, i64)>,
    /// A linear curve, when drawn. Curves alone: the surface does not know it is
    /// there, so its slope can move under a slider.
    pub linear_curves: Option<LinearCurves>,
    pub style: Style,
    /// The catalog entry this came from, for provenance and the title.
    pub name: String,
}

/// The default `name` spelling of a refusal: a field by its key in a
/// manifest's `surface`.
pub fn surface_key(key: &str) -> String {
    format!("surface.{key}")
}

impl SurfaceRequest {
    /// A request at Swift's defaults: a grid of 1024, no coordinate or
    /// linear curves, the default style, no name.
    pub fn new(shape: Shape) -> SurfaceRequest {
        SurfaceRequest {
            shape,
            grid: 1024,
            lines: None,
            linear_curves: None,
            style: Style::default(),
            name: String::new(),
        }
    }

    /// The projection a surface drawing opens at.
    pub const PROJECTION: ObliqueProjection =
        ObliqueProjection { shear: 0.0, x_angle: -55.0, z_angle: 30.0, flip_x: false, y_scale: None };

    /// The same, as a camera preset: no depth tolerance and the CLI's bake resolution,
    /// so the window and `kurven-cli surface` bake the same picture.
    pub fn preset() -> CameraPreset {
        CameraPreset {
            name: "oblique".to_string(),
            projection: SurfaceRequest::PROJECTION,
            depth_tolerance: 0.0,
            buffer: 3000,
        }
    }

    /// The request a drag builds while it lasts: a quarter of the grid
    /// around, and a periodic function sampled at most 240 across. The full
    /// request follows when the drag ends. A forced trajectory needs no
    /// draft: it is a linear curve, a vertex per cell of whatever grid.
    pub fn drafted(&self) -> SurfaceRequest {
        let mut r = self.clone();
        r.grid = self.grid.min(256);
        match &mut r.shape {
            Shape::Torus { .. } | Shape::Forced { .. } => {}
            Shape::Sn { grid, .. } | Shape::Periodic { grid, .. } | Shape::Relief { grid, .. } => {
                *grid = (*grid).min(240);
            }
        }
        r
    }

    /// Whether the surface carries a function, whose |f| and arg f a contour
    /// can be taken of: sn and any periodic expression, in relief or not.
    pub fn paints_function(&self) -> bool {
        matches!(self.shape, Shape::Sn { .. } | Shape::Periodic { .. } | Shape::Relief { .. })
    }

    /// Whether the function is truncated, so a contour of it can be kept
    /// below the truncation: sn in relief, whose phase lines end on the
    /// truncation edge.
    pub fn truncates_function(&self) -> bool {
        matches!(self.shape, Shape::Relief { .. })
    }

    /// Whether the surface is sn drawn conformally.
    pub fn conformal(&self) -> bool {
        matches!(self.shape, Shape::Sn { conformal: true, .. })
    }

    /// The fields this request's shape has, in the order a panel lists them.
    /// The linear curve's are listed by `LINEAR_CURVE_FIELDS` when there is one;
    /// sn's switch, `conformal`, is a field but no slider, and is not listed;
    /// nor, while it is on, is the minor radius it pins.
    pub fn fields(&self) -> Vec<Field> {
        use Field::*;
        match self.shape {
            Shape::Torus { .. } => vec![Major, Minor],
            Shape::Sn { conformal, .. } => {
                if conformal {
                    vec![Modulus, Major, Grid]
                } else {
                    vec![Modulus, Major, Minor, Grid]
                }
            }
            Shape::Relief { .. } => vec![Modulus, Major, Minor, Grid, Relief],
            Shape::Periodic { .. } => vec![Major, Minor, Grid],
            Shape::Forced { .. } => vec![Radius, Radial, Axial, TrajectoryTurns, Harmonics, Fit],
        }
    }

    pub const LINEAR_CURVE_FIELDS: [Field; 3] = [Field::Slope, Field::Turns, Field::Strands];

    /// The fields that move curves alone -- placed from the grid, nothing of
    /// the surface remade -- and so want no draft while dragged.
    pub const CURVE_FIELDS: [Field; 4] = [Field::Slope, Field::Turns, Field::Strands, Field::TrajectoryTurns];

    /// A field's value, or `None` where the shape has no such field.
    pub fn get(&self, field: Field) -> Option<f64> {
        use Field as F;
        match field {
            F::Slope => return self.linear_curves.map(|l| l.slope),
            F::Turns => return self.linear_curves.map(|l| l.turns as f64),
            F::Strands => return self.linear_curves.map(|l| l.count as f64),
            _ => {}
        }
        match (&self.shape, field) {
            (Shape::Torus { major, .. }, F::Major)
            | (Shape::Sn { major, .. }, F::Major)
            | (Shape::Periodic { major, .. }, F::Major)
            | (Shape::Relief { major, .. }, F::Major) => Some(*major),
            (Shape::Torus { minor, .. }, F::Minor)
            | (Shape::Sn { minor, .. }, F::Minor)
            | (Shape::Periodic { minor, .. }, F::Minor)
            | (Shape::Relief { minor, .. }, F::Minor) => Some(*minor),
            (Shape::Sn { modulus, .. }, F::Modulus) | (Shape::Relief { modulus, .. }, F::Modulus) => Some(*modulus),
            (Shape::Sn { conformal, .. }, F::Conformal) => Some(if *conformal { 1.0 } else { 0.0 }),
            (Shape::Relief { amplitude, .. }, F::Relief) => Some(*amplitude),
            (Shape::Sn { grid, .. }, F::Grid)
            | (Shape::Periodic { grid, .. }, F::Grid)
            | (Shape::Relief { grid, .. }, F::Grid) => Some(*grid as f64),
            (Shape::Forced { radius, .. }, F::Radius) => Some(*radius),
            (Shape::Forced { turns, .. }, F::TrajectoryTurns) => Some(*turns as f64),
            (Shape::Forced { harmonics, .. }, F::Harmonics) => Some(*harmonics as f64),
            (Shape::Forced { fit, .. }, F::Fit) => Some(*fit),
            (Shape::Forced { radial, .. }, F::Radial) => Some(*radial as f64),
            (Shape::Forced { axial, .. }, F::Axial) => Some(*axial as f64),
            _ => None,
        }
    }

    /// Set a field, as a control sets it; setting one the shape does not
    /// have, or to a value that is not a number, changes nothing.
    pub fn set(&mut self, field: Field, value: f64) {
        use Field as F;
        if !value.is_finite() {
            return;
        }
        // −0 is 0, so a bundle saved after it is byte for byte the one
        // saved after 0; a count past what an Int holds is a count
        // `check` refuses, not a trap.
        let x = if value == 0.0 { 0.0 } else { value };
        let n = x.round().clamp(-1e18, 1e18) as i64;
        match field {
            F::Slope => {
                if let Some(l) = &mut self.linear_curves {
                    l.slope = x;
                }
                return;
            }
            F::Turns => {
                if let Some(l) = &mut self.linear_curves {
                    l.turns = n.max(1);
                }
                return;
            }
            F::Strands => {
                if let Some(l) = &mut self.linear_curves {
                    l.count = n.max(1);
                }
                return;
            }
            _ => {}
        }
        match &mut self.shape {
            Shape::Torus { major, minor } => match field {
                F::Major => *major = x,
                F::Minor => *minor = x,
                _ => {}
            },
            Shape::Sn { modulus, major, minor, grid, conformal } => {
                match field {
                    F::Modulus => *modulus = x,
                    F::Major => *major = x,
                    F::Minor => *minor = x,
                    F::Grid => *grid = n,
                    F::Conformal => *conformal = x != 0.0,
                    _ => return,
                }
                // Conformal, the modulus and the major radius pin the minor
                // radius, which moves with them; one set outright is what
                // `check` refuses.
                if *conformal && field != F::Minor {
                    *minor = Shape::conformal_minor(*modulus, *major);
                }
            }
            Shape::Periodic { major, minor, grid, .. } => match field {
                F::Major => *major = x,
                F::Minor => *minor = x,
                F::Grid => *grid = n,
                _ => {}
            },
            Shape::Forced { fit, harmonics, radial, axial, radius, turns, .. } => match field {
                F::Radius => *radius = x,
                F::TrajectoryTurns => *turns = n.max(1),
                F::Harmonics => *harmonics = n,
                F::Fit => *fit = x,
                F::Radial => *radial = n,
                F::Axial => *axial = n,
                _ => {}
            },
            Shape::Relief { modulus, major, minor, amplitude, grid, .. } => match field {
                F::Modulus => *modulus = x,
                F::Major => *major = x,
                F::Minor => *minor = x,
                F::Grid => *grid = n,
                F::Relief => *amplitude = x,
                _ => {}
            },
        }
    }

    /// This request with `field` set to `value`, as a control sets it, or
    /// the refusal `check` makes of the result, naming the field as `name`
    /// spells its key (`surface_key` by default). A value that is not a
    /// number is refused rather than ignored, as `set` ignores it.
    pub fn setting(
        &self,
        field: Field,
        value: f64,
        name: &dyn Fn(&str) -> String,
    ) -> Result<SurfaceRequest, ManifestError> {
        if !value.is_finite() {
            return Err(if field == Field::Relief {
                ManifestError::Invalid(format!(
                    "{} must be from 0 to 1, not {}",
                    name("shape.amplitude"),
                    text::double(value)
                ))
            } else {
                ManifestError::Invalid(format!("{} must be a finite number, not {}", field.raw(), text::double(value)))
            });
        }
        if field == Field::Conformal && value != 0.0 && value != 1.0 {
            return Err(ManifestError::Invalid(format!(
                "{} must be 1 (on) or 0 (off), not {}",
                name("shape.conformal"),
                text::double(value)
            )));
        }
        let mut next = self.clone();
        next.set(field, value);
        next.check(name)?;
        Ok(next)
    }

    // MARK: - the schema

    /// Read from a manifest's `surface` (`SurfaceManifest`), and checked
    /// (`check`): a request that could not build a drawing is refused here,
    /// with the number that is wrong, rather than when it is built.
    pub fn from_json(json: &JsonValue) -> Result<SurfaceRequest, SchemaError> {
        let o = json.object("SurfaceRequest")?;
        let what = "SurfaceRequest.shape";
        let s = o.value("shape", "SurfaceRequest")?.object(what)?;
        let kind = s.string("kind", what)?;
        let shape = match kind.as_str() {
            "torus" => Shape::Torus { major: s.double("major", what)?, minor: s.double("minor", what)? },
            // A bundle saved before sn could be drawn conformally has no
            // `conformal`, and is drawn as it was.
            "sn" => Shape::Sn {
                modulus: s.double("modulus", what)?,
                major: s.double("major", what)?,
                minor: s.double("minor", what)?,
                grid: s.int("grid", what)?,
                conformal: s.bool("conformal", what, Some(false))?,
            },
            "periodic" => {
                let p = s.doubles("periods", what)?;
                if p.len() != 2 {
                    return Err(ManifestError::bad_shape("SurfaceRequest periods", &[p.len() as i64]).into());
                }
                Shape::Periodic {
                    expression: s.string("expression", what)?,
                    periods: (p[0], p[1]),
                    major: s.double("major", what)?,
                    minor: s.double("minor", what)?,
                    grid: s.int("grid", what)?,
                }
            }
            "forced" => Shape::Forced {
                system: s.string("system", what)?,
                fit: s.double("fit", what)?,
                harmonics: s.int("harmonics", what)?,
                radial: s.int("radial", what)?,
                axial: s.int("axial", what)?,
                radius: s.double("radius", what)?,
                turns: s.int("turns", what)?,
            },
            "relief" => {
                let k = s.ints("repeats", what)?;
                if k.len() != 2 {
                    return Err(ManifestError::bad_shape("SurfaceRequest repeats", &[k.len() as i64]).into());
                }
                Shape::Relief {
                    modulus: s.double("modulus", what)?,
                    major: s.double("major", what)?,
                    minor: s.double("minor", what)?,
                    amplitude: s.double("amplitude", what)?,
                    repeats: (k[0], k[1]),
                    grid: s.int("grid", what)?,
                    turn: s.double("turn", what)?,
                    floor: if s.get("floor") == Some(&JsonValue::Null) { None } else { Some(s.double("floor", what)?) },
                    soft: s.bool("soft", what, None)?,
                }
            }
            other => {
                return Err(ManifestError::unknown_kind(
                    other,
                    "SurfaceRequest.shape",
                    &["torus", "sn", "periodic", "forced", "relief"],
                )
                .into());
            }
        };
        if s.contains_key("conformal") && shape.kind() != "sn" {
            return Err(ManifestError::Unsupported(format!(
                "conformal on a {} shape: only plain sn is drawn in isothermal coordinates",
                shape.kind()
            ))
            .into());
        }
        let w = "SurfaceRequest.linearCurves";
        let st = "SurfaceRequest.style";
        let lines = if o.get("lines") == Some(&JsonValue::Null) {
            None
        } else {
            let l = o.value("lines", "SurfaceRequest")?.object("SurfaceRequest.lines")?;
            Some((l.int("u", "SurfaceRequest.lines")?, l.int("v", "SurfaceRequest.lines")?))
        };
        let linear_curves = if o.get("linearCurves") == Some(&JsonValue::Null) {
            None
        } else {
            let l = o.value("linearCurves", "SurfaceRequest")?.object(w)?;
            Some(LinearCurves { slope: l.double("slope", w)?, turns: l.int("turns", w)?, count: l.int("count", w)? })
        };
        let style: &JsonObject = o.value("style", "SurfaceRequest")?.object(st)?;
        let request = SurfaceRequest {
            shape,
            grid: o.int("grid", "SurfaceRequest")?,
            lines,
            linear_curves,
            style: Style {
                lines: style.double("lines", st)?,
                trajectory: style.double("trajectory", st)?,
                folds: style.double("folds", st)?,
                linear_curves: style.double("linearCurves", st)?,
            },
            name: o.string("name", "SurfaceRequest")?,
        };
        request.check(&surface_key)?;
        Ok(request)
    }

    pub fn json(&self) -> JsonValue {
        // −0 is written as 0, so a request is one bundle however it was made.
        fn double(x: f64) -> JsonValue {
            JsonValue::Double(if x == 0.0 { 0.0 } else { x })
        }
        let mut shape: JsonObject = JsonObject::new();
        let mut put = |k: &str, v: JsonValue| {
            shape.insert(k.to_string(), v);
        };
        match &self.shape {
            Shape::Torus { major, minor } => {
                put("major", double(*major));
                put("minor", double(*minor));
            }
            Shape::Sn { modulus, major, minor, grid, conformal } => {
                put("modulus", double(*modulus));
                put("major", double(*major));
                put("minor", double(*minor));
                put("grid", JsonValue::Int(*grid));
                // Written only when on, so a bundle that is not is the bundle
                // it was before sn could be.
                if *conformal {
                    put("conformal", JsonValue::Bool(true));
                }
            }
            Shape::Periodic { expression, periods, major, minor, grid } => {
                put("expression", string(expression));
                put("periods", JsonValue::Array(vec![double(periods.0), double(periods.1)]));
                put("major", double(*major));
                put("minor", double(*minor));
                put("grid", JsonValue::Int(*grid));
            }
            Shape::Forced { system, fit, harmonics, radial, axial, radius, turns } => {
                put("system", string(system));
                put("fit", double(*fit));
                put("harmonics", JsonValue::Int(*harmonics));
                put("radial", JsonValue::Int(*radial));
                put("axial", JsonValue::Int(*axial));
                put("radius", double(*radius));
                put("turns", JsonValue::Int(*turns));
            }
            Shape::Relief { modulus, major, minor, amplitude, repeats, grid, turn, floor, soft } => {
                put("modulus", double(*modulus));
                put("major", double(*major));
                put("minor", double(*minor));
                put("amplitude", double(*amplitude));
                put("repeats", crate::json::ints(&[repeats.0, repeats.1]));
                put("grid", JsonValue::Int(*grid));
                put("turn", double(*turn));
                put("floor", floor.map_or(JsonValue::Null, double));
                put("soft", JsonValue::Bool(*soft));
            }
        }
        put("kind", string(self.shape.kind()));
        object([
            ("shape", JsonValue::Object(shape)),
            ("grid", JsonValue::Int(self.grid)),
            (
                "lines",
                self.lines
                    .map_or(JsonValue::Null, |(u, v)| object([("u", JsonValue::Int(u)), ("v", JsonValue::Int(v))])),
            ),
            (
                "linearCurves",
                self.linear_curves.map_or(JsonValue::Null, |l| {
                    object([
                        ("slope", double(l.slope)),
                        ("turns", JsonValue::Int(l.turns)),
                        ("count", JsonValue::Int(l.count)),
                    ])
                }),
            ),
            (
                "style",
                object([
                    ("lines", double(self.style.lines)),
                    ("trajectory", double(self.style.trajectory)),
                    ("folds", double(self.style.folds)),
                    ("linearCurves", double(self.style.linear_curves)),
                ]),
            ),
            ("name", string(&self.name)),
        ])
    }

    /// Refuse a request no drawing can be built from, or one whose drawing
    /// would cost without bound, naming the number: a radius that is not
    /// positive, a modulus outside (0, 1), a conformal sn's minor radius
    /// other than the one its modulus pins, a relief outside [0, 1], a grid
    /// too coarse to span its rectangle or too fine to hold (`MAX_SAMPLES`),
    /// curves of more vertices than a layer holds (`MAX_VERTICES`), a forced
    /// torus revolved on one state component twice. What only building can find -- a system the
    /// catalog does not have, an expression that does not parse --
    /// `SurfaceDrawing.build` refuses. `name` spells a field, given its key
    /// in the request (`shape.modulus`), as the refusal names it: by its key
    /// in a manifest's `surface`, or as the CLI's flag.
    pub fn check(&self, name: &dyn Fn(&str) -> String) -> Result<(), ManifestError> {
        let refuse = |key: &str, why: String| ManifestError::Invalid(format!("{} {why}", name(key)));
        let positive = |x: f64, key: &str| -> Result<(), ManifestError> {
            if x.is_finite() && x > 0.0 {
                Ok(())
            } else {
                Err(refuse(key, format!("must be a positive number, not {}", text::double(x))))
            }
        };
        let finite = |x: f64, key: &str| -> Result<(), ManifestError> {
            if x.is_finite() {
                Ok(())
            } else {
                Err(refuse(key, format!("must be a finite number, not {}", text::double(x))))
            }
        };
        let within = |n: i64, lo: i64, hi: i64, key: &str, what: &str| -> Result<(), ManifestError> {
            if (lo..=hi).contains(&n) {
                Ok(())
            } else {
                Err(refuse(key, format!("must be {lo} to {hi} {what}, not {n}")))
            }
        };
        let modulus = |m: f64| -> Result<(), ManifestError> {
            if m.is_finite() && m > 0.0 && m < 1.0 {
                Ok(())
            } else {
                Err(refuse("shape.modulus", format!("must lie strictly between 0 and 1, not {}", text::double(m))))
            }
        };
        let radii = |r_major: f64, r_minor: f64| -> Result<(), ManifestError> {
            positive(r_major, "shape.major")?;
            positive(r_minor, "shape.minor")
        };
        within(self.grid, 8, SurfaceRequest::MAX_GRID, "grid", "samples")?;
        match &self.shape {
            Shape::Torus { major, minor } => radii(*major, *minor)?,
            Shape::Sn { modulus: m, major, minor, grid, conformal } => {
                modulus(*m)?;
                if *conformal {
                    positive(*major, "shape.major")?;
                    let pinned = Shape::conformal_minor(*m, *major);
                    if !(pinned.is_finite() && pinned > 0.0 && pinned < *major) {
                        return Err(refuse(
                            "shape.modulus",
                            format!(
                                "is too near 0 for sn to be drawn conformally, {}: K′ is out of reach, and the tube it \
                                 pins would close on the axis",
                                text::double(*m)
                            ),
                        ));
                    }
                    if *minor != pinned {
                        return Err(refuse(
                            "shape.minor",
                            format!(
                                "is pinned by the modulus on a conformal sn, to {} round a ring of {}, not {}",
                                text::double(pinned),
                                text::double(*major),
                                text::double(*minor)
                            ),
                        ));
                    }
                }
                radii(*major, *minor)?;
                within(*grid, 8, SurfaceRequest::MAX_FUNCTION_GRID, "shape.grid", "samples")?;
            }
            Shape::Periodic { expression, periods, major, minor, grid } => {
                // Foundation's `.whitespaces`: tab and the space separators, not
                // the line breaks.
                if expression
                    .trim_matches(|c: char| {
                        c == '\t' || (c.is_whitespace() && !c.is_control() && c != '\u{2028}' && c != '\u{2029}')
                    })
                    .is_empty()
                {
                    return Err(refuse("shape.expression", "is empty".to_string()));
                }
                positive(periods.0, "shape.periods")?;
                positive(periods.1, "shape.periods")?;
                radii(*major, *minor)?;
                within(*grid, 8, SurfaceRequest::MAX_FUNCTION_GRID, "shape.grid", "samples")?;
            }
            Shape::Forced { system, fit, harmonics, radial, axial, radius, turns } => {
                if system.is_empty() {
                    return Err(refuse("shape.system", "is unnamed".to_string()));
                }
                positive(*fit, "shape.fit")?;
                positive(*radius, "shape.radius")?;
                if *fit > SurfaceRequest::MAX_FIT {
                    return Err(refuse(
                        "shape.fit",
                        format!(
                            "must be at most {} time units, not {}",
                            SurfaceRequest::MAX_FIT as i64,
                            text::double(*fit)
                        ),
                    ));
                }
                within(*harmonics, 1, SurfaceRequest::MAX_HARMONICS, "shape.harmonics", "harmonics")?;
                within(*radial, 0, 2, "shape.radial", "(x, x′ or x″)")?;
                within(*axial, 0, 2, "shape.axial", "(x, x′ or x″)")?;
                if radial == axial {
                    return Err(refuse(
                        "shape.radial",
                        format!(
                            "and {} are both state component {radial}: one component as both is a curve, not a torus",
                            name("shape.axial")
                        ),
                    ));
                }
                within(*turns, 1, SurfaceRequest::MAX_VERTICES, "shape.turns", "turns")?;
                let vertices = *turns as f64 * self.sample_bound().0 as f64;
                if vertices > SurfaceRequest::MAX_VERTICES as f64 {
                    return Err(refuse(
                        "shape.turns",
                        format!(
                            "would take {} vertices for the trajectory, more than the {} a layer holds",
                            text::g3(vertices),
                            SurfaceRequest::MAX_VERTICES
                        ),
                    ));
                }
            }
            Shape::Relief { modulus: m, major, minor, amplitude, repeats, grid, turn, floor, .. } => {
                modulus(*m)?;
                radii(*major, *minor)?;
                if !(0.0..=1.0).contains(amplitude) {
                    return Err(refuse(
                        "shape.amplitude",
                        format!("must be from 0 to 1, not {}", text::double(*amplitude)),
                    ));
                }
                finite(*turn, "shape.turn")?;
                if let Some(floor) = floor
                    && !(0.0 <= *floor && *floor < SurfaceRequest::RELIEF_CEILING)
                {
                    return Err(refuse(
                        "shape.floor",
                        format!(
                            "must be at least 0 and below the truncation height {}, not {}",
                            text::double(SurfaceRequest::RELIEF_CEILING),
                            text::double(*floor)
                        ),
                    ));
                }
                within(repeats.0, 1, SurfaceRequest::MAX_REPEATS, "shape.repeats", "whole periods each way")?;
                within(repeats.1, 1, SurfaceRequest::MAX_REPEATS, "shape.repeats", "whole periods each way")?;
                within(*grid, 8, SurfaceRequest::MAX_FUNCTION_GRID, "shape.grid", "samples")?;
            }
        }
        let (u, v) = self.sample_bound();
        if u as f64 * v as f64 > SurfaceRequest::MAX_SAMPLES as f64 {
            return Err(refuse(
                "grid",
                format!(
                    "and the radii make a grid of {u} × {v} samples, more than the {} a surface holds",
                    SurfaceRequest::MAX_SAMPLES
                ),
            ));
        }
        if let Some((lu, lv)) = self.lines {
            within(lu, 0, SurfaceRequest::MAX_VERTICES, "lines", "coordinate curves")?;
            within(lv, 0, SurfaceRequest::MAX_VERTICES, "lines", "coordinate curves")?;
            if let Some(why) = SurfaceRequest::vertices_of_coordinate_curves((lu, lv), self.line_resolution()) {
                return Err(refuse("lines", why));
            }
        }
        if let Some(w) = &self.linear_curves
            && let Some(why) = self.vertices_of_linear_curves(w)
        {
            return Err(refuse("linearCurves", why));
        }
        for (width, key) in [
            (self.style.lines, "style.lines"),
            (self.style.trajectory, "style.trajectory"),
            (self.style.folds, "style.folds"),
            (self.style.linear_curves, "style.linearCurves"),
        ] {
            if !(width.is_finite() && width >= 0.0) {
                return Err(refuse(key, format!("must be a number not below 0, not {}", text::double(width))));
            }
        }
        Ok(())
    }

    /// The most samples a surface's grid holds: 8192 round the ring by 4096
    /// round the tube.
    pub const MAX_SAMPLES: i64 = 1 << 25;
    /// The most vertices one layer's curves hold.
    pub const MAX_VERTICES: i64 = 1 << 24;
    /// The most samples round the ring, and across a function's rectangle.
    pub const MAX_GRID: i64 = 1 << 16;
    pub const MAX_FUNCTION_GRID: i64 = 4096;
    /// The longest run a forced torus is fitted from, and its most harmonics.
    pub const MAX_FIT: f64 = 100_000.0;
    pub const MAX_HARMONICS: i64 = 64;
    /// The most periods a relief repeats each way.
    pub const MAX_REPEATS: i64 = 16;
    /// The height |sn| is truncated at in relief. A floor is at least 0 and
    /// below it, so the height displaced, |sn| less the floor, lies in
    /// [0, ceiling − floor] with ceiling − floor positive. A relief's
    /// amplitudes lie in [0, 1]: not below 0, since displaced inward past
    /// its reach the torus would pass through itself while the relief still
    /// claimed to bound a solid (`PeriodicTorus.Drawing.relief`); at most 1,
    /// a displacement of at most the truncation height. A relief past its
    /// reach is drawn with its back faces kept, so the bound is on cost, not
    /// on embedding.
    pub const RELIEF_CEILING: f64 = 3.0;

    /// Whether the surface bounds a solid, and so has an outward side, which
    /// an isophote and a half-Lambert tone are read from: a torus whose tube
    /// does not reach its axis, R > r, and in relief only while the
    /// displacement, |amplitude| times the truncated height less the floor,
    /// stays under the tube's reach, min(r, R − r), as
    /// `PeriodicTorus.Drawing.relief` decides it. `None` for a forced torus,
    /// which bounds one where its fitted revolution passes through itself
    /// nowhere: that is decided when it is built.
    pub fn bounds_solid(&self) -> Option<bool> {
        match &self.shape {
            Shape::Torus { major, minor } | Shape::Sn { major, minor, .. } | Shape::Periodic { major, minor, .. } => {
                Some(major > minor)
            }
            Shape::Relief { major, minor, amplitude, floor, .. } => {
                let ceiling = floor.map_or(SurfaceRequest::RELIEF_CEILING, |f| SurfaceRequest::RELIEF_CEILING - f);
                Some(major > minor && ceiling >= 0.0 && amplitude.abs() * ceiling < minor.min(major - minor))
            }
            Shape::Forced { .. } => None,
        }
    }

    /// At most how many samples the surface's grid has round the ring and
    /// round the tube: exactly, but for sn, whose rectangle's aspect its
    /// modulus sets, where this takes the aspect at most 1.
    pub fn sample_bound(&self) -> (i64, i64) {
        fn clamped(x: f64) -> i64 {
            x.max(0.0).min(f64::from(i32::MAX)) as i64
        }
        // A periodic torus sizes its tube to keep the cells square
        // (`PeriodicTorus.torus(over:major:minor:grid:)`).
        fn tube(grid: i64, aspect: f64, major: f64, minor: f64) -> i64 {
            clamped(grid as f64 * aspect.min(1.0) * minor / major).max(32)
        }
        let grid = self.grid;
        match &self.shape {
            Shape::Torus { .. } => (grid, (grid / 2).max(8)),
            Shape::Forced { .. } => (grid, grid / 2),
            Shape::Sn { major, minor, .. } => (grid, tube(grid, 1.0, *major, *minor)),
            Shape::Periodic { periods, major, minor, .. } => (grid, tube(grid, periods.1 / periods.0, *major, *minor)),
            Shape::Relief { major, minor, repeats, .. } => {
                let base = grid.min(1024);
                (
                    clamped(base as f64 * repeats.0 as f64).min(grid),
                    clamped(tube(base, 1.0, *major, *minor) as f64 * repeats.1 as f64).max(32),
                )
            }
        }
    }

    /// How many vertices a coordinate curve has along its axis: four to a
    /// grid cell, two on a fitted torus, where each is the Fourier series.
    pub fn line_resolution(&self) -> i64 {
        if let Shape::Forced { .. } = self.shape { 2 * self.grid } else { 4 * self.grid }
    }

    /// Why coordinate curves at `counts` would hold more vertices than a
    /// layer does, or `None`.
    pub fn vertices_of_coordinate_curves(counts: (i64, i64), resolution: i64) -> Option<String> {
        let vertices = (counts.0 as f64 + counts.1 as f64) * (resolution as f64 + 1.0);
        (vertices > SurfaceRequest::MAX_VERTICES as f64).then(|| {
            format!(
                "would take {} vertices for {} + {} coordinate curves, more than the {} a layer holds",
                text::g3(vertices),
                counts.0,
                counts.1,
                SurfaceRequest::MAX_VERTICES
            )
        })
    }

    /// Why a linear curve cannot be drawn on this surface, or `None`: a slope
    /// that is not a number, no turn or no strand, or more vertices than a
    /// layer holds, at a vertex per grid cell along the steeper axis
    /// (`ParametricSurface.linearCurves`).
    pub fn vertices_of_linear_curves(&self, w: &LinearCurves) -> Option<String> {
        if !w.slope.is_finite() {
            return Some(format!("wants a finite slope, not {}", text::double(w.slope)));
        }
        let max = SurfaceRequest::MAX_VERTICES;
        if !((1..=max).contains(&w.turns) && (1..=max).contains(&w.count)) {
            return Some(format!(
                "wants at least one turn of one strand, and at most {max}, not {} turns of {}",
                w.turns, w.count
            ));
        }
        let (u, v) = self.sample_bound();
        let steps = (u as f64).max((w.slope.abs() * v as f64).ceil());
        let vertices = w.count as f64 * (w.closes().unwrap_or(w.turns) as f64 * steps + 1.0);
        (vertices > max as f64)
            .then(|| format!("would take {} vertices, more than the {max} a layer holds", text::g3(vertices)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conformal_sn_pins_its_minor_radius() {
        let mut r =
            SurfaceRequest::new(Shape::Sn { modulus: 0.5, major: 2.0, minor: 1.0, grid: 120, conformal: false });
        r.grid = 64;
        let on = r.setting(Field::Conformal, 1.0, &surface_key).unwrap();
        let pinned = on.get(Field::Minor).unwrap();
        assert_eq!(pinned, Shape::conformal_minor(0.5, 2.0));
        assert!(pinned > 0.0 && pinned < 2.0);
        let refused = on.setting(Field::Minor, 0.9, &surface_key).unwrap_err().to_string();
        assert!(refused.starts_with("manifest: surface.shape.minor is pinned by the modulus"), "{refused}");
        assert!(on.setting(Field::Conformal, 0.5, &surface_key).is_err());
    }

    #[test]
    fn fields_get_and_set() {
        let mut r = SurfaceRequest::new(Shape::Torus { major: 2.0, minor: 1.0 });
        assert_eq!(r.get(Field::Modulus), None);
        r.set(Field::Major, -0.0);
        assert!(!kurven_math::real::sign_minus(r.get(Field::Major).unwrap()), "−0 is 0");
        r.set(Field::Modulus, 0.3);
        assert_eq!(r.shape, Shape::Torus { major: 0.0, minor: 1.0 }, "a field the shape has not changes nothing");
        assert_eq!(linear_curve_closes(0.4, 24), Some(5));
        assert_eq!(linear_curve_closes(core::f64::consts::SQRT_2, 24), None);
    }

    #[test]
    fn drafts_and_bounds() {
        let r = SurfaceRequest::new(Shape::Periodic {
            expression: "sn(z, 0.5)".to_string(),
            periods: (2.0, 1.0),
            major: 2.0,
            minor: 1.0,
            grid: 1000,
        });
        let d = r.drafted();
        assert_eq!(d.grid, 256);
        assert!(matches!(d.shape, Shape::Periodic { grid: 240, .. }));
        assert_eq!(r.sample_bound(), (1024, 256));
        assert_eq!(r.bounds_solid(), Some(true));
        assert!(r.check(&surface_key).is_ok());
    }
}
