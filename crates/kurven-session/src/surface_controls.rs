//! The numbers a surface is made of, as controls: the model half of
//! `Sources/KurvenApp/SurfaceSection.swift`. Swift's view holds how each
//! field is shown and how far it reaches, where its slider rests, and the
//! edits its switches and pickers make; here they are the engine's, so the
//! page builds its controls from what the state says (`json`) and every
//! edit is the document's, one undo step named for the control.

use kurven_core::json::{self, JsonValue};
use kurven_core::surface_request::{Field, LinearCurves, Shape, SurfaceRequest, surface_key};
use kurven_math::continued_fraction::{ContinuedFraction, Convergent};
use kurven_surfaces::drawing::SurfacePreset;

use crate::document::Document;
use crate::sliders::Slider;

/// How a field is shown and how far it reaches (`SurfaceSection.Control`).
#[derive(Debug, Clone, PartialEq)]
pub struct Control {
    pub label: &'static str,
    pub range: (f64, f64),
    pub step: f64,
    /// A slider, or only a field and a stepper: a refit is half a second,
    /// and a slider that stalls under the hand is worse than none.
    pub slides: bool,
    pub digits: i64,
    pub help: &'static str,
}

const fn c(label: &'static str, range: (f64, f64), step: f64, help: &'static str) -> Control {
    Control { label, range, step, slides: true, digits: 3, help }
}

/// `SurfaceSection.control(_:)`.
pub fn control(field: Field) -> Control {
    match field {
        Field::Major => c("Major radius", (0.5, 5.0), 0.05, "The radius of the ring the tube runs round."),
        Field::Minor => c(
            "Minor radius",
            (0.1, 3.0),
            0.05,
            "The tube's radius. At or above the major radius the torus reaches its axis, bounds nothing, and hides \
             nothing.",
        ),
        Field::Modulus => c(
            "Modulus",
            (0.01, 0.99),
            0.01,
            "m in sn(z, m). It sets the periods, so the rectangle the torus is rolled from changes with it, and the \
             function is sampled again. Conformal, it sets the minor radius too.",
        ),
        Field::Grid => Control {
            digits: 0,
            ..c(
                "Samples",
                (120.0, 1200.0),
                20.0,
                "Samples across the period rectangle that the contours are traced on. While you drag, at most 240.",
            )
        },
        Field::Radius => c(
            "Ring radius",
            (0.5, 5.0),
            0.05,
            "The radius the torus is revolved at. Below about 1 a slice reaches the axis and the torus passes \
             through itself.",
        ),
        Field::TrajectoryTurns => Control {
            digits: 0,
            ..c(
                "Turns",
                (1.0, 600.0),
                1.0,
                "Turns of the forcing the trajectory is drawn for. The trajectory is the linear curve at Ω/ω, and \
                 the slider rests at the denominators of Ω/ω's convergents, where it cuts the torus most evenly; \
                 between them it bands.",
            )
        },
        Field::Harmonics => Control {
            slides: false,
            digits: 0,
            ..c(
                "Harmonics",
                (4.0, 40.0),
                1.0,
                "The order of the Fourier fit. Too few and the fit misses the trajectory by more than it allows; the \
                 status line says so.",
            )
        },
        Field::Fit => Control {
            slides: false,
            digits: 0,
            ..c("Fit length", (1000.0, 20000.0), 500.0, "Time units of the run the torus is fitted from.")
        },
        Field::Radial => Control { digits: 0, ..c("Radius from", (0.0, 2.0), 1.0, "") },
        Field::Axial => Control { digits: 0, ..c("Height from", (0.0, 2.0), 1.0, "") },
        Field::Slope => c(
            "Slope",
            (0.0, 2.0),
            0.001,
            "Turns through the tube per turn round the ring. The slider rests at fractions, where the linear curve \
             closes, and on the forced torus at the trajectory's own Ω/ω.",
        ),
        Field::Turns => Control {
            digits: 0,
            ..c("Turns", (1.0, 64.0), 1.0, "How many turns round the ring a strand runs, unless it closes first.")
        },
        Field::Strands => Control {
            digits: 0,
            ..c("Strands", (1.0, 12.0), 1.0, "Parallel linear curves, evenly spaced through the tube.")
        },
        Field::Conformal => Control { digits: 0, ..c("Conformal", (0.0, 1.0), 1.0, "") },
        // `SurfaceRequest.reliefRange`, which `check` holds an amplitude to.
        Field::Relief => c(
            "Relief",
            (0.0, 1.0),
            0.01,
            "How far the torus is displaced along its normal per unit of |sn|, truncated at 3. Past the torus's \
             reach the relief passes through itself, and its back faces are kept.",
        ),
    }
}

/// How the coordinate curves' counts reach (`count(_:label:_:)`'s stepper).
pub const LINE_COUNTS: (i64, i64, i64) = (1, 200, 2);

/// The trajectory's own linear curve, when the drawing on screen is a fitted
/// torus: its rotation number, Ω/ω.
pub fn own_slope(document: &Document) -> Option<f64> {
    document.surface_drawing().and_then(|d| d.torus()).map(|t| t.rotation_number())
}

/// The rotation number's convergents within the turns slider's range.
fn convergents(document: &Document) -> Vec<Convergent> {
    own_slope(document).map(|x| ContinuedFraction::convergents(x, 600, 1e-12)).unwrap_or_default()
}

/// Where a slider rests: the slope's, on a forced torus, the rotation number
/// and its convergents -- the closed linear curves the trajectory is nearly;
/// on any other torus every fraction with a denominator up to eight. The
/// trajectory's turns: the convergents' denominators.
pub fn detents(document: &Document, field: Field) -> Vec<f64> {
    match field {
        Field::Slope => {
            if let Some(own) = own_slope(document) {
                let mut out = vec![own];
                out.extend(convergents(document).iter().map(|c| c.value()));
                return out;
            }
            let mut out: Vec<f64> = Vec::new();
            for q in 1..=8i64 {
                for p in 0..=(2 * q) {
                    if q == 1 || p % q != 0 {
                        let x = p as f64 / q as f64;
                        if !out.iter().any(|&y| (y - x).abs() < 1e-12) {
                            out.push(x);
                        }
                    }
                }
            }
            out
        }
        Field::TrajectoryTurns => convergents(document).iter().map(|c| c.q as f64).filter(|&q| q >= 2.0).collect(),
        _ => Vec::new(),
    }
}

/// How close to a detent a slider value lands on it: a pixel or two of a
/// sidebar slider. The slope's range is two, the turns' six hundred.
pub fn detent_reach(field: Field) -> f64 {
    if field == Field::TrajectoryTurns { 4.0 } else { 0.015 }
}

/// A slider value to its step, or to the detent within reach of it
/// (`quantize`): the first nearest, as Swift's `min(by:)` keeps it.
pub fn quantize(x: f64, step: f64, detents: &[f64], reach: f64) -> f64 {
    let stepped = (x / step).round() * step;
    let mut nearest: Option<f64> = None;
    for &d in detents {
        if nearest.is_none_or(|n| (d - x).abs() < (n - x).abs()) {
            nearest = Some(d);
        }
    }
    match nearest {
        Some(d) if (d - x).abs() <= reach => d,
        _ => stepped,
    }
}

/// A slider's raw value as the slider sets it: clamped to the control's
/// range, then to its step or a detent within reach.
pub fn slid(document: &Document, field: Field, x: f64) -> f64 {
    let ctl = control(field);
    let x = kurven_math::swift::min(kurven_math::swift::max(x, ctl.range.0), ctl.range.1);
    quantize(x, ctl.step, &detents(document, field), detent_reach(field))
}

/// A value typed or stepped, as the field's binding sets it: clamped to the
/// control's range.
pub fn typed(field: Field, x: f64) -> f64 {
    let ctl = control(field);
    kurven_math::swift::min(kurven_math::swift::max(x, ctl.range.0), ctl.range.1)
}

/// The linear curve, in words (`describe`).
pub fn describe(document: &Document, w: &LinearCurves) -> String {
    let strands = if w.count == 1 { "One strand".to_string() } else { format!("{} strands", w.count) };
    if let Some(own) = own_slope(document)
        && (w.slope - own).abs() < 1e-12
    {
        return format!("{strands} at the trajectory's own Ω/ω, {own:.6}: open, {} turns.", w.turns);
    }
    if let Some(q) = w.closes() {
        let p = (w.slope * q as f64).round() as i64;
        let convergent = convergents(document).iter().any(|c| c.p == p && c.q == q);
        let closes = if q == 1 { "closes after one turn.".to_string() } else { format!("closes after {q} turns.") };
        return format!("{strands} at {p}/{q}{}: {closes}", if convergent { ", a convergent of Ω/ω" } else { "" });
    }
    format!("{strands}, open: {} turns.", w.turns)
}

impl Document {
    /// A field typed, stepped or switched (sn's Conformal): clamped to the
    /// control's range and set as `SurfaceFieldSlider.set` sets it, one undo
    /// step named for the control. A value the request or the layers refuse
    /// is not taken, and the status says why.
    pub fn type_surface_field(&mut self, field: Field, value: f64) {
        Slider::SurfaceField { field, label: control(field).label.to_string() }.set(self, typed(field, value));
    }

    /// The coordinate curves switched on, at 36 by 18, or off
    /// ("Coordinate Curves").
    pub fn switch_lines(&mut self, on: bool) {
        let Some(mut next) = self.surface.clone() else { return };
        next.lines = on.then_some((36, 18));
        self.edit_surface(next, "Coordinate Curves");
    }

    /// One of the coordinate curves' counts, around (`u`) or across (`v`),
    /// clamped to 1...200 ("Lines Around", "Lines Across").
    pub fn set_line_count(&mut self, around: bool, count: i64) {
        let Some(mut next) = self.surface.clone() else { return };
        let Some((u, v)) = next.lines else { return };
        let n = count.clamp(LINE_COUNTS.0, LINE_COUNTS.1);
        next.lines = Some(if around { (n, v) } else { (u, n) });
        self.edit_surface(next, if around { "Lines Around" } else { "Lines Across" });
    }

    /// The linear curve switched on or off ("Linear curve"). On, it starts at
    /// the trajectory's own linear curve when there is one to match, else at
    /// 2/5, which closes; for 24 turns, one strand (`LinearCurves(slope:)`'s
    /// defaults).
    pub fn switch_linear_curve(&mut self, on: bool) {
        let Some(mut next) = self.surface.clone() else { return };
        next.linear_curves = on.then(|| LinearCurves { slope: own_slope(self).unwrap_or(0.4), turns: 24, count: 1 });
        self.edit_surface(next, "Linear curve");
    }

    /// Which state component is drawn as the ring's radius (`radial`) or its
    /// height (`axial`). The two must differ -- one component as both is a
    /// curve, not a torus -- so choosing the other's swaps them.
    pub fn choose_component(&mut self, field: Field, chosen: i64) {
        let Some(request) = self.surface.clone() else { return };
        let other = match field {
            Field::Radial => Field::Axial,
            Field::Axial => Field::Radial,
            _ => return,
        };
        let mut next = request.clone();
        let as_int = |x: Option<f64>, default: i64| x.map_or(default, |v| v as i64);
        if as_int(request.get(other), -1) == chosen {
            next.set(other, request.get(field).unwrap_or(0.0));
        }
        next.set(field, chosen as f64);
        self.edit_surface(next, control(field).label);
    }
}

/// One field's row, as the page builds its control.
fn row(document: &Document, request: &SurfaceRequest, field: Field) -> JsonValue {
    let ctl = control(field);
    json::object([
        ("field", json::string(field.raw())),
        ("label", json::string(ctl.label)),
        ("value", request.get(field).map_or(JsonValue::Null, JsonValue::Double)),
        ("min", JsonValue::Double(ctl.range.0)),
        ("max", JsonValue::Double(ctl.range.1)),
        ("step", JsonValue::Double(ctl.step)),
        ("slides", JsonValue::Bool(ctl.slides)),
        ("digits", JsonValue::Int(ctl.digits)),
        ("help", json::string(ctl.help)),
        ("detents", json::doubles(&detents(document, field))),
    ])
}

/// The surface's controls as the page builds them, from the request the
/// controls say: its notes, a row per field (the components as a picker),
/// sn's switch, the coordinate curves and the linear curve.
pub fn json(document: &Document) -> JsonValue {
    let Some(request) = document.surface_request() else { return JsonValue::Null };
    let notes = SurfacePreset::named(&request.name).map(|p| p.notes);
    let rows: Vec<JsonValue> = request
        .fields()
        .into_iter()
        .filter(|f| !matches!(f, Field::Radial | Field::Axial))
        .filter(|&f| request.get(f).is_some())
        .map(|f| row(document, request, f))
        .collect();
    let components = match request.shape {
        Shape::Forced { radial, axial, .. } => {
            json::object([("radial", JsonValue::Int(radial)), ("axial", JsonValue::Int(axial))])
        }
        _ => JsonValue::Null,
    };
    let conformal = request.get(Field::Conformal).map_or(JsonValue::Null, |v| JsonValue::Bool(v != 0.0));
    let is_torus = matches!(request.shape, Shape::Torus { .. });
    let linear = request.linear_curves.map_or(JsonValue::Null, |w| {
        json::object([
            (
                "rows",
                JsonValue::Array(
                    SurfaceRequest::LINEAR_CURVE_FIELDS.iter().map(|&f| row(document, request, f)).collect(),
                ),
            ),
            ("description", json::string(&describe(document, &w))),
        ])
    });
    json::object([
        ("kind", json::string(request.shape.kind())),
        ("notes", notes.map_or(JsonValue::Null, |n| json::string(&n))),
        ("rows", JsonValue::Array(rows)),
        ("components", components),
        ("conformal", conformal),
        // A plain torus is its lines: it has no switch for them.
        ("linesSwitch", JsonValue::Bool(!is_torus)),
        ("lines", request.lines.map_or(JsonValue::Null, |(u, v)| json::ints(&[u, v]))),
        (
            "lineCounts",
            json::object([
                ("min", JsonValue::Int(LINE_COUNTS.0)),
                ("max", JsonValue::Int(LINE_COUNTS.1)),
                ("step", JsonValue::Int(LINE_COUNTS.2)),
            ]),
        ),
        ("linearCurve", linear),
        ("building", JsonValue::Bool(document.building)),
    ])
}

/// Whether `field` is one a typed value may set on this request: a field
/// the shape has, which the request takes at the value clamped.
pub fn settable(request: &SurfaceRequest, field: Field, value: f64) -> Result<(), String> {
    if request.get(field).is_none() {
        return Err(format!("the surface has no field '{}'", field.raw()));
    }
    request.setting(field, typed(field, value), &surface_key).map(|_| ()).map_err(|e| e.to_string())
}
