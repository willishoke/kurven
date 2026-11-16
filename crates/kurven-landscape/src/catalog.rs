//! The menu: presets worth looking at, and the language they are written in.
//! The port of `Sources/KurvenLandscape/Catalog.swift`.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use kurven_core::json::{JsonValue, Members};
use kurven_core::manifest::{Domain, Interval, LayerSpec, SchemaError, Truncation};
use kurven_math::Language;

use crate::landscape::NativeLandscape;

/// A function worth looking at, and the window it is worth looking at in.
///
/// `window` is how far the domain sliders reach, not what they start at: the
/// interesting rectangle is `domain`, and the window is the room around it
/// where the picture still means something.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionPreset {
    pub name: String,
    pub label: String,
    pub expression: String,
    /// The rectangle worth looking at.
    pub domain: Domain,
    /// How far the sliders reach around it.
    pub window: Domain,
    /// Where to truncate, when the function has a pole worth truncating.
    pub truncation_height: Option<f64>,
    pub notes: String,
}

impl FunctionPreset {
    pub fn id(&self) -> &str {
        &self.name
    }

    pub fn from_json(json: &JsonValue) -> Result<FunctionPreset, SchemaError> {
        let o = json.object("FunctionPreset")?;
        Ok(FunctionPreset {
            name: o.string("name", "FunctionPreset")?,
            label: o.string("label", "FunctionPreset")?,
            expression: o.string("expression", "FunctionPreset")?,
            domain: Domain::from_json(o.value("domain", "FunctionPreset")?)?,
            window: Domain::from_json(o.value("window", "FunctionPreset")?)?,
            truncation_height: o.optional_double("truncation"),
            notes: o.string("notes", "FunctionPreset").unwrap_or_default(),
        })
    }

    pub fn json(&self) -> JsonValue {
        let mut o = BTreeMap::new();
        o.insert("name".to_string(), JsonValue::String(self.name.clone()));
        o.insert("label".to_string(), JsonValue::String(self.label.clone()));
        o.insert("expression".to_string(), JsonValue::String(self.expression.clone()));
        o.insert("domain".to_string(), self.domain.json());
        o.insert("window".to_string(), self.window.json());
        o.insert("truncation".to_string(), self.truncation_height.map(JsonValue::Double).unwrap_or(JsonValue::Null));
        o.insert("notes".to_string(), JsonValue::String(self.notes.clone()));
        JsonValue::Object(o)
    }

    pub fn truncation(&self) -> Truncation {
        self.truncation_height.map(Truncation::Uniform).unwrap_or(Truncation::None)
    }
}

/// One function of the expression language, as a picker describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanguageFunction {
    pub name: String,
    pub arity: i64,
    pub aliases: Vec<String>,
    pub help: String,
}

impl LanguageFunction {
    pub fn id(&self) -> &str {
        &self.name
    }

    pub fn from_json(json: &JsonValue) -> Result<LanguageFunction, SchemaError> {
        let o = json.object("LanguageFunction")?;
        Ok(LanguageFunction {
            name: o.string("name", "LanguageFunction")?,
            arity: o.int("arity", "LanguageFunction").unwrap_or(1),
            aliases: o.strings("aliases", "LanguageFunction").unwrap_or_default(),
            help: o.string("help", "LanguageFunction").unwrap_or_default(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Catalog {
    pub presets: Vec<FunctionPreset>,
    pub functions: Vec<LanguageFunction>,
    pub constants: Vec<String>,
    pub default_resolution: i64,
    pub default_buffer: i64,
}

static NATIVE: LazyLock<Catalog> = LazyLock::new(|| Catalog {
    presets: NativeCatalog::presets(),
    functions: Language::functions()
        .iter()
        .map(|f| LanguageFunction {
            name: f.name.to_string(),
            arity: f.arity as i64,
            aliases: f.aliases.iter().map(|a| a.to_string()).collect(),
            help: f.help.to_string(),
        })
        .collect(),
    constants: Language::constants().iter().map(|c| c.name.to_string()).collect(),
    default_resolution: NativeLandscape::DEFAULT_RESOLUTION,
    default_buffer: NativeLandscape::DEFAULT_BUFFER,
});

impl Catalog {
    pub fn preset(&self, name: &str) -> Option<&FunctionPreset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// The menu: the presets, ported from the Python package's catalog and
    /// defined here since, and the language `kurven-math` implements, whose
    /// table `tests/fixtures/expr/language.json` freezes (`Catalog.native`).
    pub fn native() -> &'static Catalog {
        &NATIVE
    }
}

/// What to sample, over what, how finely -- and, once the user has edited it,
/// how it is drawn.
///
/// `truncation` and `layers` are absent until the client has changed them, and absent
/// means "whatever the rules derive". Keeping them across a change of window is
/// what makes a truncation and a hatch spacing survive a drag: the landscape is
/// resampled, the styling is not re-decided. This is `kurven.landscape.Spec`.
#[derive(Debug, Clone, PartialEq)]
pub struct LandscapeRequest {
    pub expression: String,
    pub domain: Domain,
    pub resolution: i64,
    pub truncation: Option<Truncation>,
    pub spacing: Option<f64>,
    pub layers: Option<Vec<LayerSpec>>,
    /// The catalog entry this came from, for provenance. Empty when the
    /// expression was typed.
    pub name: String,
}

impl LandscapeRequest {
    /// Swift's `init(expression:domain:resolution:)` with its defaults: no
    /// truncation, spacing or layers, and no name.
    pub fn new(expression: &str, domain: Domain, resolution: i64) -> LandscapeRequest {
        LandscapeRequest {
            expression: expression.to_string(),
            domain,
            resolution,
            truncation: None,
            spacing: None,
            layers: None,
            name: String::new(),
        }
    }

    pub fn from_preset(preset: &FunctionPreset, resolution: i64) -> LandscapeRequest {
        LandscapeRequest {
            truncation: Some(preset.truncation()),
            name: preset.name.clone(),
            ..LandscapeRequest::new(&preset.expression, preset.domain, resolution)
        }
    }

    /// `Spec.to_dict`, as a request or a fixture spells it.
    pub fn from_json(json: &JsonValue) -> Result<LandscapeRequest, SchemaError> {
        let o = json.object("Spec")?;
        let truncation = match o.get("truncation") {
            Some(t @ JsonValue::Object(_)) => Some(Truncation::from_json(t)?),
            _ => None,
        };
        let layers = match o.get("layers") {
            Some(JsonValue::Array(list)) => Some(list.iter().map(LayerSpec::from_json).collect::<Result<Vec<_>, _>>()?),
            _ => None,
        };
        Ok(LandscapeRequest {
            expression: o.string("expression", "Spec")?,
            domain: Domain::from_json(o.value("domain", "Spec")?)?,
            resolution: o.int("resolution", "Spec").unwrap_or(NativeLandscape::DEFAULT_RESOLUTION),
            truncation,
            spacing: o.optional_double("spacing"),
            layers,
            name: o.string("name", "Spec").unwrap_or_default(),
        })
    }

    /// True when the two ask for the same *samples*. Everything else is
    /// styling, which the consumer applies itself rather than asking for.
    pub fn samples_as(&self, other: &LandscapeRequest) -> bool {
        self.expression == other.expression && self.domain == other.domain && self.resolution == other.resolution
    }

    pub fn json(&self) -> JsonValue {
        let mut o = BTreeMap::new();
        o.insert("expression".to_string(), JsonValue::String(self.expression.clone()));
        o.insert("domain".to_string(), self.domain.json());
        o.insert("resolution".to_string(), JsonValue::Int(self.resolution));
        o.insert("name".to_string(), JsonValue::String(self.name.clone()));
        if let Some(t) = &self.truncation {
            o.insert("truncation".to_string(), t.json());
        }
        if let Some(s) = self.spacing {
            o.insert("spacing".to_string(), JsonValue::Double(s));
        }
        if let Some(layers) = &self.layers {
            o.insert("layers".to_string(), JsonValue::Array(layers.iter().map(LayerSpec::json).collect()));
        }
        JsonValue::Object(o)
    }
}

/// `kurven.landscape.CATALOG`, entry for entry.
pub enum NativeCatalog {}

impl NativeCatalog {
    pub fn domain(r0: f64, r1: f64, i0: f64, i1: f64) -> Domain {
        Domain::new(Interval::new(r0, r1), Interval::new(i0, i1))
    }

    pub fn presets() -> Vec<FunctionPreset> {
        let d = NativeCatalog::domain;
        let preset = |name: &str,
                      label: &str,
                      expression: &str,
                      domain: Domain,
                      window: Domain,
                      truncation_height: Option<f64>,
                      notes: &str| FunctionPreset {
            name: name.to_string(),
            label: label.to_string(),
            expression: expression.to_string(),
            domain,
            window,
            truncation_height,
            notes: notes.to_string(),
        };
        vec![
            preset(
                "rgamma",
                "1/Γ(z) — reciprocal factorial",
                "1/gamma(z)",
                d(-5.5, 4.0, 0.0, 2.5),
                d(-12.0, 8.0, -6.0, 6.0),
                Some(5.0),
                "Entire, with zeros at the non-positive integers: the ripple along \
                 the negative real axis. Jahnke-Emde's Fig. 6.",
            ),
            preset(
                "gamma",
                "Γ(z) — gamma",
                "gamma(z)",
                d(-4.5, 4.5, 0.0, 2.5),
                d(-10.0, 10.0, -6.0, 6.0),
                Some(5.0),
                "A pole spire at each non-positive integer. Truncating per band \
                 cuts each spire at its own height, as the published drawing does.",
            ),
            preset(
                "zeta",
                "ζ(z) — Riemann zeta",
                "zeta(z)",
                d(-6.0, 8.0, 0.0, 30.0),
                d(-20.0, 20.0, -40.0, 40.0),
                Some(6.0),
                "The pole at s = 1 and the zeros on the critical line. Sampled \
                 here rather than loaded from a cache.",
            ),
            preset(
                "cn",
                "cn(z, 0.64) — Jacobi elliptic",
                "cn(z, 0.64)",
                d(-4.0, 4.0, -3.5, 3.5),
                d(-12.0, 12.0, -12.0, 12.0),
                Some(4.0),
                "Doubly periodic: a lattice of poles, one per quarter-period cell.",
            ),
            preset(
                "tan",
                "tan z",
                "tan(z)",
                d(-4.8, 4.8, -1.6, 1.6),
                d(-12.0, 12.0, -4.0, 4.0),
                Some(5.0),
                "A pole at every half-period along the real axis, and flat plateaus \
                 of |tan| -> 1 away from it.",
            ),
            preset(
                "sin",
                "sin z",
                "sin(z)",
                d(-5.0, 5.0, -1.5, 1.5),
                d(-12.0, 12.0, -5.0, 5.0),
                Some(2.5),
                "Entire: zeros on the real axis, growing exponentially away from it.",
            ),
            preset(
                "exp_inv",
                "exp(1/z) — essential singularity",
                "exp(1/z)",
                d(-1.2, 1.2, -1.2, 1.2),
                d(-4.0, 4.0, -4.0, 4.0),
                Some(5.0),
                "Every value, infinitely often, in every neighbourhood of the \
                 origin. The landscape shows why.",
            ),
            preset(
                "cubic",
                "1/(z³ - 1)",
                "1/(z^3 - 1)",
                d(-2.0, 2.0, -2.0, 2.0),
                d(-6.0, 6.0, -6.0, 6.0),
                Some(5.0),
                "Three simple poles at the cube roots of unity, evenly spaced on \
                 the unit circle. The polynomial itself is not a landscape at any \
                 setting -- |z³| runs to 16 in the corners of a window this small, \
                 so it reads as a tower or, truncated, as a table.",
            ),
            preset(
                "moebius",
                "(z² - 1)/(z² + 1)",
                "(z^2 - 1)/(z^2 + 1)",
                d(-2.5, 2.5, -2.0, 2.0),
                d(-6.0, 6.0, -6.0, 6.0),
                Some(4.0),
                "Zeros at ±1, poles at ±i: two spires and two pits.",
            ),
            preset(
                "sqrt",
                "√z — a branch cut",
                "sqrt(z)",
                d(-2.5, 2.5, -2.0, 2.0),
                d(-6.0, 6.0, -6.0, 6.0),
                None,
                "The principal branch: the phase contours end on the cut along the \
                 negative real axis, where the surface is continuous and arg is not.",
            ),
            preset(
                "log",
                "log z",
                "log(z)",
                d(-2.5, 2.5, -2.0, 2.0),
                d(-6.0, 6.0, -6.0, 6.0),
                Some(3.0),
                "A logarithmic pole at the origin: the one spire that grows slowly \
                 enough to see the shape of.",
            ),
            preset(
                "erf",
                "erf z",
                "erf(z)",
                d(-3.0, 3.0, -2.5, 2.5),
                d(-8.0, 8.0, -8.0, 8.0),
                Some(5.0),
                "Entire, but it grows like exp(z²) off the real axis.",
            ),
            preset(
                "besselj",
                "J₀(z)",
                "besselj(0, z)",
                d(-11.0, 11.0, -3.0, 3.0),
                d(-25.0, 25.0, -8.0, 8.0),
                Some(3.0),
                "Oscillating and decaying along the real axis, growing off it.",
            ),
            preset(
                "digamma",
                "ψ(z) — digamma",
                "digamma(z)",
                d(-4.5, 4.5, 0.0, 2.5),
                d(-10.0, 10.0, -6.0, 6.0),
                Some(5.0),
                "Γ's logarithmic derivative: simple poles at the non-positive \
                 integers, and no zeros in sight.",
            ),
        ]
    }
}
