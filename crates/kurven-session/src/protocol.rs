//! The session as messages: an edit goes in as JSON, the state comes back as
//! JSON, and strokes, grids and files come back as typed arrays beside it
//! (HANDOFF-web-port.md, Architecture). `kurven-wasm` exports this to the
//! page's Worker; `web/app/src/engine` is the other end, and its types are
//! the reference for every message's fields.
//!
//! A request is a JSON object whose `kind` says what it is. A reply is
//! `{"ok": true, "state": ...}` plus what the request asked for, or
//! `{"ok": false, "error": "...", "state": ...}`; either names the arrays
//! that go with it, in order (`arrays`), each `{"name", "type"}` with type
//! `f64`, `f32`, `u32` or `u8`. Every edit settles the document before it
//! replies, as the app's headless run waits for it after every edit, so the
//! state a reply carries is the one the edit left.
//!
//! A bundle's files are staged first (`Session::stage`), since they are
//! bytes, and opened by name (`{"kind": "open", "name": ...}`).

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use kurven_core::bundle::BundleFiles;
use kurven_core::depth::DepthFrame;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::{CameraPreset, DomainAxis, LayerSource, LayerSpec, Truncation};
use kurven_core::navigation::Framing;
use kurven_core::navigation::{Gesture, Viewport};
use kurven_core::polyline::Layer;
use kurven_core::scene::PreviewMode;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::{Angle, P2, P3};
use kurven_core::surface_request::{Field, surface_key};
use kurven_core::svg;
use kurven_landscape::catalog::{Catalog, LandscapeRequest};
use kurven_surfaces::drawing::SurfacePreset;

use crate::document::{Document, State};
use crate::edits::Edited;
use crate::headless::{self, Output};
use crate::landscape::{Styling, applying_overrides};
use crate::preview::{Pipeline, Preview, Resources};
use crate::sliders::{Slider, reach};

/// A typed array a reply carries: its element type and its bytes,
/// little-endian, as the page's typed array holds them. Encoded once, from
/// the values where they lie, and moved, never copied again, to the module's
/// reply (`into_bytes`); the one copy after is the host's out of linear
/// memory, which the Worker transfers.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    kind: &'static str,
    bytes: Vec<u8>,
}

impl Array {
    pub fn f64(values: impl IntoIterator<Item = f64>) -> Array {
        Array { kind: "f64", bytes: values.into_iter().flat_map(f64::to_le_bytes).collect() }
    }

    pub fn f32(values: impl IntoIterator<Item = f32>) -> Array {
        Array { kind: "f32", bytes: values.into_iter().flat_map(f32::to_le_bytes).collect() }
    }

    pub fn u32(values: impl IntoIterator<Item = u32>) -> Array {
        Array { kind: "u32", bytes: values.into_iter().flat_map(u32::to_le_bytes).collect() }
    }

    /// Bytes as they are, moved in.
    pub fn u8(bytes: Vec<u8>) -> Array {
        Array { kind: "u8", bytes }
    }

    pub fn type_name(&self) -> &'static str {
        self.kind
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// A reply: its JSON, and the arrays it names, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub json: String,
    pub arrays: Vec<(String, Array)>,
}

/// One document and the bundles staged for it: what one Worker holds.
#[derive(Default)]
pub struct Session {
    pub document: Document,
    staged: BTreeMap<String, BundleFiles>,
    /// This engine instance's name, which starts every preview key: a
    /// Worker seeds it at load (`seed`), so a key held from an earlier
    /// Worker cannot name a scene of this one. Empty natively.
    nonce: String,
}

/// Why a request was refused, and whether it had reached the document
/// first: a refusal of its arguments touches nothing, one the document
/// makes (an `open` whose bundle cannot be read, an edit the document says
/// no to) has changed it. The reply says which (`touched`), so a client
/// replaying the requests that changed the document replays these too.
struct Refusal {
    message: String,
    touched: bool,
    /// More the reply says beside the message: where an expression went
    /// wrong (`at`).
    detail: JsonObject,
}

/// A refusal before the document was touched.
fn refuse(message: impl Into<String>) -> Refusal {
    Refusal { message: message.into(), touched: false, detail: JsonObject::new() }
}

/// A refusal the document made, having been touched.
fn refused_after(message: impl Into<String>) -> Refusal {
    Refusal { message: message.into(), touched: true, detail: JsonObject::new() }
}

/// An expression refused where it went wrong: the message, and the
/// character it went wrong at and how many after it are at fault, counted as
/// the language counts them (Unicode scalar values, a CR LF pair one), so
/// the formula field can mark them (`at`; a position of null for empty
/// input). Touches nothing.
fn positioned(error: &kurven_math::expression::ExpressionError) -> Refusal {
    let mut r = refuse(error.to_string());
    r.detail.insert(
        "at".to_string(),
        json::object([
            ("message", json::string(&error.message)),
            ("position", error.position.map_or(JsonValue::Null, |p| JsonValue::Int(p as i64))),
            ("length", JsonValue::Int(error.length as i64)),
        ]),
    );
    r
}

impl From<String> for Refusal {
    fn from(s: String) -> Refusal {
        refuse(s)
    }
}

impl From<&str> for Refusal {
    fn from(s: &str) -> Refusal {
        refuse(s)
    }
}

type Answer = Result<(JsonObject, Vec<(String, Array)>), Refusal>;

fn number(o: &JsonObject, key: &str) -> Result<f64, Refusal> {
    o.get(key).and_then(JsonValue::as_f64).ok_or_else(|| refuse(format!("wants a number '{key}'")))
}

fn text<'a>(o: &'a JsonObject, key: &str) -> Result<&'a str, Refusal> {
    o.get(key).and_then(JsonValue::as_str).ok_or_else(|| refuse(format!("wants a string '{key}'")))
}

fn numbers(o: &JsonObject, key: &str, n: usize) -> Result<Vec<f64>, Refusal> {
    let list = o.get(key).and_then(JsonValue::as_array).ok_or_else(|| refuse(format!("wants '{key}'")))?;
    let values: Vec<f64> = list.iter().filter_map(JsonValue::as_f64).collect();
    if values.len() != n || list.len() != n {
        return Err(refuse(format!("'{key}' wants {n} numbers")));
    }
    Ok(values)
}

/// A whole number: a count, a size, a resolution. A number with a fraction,
/// or past what an i64 holds, is refused, not truncated.
fn integer(o: &JsonObject, key: &str) -> Result<i64, Refusal> {
    let value = o.get(key).ok_or_else(|| refuse(format!("wants a whole number '{key}'")))?;
    if let JsonValue::Int(i) = value {
        return Ok(*i);
    }
    match value.as_f64() {
        Some(x) if x.fract() == 0.0 && x.abs() < 9.0e15 => Ok(x as i64),
        _ => Err(refuse(format!("'{key}' wants a whole number"))),
    }
}

/// An index: a whole number, not negative.
fn index(o: &JsonObject, key: &str) -> Result<usize, Refusal> {
    let i = integer(o, key)?;
    usize::try_from(i).map_err(|_| refuse(format!("'{key}' wants an index, not {i}")))
}

fn vec2(o: &JsonObject, key: &str) -> Result<Vec2, Refusal> {
    let v = numbers(o, key, 2)?;
    Ok(Vec2::new(v[0], v[1]))
}

fn layer_arrays(
    prefix: &str,
    layers: &[Layer],
    visible: &[usize],
    arrays: &mut Vec<(String, Array)>,
) -> Vec<JsonValue> {
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let vertices = layer.paths.vertices.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]);
            arrays.push((format!("{prefix}.{i}.vertices"), Array::f32(vertices)));
            arrays.push((format!("{prefix}.{i}.offsets"), Array::u32(layer.paths.offsets.iter().map(|&o| o as u32))));
            json::object([
                ("name", json::string(&layer.spec.name)),
                ("width", JsonValue::Double(layer.spec.width)),
                ("color", json::string(&layer.spec.color)),
                ("clipped", JsonValue::Bool(layer.spec.clipped)),
                ("visible", JsonValue::Bool(visible.contains(&i))),
            ])
        })
        .collect()
}

impl Session {
    pub fn new() -> Session {
        Session::default()
    }

    /// Stage a file of the bundle `bundle`, at `path` in it, for an `open`
    /// or a `headless` run to read.
    pub fn stage(&mut self, bundle: &str, path: &str, bytes: Vec<u8>) {
        // Under its path as a source reads it, so `./a.npy` staged again as
        // `a.npy` replaces it rather than standing beside it.
        self.staged.entry(bundle.to_string()).or_default().insert(path, bytes);
    }

    /// Forget every staged bundle.
    pub fn unstage(&mut self) {
        self.staged.clear();
    }

    /// Answer one request.
    pub fn call(&mut self, request: &str) -> Reply {
        let parsed = json::parse(request);
        let headless = matches!(&parsed, Ok(v) if v.get("kind").and_then(JsonValue::as_str) == Some("headless"));
        let answer = match &parsed {
            Ok(JsonValue::Object(o)) => self.answer(o),
            Ok(_) => Err(refuse("a request is a JSON object")),
            Err(error) => Err(refuse(format!("not JSON: {error}"))),
        };
        let (mut o, arrays) = match answer {
            Ok((mut o, arrays)) => {
                o.insert("ok".to_string(), JsonValue::Bool(true));
                o.insert("touched".to_string(), JsonValue::Bool(true));
                (o, arrays)
            }
            Err(Refusal { message, touched, detail }) => {
                let mut o = detail;
                o.insert("ok".to_string(), JsonValue::Bool(false));
                o.insert("touched".to_string(), JsonValue::Bool(touched));
                o.insert("error".to_string(), JsonValue::String(message));
                (o, Vec::new())
            }
        };
        if !headless {
            o.insert("state".to_string(), self.state());
        }
        o.insert(
            "arrays".to_string(),
            JsonValue::Array(
                arrays
                    .iter()
                    .map(|(n, a)| json::object([("name", json::string(n)), ("type", json::string(a.type_name()))]))
                    .collect(),
            ),
        );
        Reply { json: JsonValue::Object(o).canonical(), arrays }
    }

    fn answer(&mut self, o: &JsonObject) -> Answer {
        let kind = text(o, "kind")?;
        let d = &mut self.document;
        let none = || Ok((JsonObject::new(), Vec::new()));
        match kind {
            "state" => none(),
            "catalog" => {
                // The catalog's presets, for the formula's dropdown: each
                // its name, label, notes, expression, the rectangle worth
                // looking at and how far the window's sliders reach.
                let catalog = Catalog::native();
                let mut out = JsonObject::new();
                out.insert("presets".to_string(), JsonValue::Array(catalog.presets.iter().map(|p| p.json()).collect()));
                out.insert("defaultResolution".to_string(), JsonValue::Int(catalog.default_resolution));
                out.insert("controls".to_string(), reach::json());
                // The picker's cells: every function and surface, each with
                // the key its picture is cached under (`thumbnails.rs`).
                out.insert(
                    "entries".to_string(),
                    JsonValue::Array(crate::thumbnails::Entry::all().iter().map(|e| e.json()).collect()),
                );
                Ok((out, Vec::new()))
            }
            "thumbnail" => {
                // A catalog entry's picture for the picker, built in a scene
                // of its own (the open document is not touched): the frame
                // the window's preview draws of it at rest, for the page's
                // GPU, or with `judged` the judged strokes, for a page
                // without one (`Thumbnails.swift`).
                let id = text(o, "entry")?;
                let entry = crate::thumbnails::Entry::named(id).ok_or_else(|| refuse(format!("no entry '{id}'")))?;
                let thumbnail = crate::thumbnails::draw(&entry).map_err(refused_after)?;
                let mut out = JsonObject::new();
                out.insert("key".to_string(), json::string(&crate::thumbnails::key(&entry)));
                let size = crate::thumbnails::SIZE;
                if o.get("judged").and_then(JsonValue::as_bool) == Some(true) {
                    let mut scene = thumbnail.scene;
                    // A quarter of a drawable pixel, as the window bakes.
                    scene.resolution = 0.25 * thumbnail.navigator.framing.units_per_pixel;
                    let options = kurven_bake::BakeOptions::new(size.width.max(size.height));
                    let bake = kurven_bake::bake(&scene, &options).map_err(|e| refused_after(e.to_string()))?;
                    let mut arrays = Vec::new();
                    let styles = stroke_arrays(&bake.strokes, &mut arrays);
                    out.insert("strokes".to_string(), JsonValue::Array(styles));
                    out.insert("frame".to_string(), frame_json(&thumbnail.navigator.framing.frame(size)));
                    return Ok((out, arrays));
                }
                let visible: Vec<usize> = (0..thumbnail.scene.layers().len()).collect();
                let at = crate::preview::Framed {
                    navigator: thumbnail.navigator,
                    viewport: size,
                    mode: PreviewMode::Drawing,
                };
                let mut memo = crate::preview::Memo::default();
                let preview = crate::preview::preview_scene(
                    &mut memo,
                    &thumbnail.scene,
                    at,
                    &visible,
                    false,
                    &self.nonce,
                    None,
                    &BTreeSet::new(),
                );
                let mut arrays = Vec::new();
                for (k, v) in preview_json(&preview, &mut arrays) {
                    out.insert(k, v);
                }
                Ok((out, arrays))
            }
            "edited" => {
                // What is edited now, whole (`Document::edited`): a
                // landscape's request, the styling it is drawn with and the
                // level counts. What the page's URL holds, and `restore`
                // takes back.
                let mut out = JsonObject::new();
                out.insert("edited".to_string(), d.edited().map_or(JsonValue::Null, |e| edited_json(&e)));
                Ok((out, Vec::new()))
            }
            "restore" => {
                // A landscape as a link names it: `edited`'s fields.
                let request = o.get("landscape").ok_or("wants a 'landscape'")?;
                let request = LandscapeRequest::from_json(request).map_err(|e| refuse(e.to_string()))?;
                kurven_math::expression::Expression::compile(&request.expression).map_err(|e| positioned(&e))?;
                let layers = o
                    .get("layers")
                    .and_then(JsonValue::as_array)
                    .ok_or("wants 'layers'")?
                    .iter()
                    .map(LayerSpec::from_json)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| refuse(e.to_string()))?;
                let truncation = Truncation::from_json(o.get("truncation").ok_or("wants a 'truncation'")?)
                    .map_err(|e| refuse(e.to_string()))?;
                let mut counts = BTreeMap::new();
                if let Some(c) = o.get("levelCounts") {
                    let c = c.as_object().ok_or("'levelCounts' wants an object")?;
                    for (name, n) in c {
                        let n =
                            n.as_i64().ok_or_else(|| refuse(format!("the level count of {name} is a whole number")))?;
                        counts.insert(name.clone(), n);
                    }
                }
                linkable(&request, &layers, &truncation, &counts)?;
                // Made in a document of its own, which takes what an open
                // keeps from this one, and put in this one's place only once
                // it holds the landscape: a link that cannot be made leaves
                // the open document, and what it can undo, as they were.
                let mut next = Document {
                    viewport: d.viewport,
                    mode: d.mode,
                    draft_resolution: d.draft_resolution,
                    refine_contours: d.refine_contours,
                    ..Document::default()
                };
                next.restore_landscape(request, Styling { truncation, layers }, counts);
                if next.bundle().is_none() {
                    let why = match next.state() {
                        State::Failed(_, why) => why.clone(),
                        _ => next.landscape_status.clone().unwrap_or_else(|| "nothing was sampled".to_string()),
                    };
                    return Err(refuse(format!("the link's landscape could not be made: {why}")));
                }
                *d = next;
                none()
            }
            "depthTolerance" => {
                // The Visibility section's field (Inspector.swift): how far
                // behind a surface a point may lie and still be drawn.
                let value = number(o, "value")?;
                if !(value.is_finite() && value >= 0.0) {
                    return Err(refuse(format!("a depth tolerance is a number at least 0, not {value}")));
                }
                d.depth_tolerance = value;
                none()
            }
            "unstage" => {
                // A staged bundle let go of once it is opened or refused
                // (Phase 8, step 7): the document holds what it read, and a
                // file's bytes need not stay. Nothing of the document is
                // touched; a name not staged is let go of already.
                let name = text(o, "name")?.to_string();
                self.staged.remove(&name);
                none()
            }
            "open" if o.get("untrusted").and_then(JsonValue::as_bool) == Some(true) => {
                // A bundle opened from a file (Phase 8, step 7): someone
                // else's bytes, held to the controls' reach before anything
                // is built (`files::openable`), and opened in a document of
                // its own, which takes what an open keeps from this one, put
                // in this one's place only once it is open. A bundle refused
                // or that will not open leaves the open document, and what
                // it can undo, as they were.
                let name = text(o, "name")?;
                let Session { document: d, staged, .. } = self;
                let files = staged.get(name).ok_or_else(|| refuse(format!("no bundle '{name}' staged")))?;
                let source = kurven_core::bundle::Source::Memory { name, files };
                // Said by the bundle's own name, the last part of the one
                // it is staged under.
                let shown = source.name();
                crate::files::openable(&source).map_err(|why| refuse(format!("{shown} was not opened: {why}")))?;
                let mut next = Document {
                    viewport: d.viewport,
                    mode: d.mode,
                    draft_resolution: d.draft_resolution,
                    refine_contours: d.refine_contours,
                    ..Document::default()
                };
                next.load(name, files);
                next.settle();
                if let State::Failed(_, why) = next.state() {
                    return Err(refuse(format!("{shown} could not be opened: {why}")));
                }
                *d = next;
                none()
            }
            "open" => {
                let name = text(o, "name")?;
                let Session { document: d, staged, .. } = self;
                let files = staged.get(name).ok_or_else(|| refuse(format!("no bundle '{name}' staged")))?;
                d.load(name, files);
                d.settle();
                if let State::Failed(_, why) = d.state() {
                    return Err(refused_after(why.clone()));
                }
                none()
            }
            "create" => {
                let name = text(o, "preset")?;
                let preset =
                    Catalog::native().preset(name).ok_or_else(|| refuse(format!("no preset '{name}'")))?.clone();
                d.create(&preset);
                d.settle();
                none()
            }
            "createSurface" => {
                let name = text(o, "preset")?;
                let preset = SurfacePreset::named(name).ok_or_else(|| refuse(format!("no surface '{name}'")))?;
                d.create_surface(&preset);
                d.settle();
                none()
            }
            "expression" => {
                // The formula field: a new expression, the window kept.
                let expression = text(o, "expression")?.to_string();
                kurven_math::expression::Expression::compile(&expression).map_err(|e| positioned(&e))?;
                if d.landscape_request().is_none() {
                    return Err("an expression is a landscape's".into());
                }
                d.edit_landscape("Expression", |r| {
                    r.expression = expression;
                    r.name = String::new();
                });
                d.settle();
                none()
            }
            "set" => {
                // A number typed into a control: a surface's field, or a
                // landscape's window or resolution (`--set`).
                let key = text(o, "key")?.to_string();
                let value = number(o, "value")?;
                d.paint_status = None;
                if let Some(surface) = d.surface_request().cloned() {
                    let field = Field::parse(&key)
                        .filter(|&f| surface.get(f).is_some())
                        .ok_or_else(|| refuse(format!("the surface has no field '{key}'")))?;
                    surface.setting(field, value, &surface_key).map_err(|e| refuse(e.to_string()))?;
                    Slider::SurfaceField { field, label: key }.set(d, value);
                } else if d.landscape_request().is_some() {
                    let slider = Slider::Landscape { label: key.clone() };
                    slider.set_landscape(d, |r| *r = applying_overrides(r, &[(key.as_str(), value)]));
                } else {
                    return Err("this bundle has no request to edit".into());
                }
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(refused_after(why)),
                    None => none(),
                }
            }
            "surfaceField" => {
                // A surface's number typed, stepped or switched
                // (SurfaceSection.swift's field, stepper and Conformal
                // switch): clamped to the control's range, one undo step
                // named for the control. Refused, touching nothing, for a
                // field the shape has not or a value the request refuses.
                let raw = text(o, "field")?;
                let field = Field::parse(raw).ok_or_else(|| refuse(format!("no field '{raw}'")))?;
                let value = number(o, "value")?;
                let request = d.surface_request().ok_or("this is not a surface")?;
                crate::surface_controls::settable(request, field, value).map_err(refuse)?;
                d.paint_status = None;
                d.type_surface_field(field, value);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(refused_after(why)),
                    None => none(),
                }
            }
            "surfaceLines" => {
                // The coordinate curves switched on or off; a plain torus is
                // its lines and has no switch.
                let on = o.get("on").and_then(JsonValue::as_bool).ok_or("wants 'on'")?;
                let request = d.surface_request().ok_or("this is not a surface")?;
                if matches!(request.shape, kurven_core::surface_request::Shape::Torus { .. }) {
                    return Err("a plain torus is its lines: they have no switch".into());
                }
                d.switch_lines(on);
                d.settle();
                none()
            }
            "surfaceLineCount" => {
                // One of the coordinate curves' counts, clamped to 1...200.
                let around = match text(o, "axis")? {
                    "around" => true,
                    "across" => false,
                    other => return Err(refuse(format!("an axis is around or across, not {other}"))),
                };
                let count = integer(o, "count")?;
                if d.surface_request().and_then(|r| r.lines).is_none() {
                    return Err("the surface draws no coordinate curves".into());
                }
                d.set_line_count(around, count);
                d.settle();
                none()
            }
            "linearCurve" => {
                let on = o.get("on").and_then(JsonValue::as_bool).ok_or("wants 'on'")?;
                d.surface_request().ok_or("this is not a surface")?;
                d.switch_linear_curve(on);
                d.settle();
                none()
            }
            "component" => {
                // The forced torus's radius or height drawn from x, x′ or x″.
                let field = match text(o, "field")? {
                    "radial" => Field::Radial,
                    "axial" => Field::Axial,
                    other => return Err(refuse(format!("a component is radial or axial, not {other}"))),
                };
                let value = integer(o, "value")?;
                if !(0..=2).contains(&value) {
                    return Err(refuse(format!("a component is 0, 1 or 2, not {value}")));
                }
                if d.surface_request().and_then(|r| r.get(field)).is_none() {
                    return Err("the surface has no state components".into());
                }
                d.choose_component(field, value);
                d.settle();
                none()
            }
            "periods" => {
                // The periods "Function…" offers for an expression: the
                // catalog's rectangle for it (`known`, which a sheet takes
                // as the expression is typed), and the one it opens with
                // (`offered`: the painted function's own, the catalog's, or
                // the surface's). A view: nothing is changed.
                let expression = text(o, "expression")?;
                let mut out = JsonObject::new();
                let pair = |p: (f64, f64)| json::doubles(&[p.0, p.1]);
                out.insert(
                    "known".to_string(),
                    kurven_surfaces::drawing::periods_of(expression).map_or(JsonValue::Null, pair),
                );
                out.insert("offered".to_string(), pair(d.function_periods(expression)));
                Ok((out, Vec::new()))
            }
            "drag" => {
                // A slider taken hold of, moved, let go: one undo step.
                let slider = slider(o)?;
                match &slider {
                    Slider::Levels { name } => with_levels(d, name)?,
                    Slider::Strokes { name } => with_strokes(d, name)?,
                    Slider::Light { name } => with_light(d, name)?,
                    _ => {}
                }
                match text(o, "phase")? {
                    "begin" => slider.editing_changed(d, true),
                    "end" => slider.editing_changed(d, false),
                    "move" => match &slider {
                        Slider::Landscape { label } => {
                            let value = number(o, "value")?;
                            let key = label.clone();
                            slider.set_landscape(d, |r| *r = applying_overrides(r, &[(key.as_str(), value)]));
                        }
                        Slider::Light { .. } => {
                            let v = numbers(o, "value", 2)?;
                            slider.set_light(d, v[0], v[1]);
                        }
                        Slider::SurfaceField { field, .. }
                            if o.get("rest").and_then(JsonValue::as_bool) == Some(true) =>
                        {
                            // A slider's own value: clamped, then rested at
                            // its step or a detent within reach.
                            let value = crate::surface_controls::slid(d, *field, number(o, "value")?);
                            slider.set(d, value);
                        }
                        // A banded truncation's height dragged: the whole
                        // truncation, as TruncationSection's band rows set it.
                        Slider::Truncation if o.contains_key("truncation") => {
                            let t = Truncation::from_json(o.get("truncation").ok_or("wants a 'truncation'")?)
                                .map_err(|e| refuse(e.to_string()))?;
                            d.edit_truncation(t, "Truncation");
                        }
                        _ => slider.set(d, number(o, "value")?),
                    },
                    other => return Err(refuse(format!("a drag's phase is begin, move or end, not {other}"))),
                }
                d.settle();
                none()
            }
            "levels" => {
                let name = text(o, "layer")?;
                let count = number(o, "count")?;
                with_levels(d, name)?;
                Slider::Levels { name: name.to_string() }.set(d, count);
                d.settle();
                none()
            }
            "resetLevels" => {
                let name = text(o, "layer")?;
                with_levels(d, name)?;
                d.paint_reset_levels(name);
                d.settle();
                none()
            }
            "strokes" => {
                let name = text(o, "layer")?;
                let count = number(o, "count")?;
                with_strokes(d, name)?;
                d.paint_status = None;
                Slider::Strokes { name: name.to_string() }.set(d, count);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(refused_after(why)),
                    None => none(),
                }
            }
            "relief" => {
                let value = number(o, "value")?;
                d.paint_status = None;
                Slider::SurfaceField { field: Field::Relief, label: "Relief".to_string() }.set(d, value);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(refused_after(why)),
                    None => none(),
                }
            }
            "truncation" => {
                let truncation = Truncation::from_json(o.get("truncation").ok_or("wants a 'truncation'")?)
                    .map_err(|e| refuse(e.to_string()))?;
                d.edit_truncation(truncation, "Truncation");
                d.settle();
                none()
            }
            "pick" => {
                if o.contains_key("pixel") {
                    d.pick_at_pixel(vec2(o, "pixel")?);
                } else if o.contains_key("point") {
                    let p = numbers(o, "point", 3)?;
                    d.pick(Some(P3::new(p[0], p[1], p[2])));
                } else {
                    d.pick(None);
                }
                d.settle();
                none()
            }
            "add" => {
                let name = text(o, "name")?;
                let pick = d.picked().copied().ok_or("nothing is picked to paint on")?;
                let choice = d
                    .paint_choices(&pick)
                    .into_iter()
                    .find(|c| c.id() == name)
                    .ok_or_else(|| refuse(format!("nothing called '{name}' to add here")))?;
                d.add(&choice);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(refused_after(why)),
                    None => none(),
                }
            }
            "remove" => {
                let name = text(o, "layer")?;
                let pick = d.picked().copied().ok_or("nothing is picked")?;
                let layers = d.edited().map(|e| e.layers).unwrap_or_default();
                let index = d
                    .painted_layers(&pick)
                    .into_iter()
                    .find(|&i| layers.get(i).is_some_and(|l| l.name == name))
                    .ok_or_else(|| refuse(format!("no layer '{name}' on the picked face")))?;
                d.remove_layer(index).map_err(|e| refuse(e.0))?;
                d.settle();
                none()
            }
            "function" => {
                let expression = text(o, "expression")?.to_string();
                let periods = match o.get("periods") {
                    Some(_) => {
                        let p = numbers(o, "periods", 2)?;
                        (p[0], p[1])
                    }
                    None => d.function_periods(&expression),
                };
                d.add_function(&expression, periods).map_err(|e| refuse(e.0))?;
                d.settle();
                none()
            }
            "undo" => {
                d.settle();
                d.undo();
                d.settle();
                none()
            }
            "redo" => {
                d.settle();
                d.redo();
                d.settle();
                none()
            }
            "axis" => {
                let axis = match text(o, "axis")? {
                    "real" => DomainAxis::Real,
                    "imag" => DomainAxis::Imag,
                    other => return Err(refuse(format!("an axis is real or imag, not {other}"))),
                };
                d.paint_axis(axis);
                d.settle();
                none()
            }
            "light" => {
                let l = numbers(o, "direction", 3)?;
                let name = text(o, "layer")?;
                with_light(d, name)?;
                if !crate::files::usable_light(Vec3::new(l[0], l[1], l[2])) {
                    return Err(refuse("a light is a direction: finite, not zero, and short enough to normalize"));
                }
                d.paint_light(Vec3::new(l[0], l[1], l[2]), name);
                d.settle();
                none()
            }
            "hide" => {
                let layer = index(o, "layer")?;
                if layer >= d.layers().len() {
                    return Err(refuse(format!("no layer {layer}")));
                }
                d.toggle(layer);
                none()
            }
            "gesture" => {
                let g = o.get("gesture").and_then(JsonValue::as_object).ok_or("wants a 'gesture'")?;
                d.apply(&gesture(g)?);
                none()
            }
            "gestures" => {
                // Gestures in order, each applied as its own `gesture`
                // request applies it: what the page sends for the gestures
                // a hand gave while the engine was busy, so the camera is
                // the one they make one by one, to the bit, with one
                // request where there would be a queue. All read before
                // any is applied, so a malformed one applies none.
                let list = o.get("gestures").and_then(JsonValue::as_array).ok_or("wants 'gestures'")?;
                let gestures = list
                    .iter()
                    .map(|g| g.as_object().ok_or_else(|| refuse("a gesture is an object")).and_then(gesture))
                    .collect::<Result<Vec<_>, _>>()?;
                for g in &gestures {
                    d.apply(g);
                }
                none()
            }
            "camera" => {
                // A camera set outright: what a run of gestures left it, as
                // the state's `camera.exact` reports it, to the bit. Gestures
                // move nothing else (`Navigator::applying`), so a client can
                // replay a run of them as this one request.
                let Some(mut n) = d.navigator else { return none() };
                let target = numbers(o, "target", 3)?;
                let center = numbers(o, "center", 2)?;
                let angles = [number(o, "azimuth")?, number(o, "elevation")?];
                if !target.iter().chain(&center).chain(&angles).all(|x| x.is_finite()) {
                    return Err(refuse("a camera's angles, target and centre are finite numbers"));
                }
                let upp = number(o, "unitsPerPixel")?;
                if !(upp.is_finite() && upp > 0.0) {
                    return Err(refuse(format!("a camera's units per pixel are a positive number, not {upp}")));
                }
                let distance = match o.get("distance") {
                    None | Some(JsonValue::Null) => None,
                    Some(_) => Some(number(o, "distance")?),
                };
                if let Some(x) = distance
                    && !(x.is_finite() && x > 0.0)
                {
                    return Err(refuse(format!("a camera's distance is a positive number, not {x}")));
                }
                n.orbit.azimuth = Angle::from_radians(angles[0]);
                n.orbit.elevation = Angle::from_radians(angles[1]);
                n.orbit.target = P3::new(target[0], target[1], target[2]);
                n.orbit.distance = distance;
                n.framing = Framing::new(P2::new(center[0], center[1]), upp);
                d.navigator = Some(n);
                none()
            }
            "fit" => {
                d.fit();
                none()
            }
            "preset" => {
                let presets = presets(d);
                let preset = presets.get(index(o, "index")?).cloned().ok_or("no such preset")?;
                d.use_preset(&preset);
                none()
            }
            "perspective" => {
                d.set_perspective(o.get("on").and_then(JsonValue::as_bool).unwrap_or(false));
                none()
            }
            "viewport" => {
                let (width, height) = (integer(o, "width")?, integer(o, "height")?);
                if width < 1 || height < 1 {
                    return Err(refuse(format!("a viewport is at least a pixel, not {width} by {height}")));
                }
                d.viewport = Viewport::new(width, height);
                none()
            }
            "mode" => {
                d.mode = match text(o, "mode")? {
                    "drawing" => PreviewMode::Drawing,
                    "depth" => PreviewMode::Depth,
                    "shaded" => PreviewMode::Shaded(kurven_core::scene::Lighting::default()),
                    other => return Err(refuse(format!("no mode '{other}'"))),
                };
                none()
            }
            "refine" => {
                d.set_refinement(o.get("on").and_then(JsonValue::as_bool).unwrap_or(true));
                d.settle();
                none()
            }
            "draftResolution" => {
                let value = integer(o, "value")?;
                if value < 8 {
                    return Err(refuse(format!("a draft resolution is at least 8, not {value}")));
                }
                d.draft_resolution = value;
                none()
            }
            "bakeResolution" => {
                // Refused as the bake would refuse it, before it is set.
                let value = integer(o, "value")?;
                if let Some(why) = CameraPreset::unsupported_buffer(value) {
                    return Err(refuse(format!("the bake's resolution {why}")));
                }
                d.bake_resolution = value;
                none()
            }
            "frame" => {
                // The curves a frame draws, in world space: a draft's, or the
                // window's layout at rest, of the scene the window shows (the
                // picked face's boundary last), each saying whether it is
                // drawn (`shown_visible_layers`).
                let draft = o.get("draft").and_then(JsonValue::as_bool).unwrap_or(true);
                let layers = if draft { d.draft_layers() } else { d.rest_layout() };
                let mut arrays = Vec::new();
                let described = layer_arrays("layer", &layers, &d.shown_visible_layers(), &mut arrays);
                let mut out = JsonObject::new();
                out.insert("layers".to_string(), JsonValue::Array(described));
                Ok((out, arrays))
            }
            "seed" => {
                // The engine instance's name, which a Worker gives at load:
                // letters and digits, at most 64.
                let nonce = text(o, "nonce")?;
                if nonce.is_empty() || nonce.len() > 64 || !nonce.chars().all(|c| c.is_ascii_alphanumeric()) {
                    return Err(refuse("a nonce is 1 to 64 letters and digits"));
                }
                self.nonce = nonce.to_string();
                none()
            }
            "preview" => {
                // A frame of the preview, for the page's GPU: the resources
                // and each layer's segments, under keys, and the uniforms.
                // What the page says it holds (`have`) is not sent again.
                let draft = o.get("draft").and_then(JsonValue::as_bool).unwrap_or(true);
                let have = o.get("have").and_then(JsonValue::as_object);
                let content = have.and_then(|h| h.get("content")).and_then(JsonValue::as_str);
                let keys: BTreeSet<String> = have
                    .and_then(|h| h.get("layers"))
                    .and_then(JsonValue::as_array)
                    .map(|a| a.iter().filter_map(|k| k.as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                let Some(preview) = self.document.preview(draft, &self.nonce, content, &keys) else {
                    return Err(refuse("nothing to preview"));
                };
                let mut arrays = Vec::new();
                let out = preview_json(&preview, &mut arrays);
                Ok((out, arrays))
            }
            "surfaceImage" => {
                // A parametric surface's coordinate image as the bake's CPU
                // rasterizer renders it (`SurfaceImage::rasterize`), over the
                // frame covering its grid at `resolution`, with what the GPU
                // needs to render the same frame: the view, the frame's clip
                // (buffer order) and the coordinate pass's depth range,
                // padded as `renderSurface` pads it. What the browser lane
                // holds the WebGPU coordinate pass to.
                let resolution = integer(o, "resolution")?;
                if !(2..=4096).contains(&resolution) {
                    return Err(refuse(format!("a resolution is 2 to 4096, not {resolution}")));
                }
                let scene = d.shown_scene().ok_or("nothing shown")?;
                let surface = scene.parametric().ok_or("the scene is not a parametric surface")?;
                let view = scene.camera.view;
                let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
                for &p in surface.positions.iter() {
                    let q = view.apply(p).v;
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
                let bounds = kurven_core::spaces::AABB::new(lo, hi);
                let frame = DepthFrame::covering(&bounds, resolution);
                let image = kurven_core::surface_image::SurfaceImage::rasterize(surface, &view, frame);
                let pad = 1e-3 * (hi.z - lo.z) + 1e-6;
                let flat = |m: &kurven_core::simd::Mat4| {
                    json::doubles(&m.columns.iter().flat_map(|c| c.to_array()).collect::<Vec<_>>())
                };
                let mut out = JsonObject::new();
                out.insert("frame".to_string(), frame_json(&frame));
                out.insert("view".to_string(), flat(&view.m));
                out.insert("clip".to_string(), flat(&frame.metal_clip()));
                out.insert("surfaceDepthRange".to_string(), json::doubles(&[lo.z - pad, hi.z + pad]));
                out.insert("span".to_string(), JsonValue::Double(hi.z - lo.z));
                let arrays = vec![
                    ("depth".to_string(), Array::f32(image.depth.values.iter().copied())),
                    ("coords".to_string(), Array::f32(image.coords.iter().flatten().copied())),
                ];
                Ok((out, arrays))
            }
            "bake" => {
                // The judged strokes, what Export writes: each layer's
                // page-space points and its offsets, and the SVG when asked.
                // The frame maps a page point to the viewport's pixel, as the
                // preview's does.
                let frame = d.navigator.map(|n| n.framing.frame(d.viewport));
                let strokes = d.bake().map_err(refuse)?;
                let mut arrays = Vec::new();
                let styles = stroke_arrays(&strokes, &mut arrays);
                if o.get("svg").and_then(JsonValue::as_bool) == Some(true) {
                    let text = svg::render(&strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS);
                    arrays.push(("svg".to_string(), Array::u8(text.into_bytes())));
                }
                let mut out = JsonObject::new();
                out.insert("strokes".to_string(), JsonValue::Array(styles));
                out.insert("frame".to_string(), frame.as_ref().map_or(JsonValue::Null, frame_json));
                Ok((out, arrays))
            }
            "grids" => {
                // What the draft preview draws the surface from: a
                // landscape's grids of |f| and arg f, a parametric surface's
                // sampled positions, a solid's triangles.
                let mut arrays = Vec::new();
                let mut out = JsonObject::new();
                if let Some(b) = d.bundle() {
                    let h = &*b.surface.height;
                    out.insert("shape".to_string(), json::ints(&[h.height as i64, h.width as i64]));
                    out.insert("domain".to_string(), b.manifest.domain.json());
                    arrays.push(("height".to_string(), Array::f32(h.values.iter().copied())));
                    if let Some(p) = &b.surface.phase {
                        arrays.push(("phase".to_string(), Array::f32(p.values.iter().copied())));
                    }
                } else if let Some(s) = d.scene().and_then(|s| s.parametric()) {
                    out.insert("shape".to_string(), json::ints(&[s.v.samples as i64, s.u.samples as i64]));
                    let p = s.positions.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]);
                    arrays.push(("positions".to_string(), Array::f32(p)));
                } else if let Some(solid) = d.scene().and_then(|s| s.solid()) {
                    let mesh = solid.mesh();
                    let v = mesh.vertices.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]);
                    let t = mesh.triangles.iter().flat_map(|t| t.map(|k| k as u32));
                    arrays.push(("vertices".to_string(), Array::f32(v)));
                    arrays.push(("triangles".to_string(), Array::u32(t)));
                }
                Ok((out, arrays))
            }
            "save" => {
                let name = text(o, "name")?;
                let files = d.save_bundle(name).map_err(refuse)?;
                let mut arrays = Vec::new();
                let mut names = Vec::new();
                for (path, bytes) in files {
                    names.push(json::string(&path));
                    arrays.push((format!("file:{path}"), Array::u8(bytes)));
                }
                let mut out = JsonObject::new();
                out.insert("files".to_string(), JsonValue::Array(names));
                Ok((out, arrays))
            }
            "headless" => {
                // The app's command line on the staged bundles: the gate's
                // run, under Node, of what `kurven-cli app` runs natively.
                let args: Vec<String> = o
                    .get("args")
                    .and_then(JsonValue::as_array)
                    .ok_or("wants 'args'")?
                    .iter()
                    .filter_map(|a| a.as_str().map(str::to_string))
                    .collect();
                let staged = &self.staged;
                let run = headless::run(&args, &|path: &str| {
                    staged.get(path).map(Cow::Borrowed).ok_or_else(|| format!("{path} is not a bundle directory"))
                });
                let mut arrays = Vec::new();
                let mut outputs = Vec::new();
                for output in run.outputs {
                    match output {
                        Output::Bundle { path, files } => {
                            outputs.push(json::string(&path));
                            for (rel, bytes) in files {
                                arrays.push((format!("{path}/{rel}"), Array::u8(bytes)));
                            }
                        }
                        Output::File { path, bytes } => {
                            outputs.push(json::string(&path));
                            arrays.push((path, Array::u8(bytes)));
                        }
                    }
                }
                let lines = |v: &[String]| JsonValue::Array(v.iter().map(|s| json::string(s)).collect());
                let mut out = JsonObject::new();
                out.insert("exit".to_string(), JsonValue::Int(run.exit as i64));
                out.insert("stdout".to_string(), lines(&run.stdout));
                out.insert("stderr".to_string(), lines(&run.stderr));
                out.insert("outputs".to_string(), JsonValue::Array(outputs));
                Ok((out, arrays))
            }
            other => Err(refuse(format!("no request '{other}'"))),
        }
    }

    /// The document as the page shows it: everything a control reads.
    pub fn state(&mut self) -> JsonValue {
        let d = &mut self.document;
        let kind = match d.state() {
            State::Empty => "empty",
            State::Failed(..) => "failed",
            State::Ready(crate::document::Opened::Landscape(_)) => "landscape",
            State::Ready(crate::document::Opened::Surface(_)) => "surface",
            State::Ready(crate::document::Opened::Polyhedron(_)) => "polyhedron",
        };
        let mut o = JsonObject::new();
        o.insert("kind".to_string(), json::string(kind));
        if let State::Failed(_, why) = d.state() {
            o.insert("failure".to_string(), json::string(why));
        }
        o.insert("title".to_string(), json::string(&d.title()));
        let opt = |s: &Option<String>| s.as_ref().map_or(JsonValue::Null, |s| json::string(s));
        o.insert(
            "status".to_string(),
            json::object([
                ("landscape", opt(&d.landscape_status)),
                ("surface", opt(&d.surface_status)),
                ("polyhedron", opt(&d.polyhedron_status)),
                ("paint", opt(&d.paint_status)),
                ("pick", opt(&d.pick_status)),
                ("bake", opt(&d.bake_status)),
            ]),
        );
        o.insert("landscape".to_string(), d.landscape_request().map_or(JsonValue::Null, |r| r.json()));
        o.insert("surface".to_string(), d.surface_request().map_or(JsonValue::Null, |r| r.json()));
        o.insert("surfaceControls".to_string(), crate::surface_controls::json(d));
        o.insert("polyhedron".to_string(), d.polyhedron_bundle().map_or(JsonValue::Null, polyhedron_json));
        o.insert("truncation".to_string(), if d.bundle().is_some() { d.truncation().json() } else { JsonValue::Null });
        // The layers as the controls show them: the edited ones, each with
        // what its control reads.
        let edited = d.edited();
        let specs = edited.as_ref().map(|e| e.layers.clone()).unwrap_or_default();
        let drawn = d.layer_specs();
        let layers: Vec<JsonValue> = specs
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                let mut l = JsonObject::new();
                l.insert("name".to_string(), json::string(&spec.name));
                l.insert("kind".to_string(), json::string(spec.source.kind()));
                l.insert("role".to_string(), json::string(spec.role.raw()));
                l.insert("width".to_string(), JsonValue::Double(spec.width));
                l.insert("color".to_string(), json::string(&spec.color));
                l.insert("hidden".to_string(), JsonValue::Bool(d.hidden_layers.contains(&i)));
                if let Some(index) = drawn.iter().position(|s| s.name == spec.name)
                    && let Some(levels) = d.levels_for_layer(index)
                {
                    l.insert("levels".to_string(), JsonValue::Int(levels.len() as i64));
                }
                if let Some(count) = d.hatch_count(i) {
                    l.insert("strokes".to_string(), JsonValue::Int(count));
                }
                if let Some(light) = spec.source.light() {
                    l.insert("light".to_string(), json::doubles(&[light.x, light.y, light.z]));
                    if let Some((azimuth, elevation)) = light_angles(light) {
                        l.insert("lightAngles".to_string(), json::doubles(&[azimuth, elevation]));
                    }
                }
                if let Some(axis) = spec.source.plateau_hatch_axis() {
                    let axis = if axis == DomainAxis::Real { "real" } else { "imag" };
                    l.insert("axis".to_string(), json::string(axis));
                }
                // The drawn layer of that name: how many paths, and whether
                // it is tested for visibility (the Layers list's count and
                // "unclipped").
                if let Some(layer) = d.layers().iter().find(|x| x.spec.name == spec.name) {
                    l.insert("paths".to_string(), JsonValue::Int(layer.paths.count() as i64));
                    l.insert("clipped".to_string(), JsonValue::Bool(layer.spec.clipped));
                }
                if let LayerSource::Hatching { hatching, .. } = &spec.source {
                    l.insert("spacing".to_string(), JsonValue::Double(hatching.spacing));
                }
                JsonValue::Object(l)
            })
            .collect();
        o.insert("layers".to_string(), JsonValue::Array(layers));
        o.insert(
            "levelCounts".to_string(),
            JsonValue::Object(d.level_counts().iter().map(|(k, v)| (k.clone(), JsonValue::Int(*v))).collect()),
        );
        if let Some(n) = d.navigator {
            o.insert(
                "camera".to_string(),
                json::object([
                    ("azimuth", JsonValue::Double(n.orbit.azimuth.degrees())),
                    ("elevation", JsonValue::Double(n.orbit.elevation.degrees())),
                    ("target", json::doubles(&[n.orbit.target.x(), n.orbit.target.y(), n.orbit.target.z()])),
                    ("center", json::doubles(&[n.framing.center.x(), n.framing.center.y()])),
                    ("unitsPerPixel", JsonValue::Double(n.framing.units_per_pixel)),
                    ("perspective", JsonValue::Bool(n.orbit.is_perspective())),
                    // The values a `camera` request sets, to the bit: the
                    // angles in radians as the orbit holds them.
                    (
                        "exact",
                        json::object([
                            ("azimuth", JsonValue::Double(n.orbit.azimuth.radians)),
                            ("elevation", JsonValue::Double(n.orbit.elevation.radians)),
                            ("target", json::doubles(&[n.orbit.target.x(), n.orbit.target.y(), n.orbit.target.z()])),
                            ("distance", n.orbit.distance.map_or(JsonValue::Null, JsonValue::Double)),
                            ("center", json::doubles(&[n.framing.center.x(), n.framing.center.y()])),
                            ("unitsPerPixel", JsonValue::Double(n.framing.units_per_pixel)),
                        ]),
                    ),
                ]),
            );
        }
        o.insert("viewport".to_string(), json::ints(&[d.viewport.width, d.viewport.height]));
        o.insert("presets".to_string(), JsonValue::Array(presets(d).iter().map(|p| json::string(&p.name)).collect()));
        let picked = d.picked().copied();
        o.insert(
            "picked".to_string(),
            match picked {
                None => JsonValue::Null,
                Some(pick) => {
                    let choices: Vec<JsonValue> = d
                        .paint_choices(&pick)
                        .iter()
                        .map(|c| json::object([("id", json::string(c.id())), ("label", json::string(&c.label))]))
                        .collect();
                    let on: Vec<JsonValue> = d
                        .painted_layers(&pick)
                        .iter()
                        .filter_map(|&i| specs.get(i).map(|s| json::string(&s.name)))
                        .collect();
                    let painted = match d.edited().and_then(|e| e.request).map(|r| r.shape) {
                        Some(kurven_core::surface_request::Shape::Periodic { expression, .. }) => expression,
                        _ => String::new(),
                    };
                    json::object([
                        ("face", json::string(&pick.face.to_string())),
                        ("copy", JsonValue::Int(pick.copy as i64)),
                        ("title", json::string(&d.paint_title(&pick))),
                        ("caption", json::string(&d.paint_caption(&pick))),
                        ("choices", JsonValue::Array(choices)),
                        ("layers", JsonValue::Array(on)),
                        ("takesFunction", JsonValue::Bool(d.offers_function(&pick))),
                        ("expression", json::string(&painted)),
                    ])
                }
            },
        );
        let stack = d.undo_stack();
        o.insert(
            "undo".to_string(),
            json::object([
                ("can", JsonValue::Bool(stack.can_undo())),
                ("name", stack.undo_name().map_or(JsonValue::Null, json::string)),
            ]),
        );
        o.insert(
            "redo".to_string(),
            json::object([
                ("can", JsonValue::Bool(stack.can_redo())),
                ("name", stack.redo_name().map_or(JsonValue::Null, json::string)),
            ]),
        );
        o.insert("current".to_string(), JsonValue::Bool(d.is_current()));
        o.insert("depthTolerance".to_string(), JsonValue::Double(d.depth_tolerance));
        o.insert("bakeResolution".to_string(), JsonValue::Int(d.bake_resolution));
        o.insert("draftResolution".to_string(), JsonValue::Int(d.draft_resolution));
        o.insert("canBake".to_string(), JsonValue::Bool(d.can_bake()));
        o.insert("refine".to_string(), JsonValue::Bool(d.refine_contours));
        o.insert(
            "mode".to_string(),
            json::string(match d.mode {
                PreviewMode::Drawing => "drawing",
                PreviewMode::Shaded(_) => "shaded",
                PreviewMode::Depth => "depth",
            }),
        );
        JsonValue::Object(o)
    }
}

/// A slider named in a request: `{"slider": "landscape", "label": "rMax"}`,
/// `{"slider": "levels", "name": ...}`, `strokes`, `light`, `truncation`, or
/// `{"slider": "surfaceField", "field": "major"}`.
fn slider(o: &JsonObject) -> Result<Slider, Refusal> {
    let s = o.get("control").and_then(JsonValue::as_object).ok_or("wants a 'control'")?;
    Ok(match text(s, "slider")? {
        "landscape" => Slider::Landscape { label: text(s, "label")?.to_string() },
        "levels" => Slider::Levels { name: text(s, "name")?.to_string() },
        "strokes" => Slider::Strokes { name: text(s, "name")?.to_string() },
        "light" => Slider::Light { name: text(s, "name")?.to_string() },
        "truncation" => Slider::Truncation,
        "surfaceField" => {
            let raw = text(s, "field")?;
            let field = Field::parse(raw).ok_or_else(|| refuse(format!("no field '{raw}'")))?;
            Slider::SurfaceField { field, label: s.get("label").and_then(JsonValue::as_str).unwrap_or(raw).to_string() }
        }
        other => return Err(refuse(format!("no slider '{other}'"))),
    })
}

// A layer a control names must be one that has the control. Swift's
// `paintLevels`, `paintStrokes`, `paintLight` and `paintResetLevels` pass
// over a name they do not find, saying nothing; its headless run refuses
// one first ("no layer ... with levels to drag"). The protocol refuses it
// as the headless run does, touching nothing, so a client does not journal
// an edit that did nothing (crates/FINDINGS.md, second review of PR #70).

/// `name` is a layer with levels to set (`--levels`'s test).
fn with_levels(d: &Document, name: &str) -> Result<(), Refusal> {
    let index = d.layer_specs().iter().position(|s| s.name == name);
    match index.and_then(|i| d.levels_for_layer(i)) {
        Some(_) => Ok(()),
        None => Err(refuse(format!("no layer '{name}' with levels to set"))),
    }
}

/// `name` is a hatching with strokes to set (`--strokes`'s test).
fn with_strokes(d: &Document, name: &str) -> Result<(), Refusal> {
    let index = d.edited().and_then(|e| e.layers.iter().position(|l| l.name == name));
    match index.and_then(|i| d.hatch_count(i)) {
        Some(_) => Ok(()),
        None => Err(refuse(format!("no hatching '{name}' with strokes to set"))),
    }
}

/// `name` is a layer read under a light (`--light`'s test).
fn with_light(d: &Document, name: &str) -> Result<(), Refusal> {
    let lit = d.edited().and_then(|e| e.layers.iter().find(|l| l.name == name).map(|l| l.source.light().is_some()));
    if lit == Some(true) {
        Ok(())
    } else {
        Err(refuse(format!("no layer '{name}' read under a light: an isophote or a half-Lambert tone")))
    }
}

/// A gesture as a request names it.
fn gesture(g: &JsonObject) -> Result<Gesture, Refusal> {
    Ok(match text(g, "kind")? {
        "orbit" => Gesture::Orbit(vec2(g, "delta")?),
        "pan" => Gesture::Pan(vec2(g, "delta")?),
        "zoom" => Gesture::Zoom { factor: number(g, "factor")?, at: vec2(g, "at")? },
        "roll" => Gesture::Roll(number(g, "radians")?),
        "retarget" => {
            let p = numbers(g, "point", 3)?;
            Gesture::Retarget(P3::new(p[0], p[1], p[2]))
        }
        other => return Err(refuse(format!("no gesture '{other}'"))),
    })
}

/// The most layers a link may carry: the catalog's landscapes have nine.
const LINK_LAYERS: usize = 64;
/// The most levels a contour layer of a link may list. Not a control's
/// reach: a layer lists the levels its landscape chose (forty for a minor
/// family of the catalog's), which a level slider's count resamples; this
/// bounds the work a link can ask for.
const LINK_LEVELS: usize = 1000;
/// The most seeds a link's slope lines may list, or be spaced to along
/// the window's longer side; a bound on work, as a strokes slider's top.
const LINK_SEEDS: f64 = 400.0;
/// The most bands a link's truncation may have; a bound on work (each is a
/// riser and its hatching), far past the plates' four.
const LINK_BANDS: usize = 64;

/// Whether a link asks for no more than the window's controls can: a link
/// is someone else's text, and the page samples what it says on open.
/// Each kind of layer a landscape's link can carry is held to a reach
/// (`sliders::reach`) or a stated bound on work; a kind with none -- a layer
/// stored as arrays, a parametric surface's or a polyhedron's -- is refused,
/// as are a window without area, a resolution or a level count past its
/// slider, and a truncation that is not finite. Refused, touching nothing,
/// saying which.
fn linkable(
    request: &LandscapeRequest,
    layers: &[LayerSpec],
    truncation: &Truncation,
    counts: &BTreeMap<String, i64>,
) -> Result<(), Refusal> {
    let (least, most, _) = reach::SAMPLES;
    if !(least..=most).contains(&request.resolution) {
        return Err(refuse(format!(
            "a link's landscape is sampled {least} to {most} across, as the Samples slider reaches, not {}",
            request.resolution
        )));
    }
    let d = request.domain;
    for (axis, i) in [("Re", d.real), ("Im", d.imag)] {
        if !(i.lo.is_finite() && i.hi.is_finite() && i.lo < i.hi) {
            return Err(refuse(format!("a link's window has {axis} from {} to {}, which is no interval", i.lo, i.hi)));
        }
    }
    let (least, most) = reach::LEVEL_COUNTS;
    for (name, &n) in counts {
        if !(least..=most).contains(&n) {
            return Err(refuse(format!("a link's {name} has {n} levels; a level slider reaches {least} to {most}")));
        }
    }
    let finite = |x: f64| x.is_finite();
    match truncation {
        Truncation::None => {}
        Truncation::Uniform(h) if finite(*h) => {}
        Truncation::RealBands { bands, beyond }
            if bands.len() <= LINK_BANDS
                && bands.iter().all(|b| finite(b.below) && finite(b.height))
                && !beyond.is_nan() => {}
        _ => return Err(refuse(format!("a link's truncation is not finite, or has more than {LINK_BANDS} bands"))),
    }
    if layers.len() > LINK_LAYERS {
        return Err(refuse(format!("a link carries {} layers, more than {LINK_LAYERS}", layers.len())));
    }
    let reach = d.real.length().max(d.imag.length());
    let strokes = reach::STROKES.1 as f64;
    for layer in layers {
        let name = &layer.name;
        let fine = match &layer.source {
            LayerSource::Contour { levels, light, .. } => {
                if levels.len() > LINK_LEVELS {
                    return Err(refuse(format!(
                        "a link's {name} lists {} levels, more than {LINK_LEVELS}",
                        levels.len()
                    )));
                }
                levels.iter().all(|&x| finite(x)) && light.is_none_or(crate::files::usable_light)
            }
            LayerSource::Hatching { hatching, .. } => {
                let s = hatching.spacing;
                if !(finite(s) && s > 0.0 && (reach / s).round() <= strokes) {
                    return Err(refuse(format!(
                        "a link's {name} is hatched at a spacing of {s}: more than {strokes} strokes across the \
                         window, as a strokes slider reaches"
                    )));
                }
                finite(hatching.angle)
            }
            LayerSource::WallEdges { edges, base } => {
                // A landscape's walls are its window's four sides.
                let mut seen = BTreeSet::new();
                finite(*base) && edges.len() <= 4 && edges.iter().all(|&e| (0..4).contains(&e) && seen.insert(e))
            }
            LayerSource::TruncationEdge { .. } | LayerSource::FoldCurves => true,
            LayerSource::IntegralCurves { seeds, .. } => {
                let many = match &seeds.spacing {
                    kurven_core::manifest::Spacing::Values(_, values) => {
                        values.len() as f64 > LINK_SEEDS || !values.iter().all(|&x| finite(x))
                    }
                    kurven_core::manifest::Spacing::ArcLength(s) => {
                        !(finite(*s) && *s > 0.0 && reach / s <= LINK_SEEDS)
                    }
                };
                if many {
                    return Err(refuse(format!(
                        "a link's {name} is seeded more finely than {LINK_SEEDS} seeds across the window"
                    )));
                }
                finite(seeds.level)
            }
            other => {
                return Err(refuse(format!("a landscape's link carries no {} layer, as {name} is", other.kind())));
            }
        };
        if !fine {
            return Err(refuse(format!("a link's {name} holds a number that is not finite, or an edge no wall has")));
        }
    }
    Ok(())
}

/// The open bundle's camera presets.
fn presets(d: &Document) -> Vec<kurven_core::manifest::CameraPreset> {
    if let Some(b) = d.bundle() {
        return b.manifest.presets.clone();
    }
    if let Some(b) = d.surface_bundle() {
        return b.manifest.presets.clone();
    }
    if let Some(b) = d.polyhedron_bundle() {
        return b.manifest.presets.clone();
    }
    Vec::new()
}

/// A light's azimuth round z and elevation over the xy plane, in degrees
/// (`LightSlider.angles(of:)`).
/// None for a light with no direction (zero, or too long to normalize),
/// which nothing that sets a light lets through (`files::usable_light`).
/// Swift's `simd_normalize` of one is NaN, and its sliders show NaN.
fn light_angles(light: Vec3) -> Option<(f64, f64)> {
    if !crate::files::usable_light(light) {
        return None;
    }
    let l = light.normalized();
    let degrees = 180.0 / core::f64::consts::PI;
    let angles = (
        kurven_math::real::atan2(l.y, l.x) * degrees,
        kurven_math::real::asin(kurven_math::swift::min(kurven_math::swift::max(l.z, -1.0), 1.0)) * degrees,
    );
    (angles.0.is_finite() && angles.1.is_finite()).then_some(angles)
}

/// A polyhedron as the Polyhedron section describes it (Inspector.swift's
/// `polyhedron(_:)`): what it is, its boundary, and its genus or shells.
fn polyhedron_json(bundle: &kurven_core::polyhedron_bundle::PolyhedronBundle) -> JsonValue {
    let solid = bundle.solid();
    let edges = solid.is_diagonal.iter().filter(|&&d| !d).count();
    let shape = match solid.genus() {
        Some(genus) => ("Genus", format!("{genus}{}", if solid.is_convex() { ", convex" } else { ", not convex" })),
        None => ("Shells", format!("{}, χ = {}", solid.shell_count(), solid.euler_characteristic())),
    };
    json::object([
        ("solid", json::string(&bundle.manifest.provenance.function)),
        (
            "boundary",
            json::string(&format!("{} vertices, {edges} edges, {} faces", solid.vertices.len(), solid.faces.len())),
        ),
        ("shapeLabel", json::string(shape.0)),
        ("shape", json::string(&shape.1)),
    ])
}

/// What is edited, as `edited` sends it and `restore` reads it.
fn edited_json(e: &Edited) -> JsonValue {
    json::object([
        ("landscape", e.landscape.as_ref().map_or(JsonValue::Null, LandscapeRequest::json)),
        ("layers", JsonValue::Array(e.layers.iter().map(LayerSpec::json).collect())),
        ("truncation", e.truncation.as_ref().map_or(JsonValue::Null, Truncation::json)),
        (
            "levelCounts",
            JsonValue::Object(e.level_counts.iter().map(|(k, v)| (k.clone(), JsonValue::Int(*v))).collect()),
        ),
    ])
}

/// Judged strokes as the page draws them: each layer's page-space points
/// and offsets pushed as arrays, and its style.
fn stroke_arrays(strokes: &kurven_core::svg::Strokes, arrays: &mut Vec<(String, Array)>) -> Vec<JsonValue> {
    strokes
        .layers
        .iter()
        .enumerate()
        .map(|(i, (style, paths))| {
            let points = paths.vertices.iter().flat_map(|p| [p.x(), p.y()]);
            let offsets = paths.offsets.iter().map(|&o| o as u32);
            arrays.push((format!("stroke.{i}.points"), Array::f64(points)));
            arrays.push((format!("stroke.{i}.offsets"), Array::u32(offsets)));
            json::object([("width", JsonValue::Double(style.width)), ("color", json::string(&style.color))])
        })
        .collect()
}

/// A pixel grid as the page reads it: `DepthFrame::index`'s terms.
fn frame_json(f: &DepthFrame) -> JsonValue {
    json::object([
        ("axis0", json::doubles(&[f.axis0.lo, f.axis0.hi])),
        ("axis1", json::doubles(&[f.axis1.lo, f.axis1.hi])),
        ("rows", JsonValue::Int(f.rows)),
        ("cols", JsonValue::Int(f.cols)),
        ("nudge", JsonValue::Double(f.nudge)),
    ])
}

/// A number, or null for Swift's infinity.
fn finite(x: Option<f64>) -> JsonValue {
    x.map_or(JsonValue::Null, JsonValue::Double)
}

fn floats(xs: &[f32]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|&x| JsonValue::Double(f64::from(x))).collect())
}

/// A frame of the preview as JSON, its arrays pushed beside it, single
/// precision, as the GPU takes them.
fn preview_json(p: &Preview, arrays: &mut Vec<(String, Array)>) -> JsonObject {
    let f32s = |xs: &mut dyn Iterator<Item = f64>| Array::f32(xs.map(|x| x as f32).collect::<Vec<_>>());
    let resources = match &p.resources {
        None => JsonValue::Null,
        Some(Resources::Heightfield {
            heights,
            grid_width,
            grid_height,
            truncation_baked,
            step,
            decimated_width,
            decimated_height,
            domain_lo,
            domain_size,
            tiles,
            region,
            walls,
        }) => {
            arrays.push(("resource.heights".to_string(), Array::f32(heights.values.iter().copied())));
            arrays.push(("resource.tiles".to_string(), f32s(&mut tiles.iter().flatten().copied())));
            arrays.push(("resource.region".to_string(), f32s(&mut region.iter().flatten().copied())));
            arrays.push(("resource.walls".to_string(), f32s(&mut walls.iter().flatten().copied())));
            json::object([
                ("kind", json::string("heightfield")),
                ("gridSize", json::ints(&[*grid_width as i64, *grid_height as i64])),
                ("grid", json::ints(&[*decimated_width as i64, *decimated_height as i64])),
                ("domainLo", json::doubles(domain_lo)),
                ("domainSize", json::doubles(domain_size)),
                ("truncationBaked", JsonValue::Bool(*truncation_baked)),
                ("step", JsonValue::Int(*step as i64)),
            ])
        }
        Some(Resources::Solid { walls }) => {
            arrays.push(("resource.walls".to_string(), f32s(&mut walls.iter().flatten().copied())));
            json::object([("kind", json::string("solid"))])
        }
        Some(Resources::Parametric { positions, samples, cells, lo, spacing }) => {
            arrays.push(("resource.positions".to_string(), f32s(&mut positions.iter().flatten().copied())));
            json::object([
                ("kind", json::string("parametric")),
                ("samples", json::ints(&[samples[0] as i64, samples[1] as i64])),
                ("cells", json::ints(&[cells[0] as i64, cells[1] as i64])),
                ("lo", json::doubles(lo)),
                ("spacing", json::doubles(spacing)),
            ])
        }
    };
    let u = &p.uniforms;
    let flat = |m: &[[f64; 4]; 4]| json::doubles(&m.iter().flatten().copied().collect::<Vec<_>>());
    let uniforms = json::object([
        ("view", flat(&u.view)),
        ("clip", flat(&u.clip)),
        ("truncationHeight", finite(u.truncation_height)),
        ("frame", frame_json(&u.frame)),
        ("depthTolerance", JsonValue::Double(u.depth_tolerance)),
        ("lightDirection", floats(&u.light_direction)),
        ("ambient", JsonValue::Double(f64::from(u.ambient))),
        ("depthRange", json::doubles(&u.depth_range)),
        ("surfaceDepthRange", u.surface_depth_range.map_or(JsonValue::Null, |r| json::doubles(&r))),
        ("sight", json::doubles(&u.sight)),
        ("eye", json::doubles(&u.eye)),
        ("perspective", JsonValue::Bool(u.perspective)),
    ]);
    let layers = p
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (pipeline, on_folds) = match l.pipeline {
                Pipeline::Stroke => ("stroke", false),
                Pipeline::Surface { on_folds } => ("surface", on_folds),
            };
            if let Some(g) = &l.segments {
                arrays.push((format!("layer.{i}.positions"), Array::f32(g.positions.iter().copied())));
                arrays.push((format!("layer.{i}.normals"), Array::f32(g.normals.iter().copied())));
                if !g.coords.is_empty() {
                    arrays.push((format!("layer.{i}.coords"), Array::f32(g.coords.iter().copied())));
                }
            }
            json::object([
                ("name", json::string(&l.name)),
                ("key", json::string(&l.key)),
                ("visible", JsonValue::Bool(l.visible)),
                ("pipeline", json::string(pipeline)),
                ("onFolds", JsonValue::Bool(on_folds)),
                ("color", floats(&l.color)),
                ("width", JsonValue::Double(l.width)),
                ("depthTolerance", finite(l.depth_tolerance)),
                ("refusal", l.refusal.as_deref().map_or(JsonValue::Null, json::string)),
                ("segments", l.segments.as_ref().map_or(JsonValue::Null, |g| JsonValue::Int(g.count() as i64))),
            ])
        })
        .collect();
    let mut out = JsonObject::new();
    out.insert("content".to_string(), json::string(&p.content));
    out.insert("resources".to_string(), resources);
    out.insert("uniforms".to_string(), uniforms);
    out.insert(
        "mode".to_string(),
        json::string(match p.mode {
            PreviewMode::Drawing => "drawing",
            PreviewMode::Shaded(_) => "shaded",
            PreviewMode::Depth => "depth",
        }),
    );
    out.insert("layers".to_string(), JsonValue::Array(layers));
    out
}
