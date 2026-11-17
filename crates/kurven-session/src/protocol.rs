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

use std::collections::BTreeMap;

use kurven_core::bundle::BundleFiles;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::{DomainAxis, LayerSource, Truncation};
use kurven_core::navigation::{Gesture, Viewport};
use kurven_core::polyline::Layer;
use kurven_core::scene::PreviewMode;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::spaces::P3;
use kurven_core::surface_request::{Field, surface_key};
use kurven_core::svg;
use kurven_landscape::catalog::Catalog;
use kurven_surfaces::drawing::SurfacePreset;

use crate::document::{Document, State};
use crate::headless::{self, Output};
use crate::landscape::applying_overrides;
use crate::sliders::Slider;

/// A typed array a reply carries.
#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    F64(Vec<f64>),
    F32(Vec<f32>),
    U32(Vec<u32>),
    U8(Vec<u8>),
}

impl Array {
    pub fn type_name(&self) -> &'static str {
        match self {
            Array::F64(_) => "f64",
            Array::F32(_) => "f32",
            Array::U32(_) => "u32",
            Array::U8(_) => "u8",
        }
    }

    /// Its bytes, little-endian, as the page's typed array holds them.
    pub fn bytes(&self) -> Vec<u8> {
        match self {
            Array::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Array::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Array::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Array::U8(v) => v.clone(),
        }
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
}

struct Refusal(String);

impl From<String> for Refusal {
    fn from(s: String) -> Refusal {
        Refusal(s)
    }
}

impl From<&str> for Refusal {
    fn from(s: &str) -> Refusal {
        Refusal(s.to_string())
    }
}

type Answer = Result<(JsonObject, Vec<(String, Array)>), Refusal>;

fn number(o: &JsonObject, key: &str) -> Result<f64, Refusal> {
    o.get(key).and_then(JsonValue::as_f64).ok_or_else(|| Refusal(format!("wants a number '{key}'")))
}

fn text<'a>(o: &'a JsonObject, key: &str) -> Result<&'a str, Refusal> {
    o.get(key).and_then(JsonValue::as_str).ok_or_else(|| Refusal(format!("wants a string '{key}'")))
}

fn numbers(o: &JsonObject, key: &str, n: usize) -> Result<Vec<f64>, Refusal> {
    let list = o.get(key).and_then(JsonValue::as_array).ok_or_else(|| Refusal(format!("wants '{key}'")))?;
    let values: Vec<f64> = list.iter().filter_map(JsonValue::as_f64).collect();
    if values.len() != n || list.len() != n {
        return Err(Refusal(format!("'{key}' wants {n} numbers")));
    }
    Ok(values)
}

fn vec2(o: &JsonObject, key: &str) -> Result<Vec2, Refusal> {
    let v = numbers(o, key, 2)?;
    Ok(Vec2::new(v[0], v[1]))
}

fn layer_arrays(prefix: &str, layers: &[Layer], arrays: &mut Vec<(String, Array)>) -> Vec<JsonValue> {
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let vertices: Vec<f32> =
                layer.paths.vertices.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]).collect();
            let offsets: Vec<u32> = layer.paths.offsets.iter().map(|&o| o as u32).collect();
            arrays.push((format!("{prefix}.{i}.vertices"), Array::F32(vertices)));
            arrays.push((format!("{prefix}.{i}.offsets"), Array::U32(offsets)));
            json::object([
                ("name", json::string(&layer.spec.name)),
                ("width", JsonValue::Double(layer.spec.width)),
                ("color", json::string(&layer.spec.color)),
                ("clipped", JsonValue::Bool(layer.spec.clipped)),
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
        self.staged.entry(bundle.to_string()).or_default().insert(path.to_string(), bytes);
    }

    /// Forget every staged bundle.
    pub fn unstage(&mut self) {
        self.staged.clear();
    }

    /// Answer one request.
    pub fn call(&mut self, request: &str) -> Reply {
        let answer = match json::parse(request) {
            Ok(JsonValue::Object(o)) => self.answer(&o),
            Ok(_) => Err(Refusal("a request is a JSON object".to_string())),
            Err(error) => Err(Refusal(format!("not JSON: {error}"))),
        };
        let headless =
            matches!(json::parse(request), Ok(ref v) if v.get("kind").and_then(JsonValue::as_str) == Some("headless"));
        let (mut o, arrays) = match answer {
            Ok((mut o, arrays)) => {
                o.insert("ok".to_string(), JsonValue::Bool(true));
                (o, arrays)
            }
            Err(Refusal(error)) => {
                let mut o = JsonObject::new();
                o.insert("ok".to_string(), JsonValue::Bool(false));
                o.insert("error".to_string(), JsonValue::String(error));
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
            "open" => {
                let name = text(o, "name")?.to_string();
                let files = self.staged.get(&name).ok_or_else(|| Refusal(format!("no bundle '{name}' staged")))?;
                let files = files.clone();
                let d = &mut self.document;
                d.load(&name, &files);
                d.settle();
                if let State::Failed(_, why) = d.state() {
                    return Err(Refusal(why.clone()));
                }
                none()
            }
            "create" => {
                let name = text(o, "preset")?;
                let preset =
                    Catalog::native().preset(name).ok_or_else(|| Refusal(format!("no preset '{name}'")))?.clone();
                d.create(&preset);
                d.settle();
                none()
            }
            "createSurface" => {
                let name = text(o, "preset")?;
                let preset = SurfacePreset::named(name).ok_or_else(|| Refusal(format!("no surface '{name}'")))?;
                d.create_surface(&preset);
                d.settle();
                none()
            }
            "expression" => {
                // The formula field: a new expression, the window kept.
                let expression = text(o, "expression")?.to_string();
                kurven_math::expression::Expression::compile(&expression).map_err(|e| Refusal(e.to_string()))?;
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
                        .ok_or_else(|| Refusal(format!("the surface has no field '{key}'")))?;
                    surface.setting(field, value, &surface_key).map_err(|e| Refusal(e.to_string()))?;
                    Slider::SurfaceField { field, label: key }.set(d, value);
                } else if d.landscape_request().is_some() {
                    let slider = Slider::Landscape { label: key.clone() };
                    slider.set_landscape(d, |r| *r = applying_overrides(r, &[(key.as_str(), value)]));
                } else {
                    return Err("this bundle has no request to edit".into());
                }
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(Refusal(why)),
                    None => none(),
                }
            }
            "drag" => {
                // A slider taken hold of, moved, let go: one undo step.
                let slider = slider(o)?;
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
                        _ => slider.set(d, number(o, "value")?),
                    },
                    other => return Err(Refusal(format!("a drag's phase is begin, move or end, not {other}"))),
                }
                d.settle();
                none()
            }
            "levels" => {
                let name = text(o, "layer")?;
                let count = number(o, "count")?;
                Slider::Levels { name: name.to_string() }.set(d, count);
                d.settle();
                none()
            }
            "resetLevels" => {
                d.paint_reset_levels(text(o, "layer")?);
                d.settle();
                none()
            }
            "strokes" => {
                let name = text(o, "layer")?;
                let count = number(o, "count")?;
                d.paint_status = None;
                Slider::Strokes { name: name.to_string() }.set(d, count);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(Refusal(why)),
                    None => none(),
                }
            }
            "relief" => {
                let value = number(o, "value")?;
                d.paint_status = None;
                Slider::SurfaceField { field: Field::Relief, label: "Relief".to_string() }.set(d, value);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(Refusal(why)),
                    None => none(),
                }
            }
            "truncation" => {
                let truncation = Truncation::from_json(o.get("truncation").ok_or("wants a 'truncation'")?)
                    .map_err(|e| Refusal(e.to_string()))?;
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
                    .ok_or_else(|| Refusal(format!("nothing called '{name}' to add here")))?;
                d.add(&choice);
                d.settle();
                match d.paint_status.clone() {
                    Some(why) => Err(Refusal(why)),
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
                    .ok_or_else(|| Refusal(format!("no layer '{name}' on the picked face")))?;
                d.remove_layer(index).map_err(|e| Refusal(e.0))?;
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
                d.add_function(&expression, periods).map_err(|e| Refusal(e.0))?;
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
                    other => return Err(Refusal(format!("an axis is real or imag, not {other}"))),
                };
                d.paint_axis(axis);
                d.settle();
                none()
            }
            "light" => {
                let l = numbers(o, "direction", 3)?;
                d.paint_light(Vec3::new(l[0], l[1], l[2]), text(o, "layer")?);
                d.settle();
                none()
            }
            "hide" => {
                d.toggle(number(o, "layer")? as usize);
                none()
            }
            "gesture" => {
                let g = o.get("gesture").and_then(JsonValue::as_object).ok_or("wants a 'gesture'")?;
                let gesture = match text(g, "kind")? {
                    "orbit" => Gesture::Orbit(vec2(g, "delta")?),
                    "pan" => Gesture::Pan(vec2(g, "delta")?),
                    "zoom" => Gesture::Zoom { factor: number(g, "factor")?, at: vec2(g, "at")? },
                    "roll" => Gesture::Roll(number(g, "radians")?),
                    "retarget" => {
                        let p = numbers(g, "point", 3)?;
                        Gesture::Retarget(P3::new(p[0], p[1], p[2]))
                    }
                    other => return Err(Refusal(format!("no gesture '{other}'"))),
                };
                d.apply(&gesture);
                none()
            }
            "fit" => {
                d.fit();
                none()
            }
            "preset" => {
                let presets = presets(d);
                let index = number(o, "index")? as usize;
                let preset = presets.get(index).cloned().ok_or("no such preset")?;
                d.use_preset(&preset);
                none()
            }
            "perspective" => {
                d.set_perspective(o.get("on").and_then(JsonValue::as_bool).unwrap_or(false));
                none()
            }
            "viewport" => {
                d.viewport = Viewport::new(number(o, "width")? as i64, number(o, "height")? as i64);
                none()
            }
            "mode" => {
                d.mode = match text(o, "mode")? {
                    "drawing" => PreviewMode::Drawing,
                    "depth" => PreviewMode::Depth,
                    "shaded" => PreviewMode::Shaded(kurven_core::scene::Lighting::default()),
                    other => return Err(Refusal(format!("no mode '{other}'"))),
                };
                none()
            }
            "refine" => {
                d.set_refinement(o.get("on").and_then(JsonValue::as_bool).unwrap_or(true));
                d.settle();
                none()
            }
            "draftResolution" => {
                d.draft_resolution = (number(o, "value")? as i64).max(8);
                none()
            }
            "bakeResolution" => {
                d.bake_resolution = number(o, "value")? as i64;
                none()
            }
            "frame" => {
                // The curves a frame draws, in world space: a draft's, or the
                // window's layout at rest.
                let draft = o.get("draft").and_then(JsonValue::as_bool).unwrap_or(true);
                let layers = if draft { d.draft_layers() } else { d.rest_layout() };
                let mut arrays = Vec::new();
                let described = layer_arrays("layer", &layers, &mut arrays);
                let mut out = JsonObject::new();
                out.insert("layers".to_string(), JsonValue::Array(described));
                Ok((out, arrays))
            }
            "bake" => {
                // The judged strokes, what Export writes: each layer's
                // page-space points and its offsets, and the SVG when asked.
                let strokes = d.bake().map_err(Refusal)?;
                let mut arrays = Vec::new();
                let mut styles = Vec::new();
                for (i, (style, paths)) in strokes.layers.iter().enumerate() {
                    let points: Vec<f64> = paths.vertices.iter().flat_map(|p| [p.x(), p.y()]).collect();
                    let offsets: Vec<u32> = paths.offsets.iter().map(|&o| o as u32).collect();
                    arrays.push((format!("stroke.{i}.points"), Array::F64(points)));
                    arrays.push((format!("stroke.{i}.offsets"), Array::U32(offsets)));
                    styles.push(json::object([
                        ("width", JsonValue::Double(style.width)),
                        ("color", json::string(&style.color)),
                    ]));
                }
                if o.get("svg").and_then(JsonValue::as_bool) == Some(true) {
                    let text = svg::render(&strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS);
                    arrays.push(("svg".to_string(), Array::U8(text.into_bytes())));
                }
                let mut out = JsonObject::new();
                out.insert("strokes".to_string(), JsonValue::Array(styles));
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
                    arrays.push(("height".to_string(), Array::F32(h.values.clone())));
                    if let Some(p) = &b.surface.phase {
                        arrays.push(("phase".to_string(), Array::F32(p.values.clone())));
                    }
                } else if let Some(s) = d.scene().and_then(|s| s.parametric()) {
                    out.insert("shape".to_string(), json::ints(&[s.v.samples as i64, s.u.samples as i64]));
                    let p: Vec<f32> =
                        s.positions.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]).collect();
                    arrays.push(("positions".to_string(), Array::F32(p)));
                } else if let Some(solid) = d.scene().and_then(|s| s.solid()) {
                    let mesh = solid.mesh();
                    let v: Vec<f32> =
                        mesh.vertices.iter().flat_map(|v| [v.x() as f32, v.y() as f32, v.z() as f32]).collect();
                    let t: Vec<u32> = mesh.triangles.iter().flat_map(|t| t.map(|k| k as u32)).collect();
                    arrays.push(("vertices".to_string(), Array::F32(v)));
                    arrays.push(("triangles".to_string(), Array::U32(t)));
                }
                Ok((out, arrays))
            }
            "save" => {
                let name = text(o, "name")?;
                let files = d.save_bundle(name).ok_or("nothing to save")?;
                let mut arrays = Vec::new();
                let mut names = Vec::new();
                for (path, bytes) in files {
                    names.push(json::string(&path));
                    arrays.push((format!("file:{path}"), Array::U8(bytes)));
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
                    staged.get(path).cloned().ok_or_else(|| format!("{path} is not a bundle directory"))
                });
                let mut arrays = Vec::new();
                let mut outputs = Vec::new();
                for output in &run.outputs {
                    match output {
                        Output::Bundle { path, files } => {
                            for (rel, bytes) in files {
                                arrays.push((format!("{path}/{rel}"), Array::U8(bytes.clone())));
                            }
                            outputs.push(json::string(path));
                        }
                        Output::File { path, bytes } => {
                            arrays.push((path.clone(), Array::U8(bytes.clone())));
                            outputs.push(json::string(path));
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
            other => Err(Refusal(format!("no request '{other}'"))),
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
                    json::object([
                        ("face", json::string(&pick.face.to_string())),
                        ("copy", JsonValue::Int(pick.copy as i64)),
                        ("title", json::string(&d.paint_title(&pick))),
                        ("choices", JsonValue::Array(choices)),
                        ("layers", JsonValue::Array(on)),
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
            let field = Field::parse(raw).ok_or_else(|| Refusal(format!("no field '{raw}'")))?;
            Slider::SurfaceField { field, label: s.get("label").and_then(JsonValue::as_str).unwrap_or(raw).to_string() }
        }
        other => return Err(Refusal(format!("no slider '{other}'"))),
    })
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
