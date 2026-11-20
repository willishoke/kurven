//! The document's side of a pool of engines: several engines holding the
//! same document (the page's Workers, each replaying the one journal; or a
//! desktop host's threads), each doing a share of the slow work, one of them
//! putting the shares together. Nothing here knows what the engines are or
//! how the shares travel: a share goes out as a request and comes back as
//! JSON and typed arrays, as every reply does (`protocol.rs`).
//!
//! Two kinds of slow work are shared:
//!
//! - **The judged strokes** (`Document::bake`), what Export writes:
//!   `bakePlan` splits the bake into pieces (`kurven_bake::pieces`, along the
//!   seam Swift's threads take, `Heightfield.visibleRuns`' items, gathered at
//!   path boundaries), `bakePiece` bakes one, and `bakeAssemble` puts their
//!   strokes together in the plan's order and answers as `bake` does, the SVG
//!   the same bytes.
//! - **The frame at rest** (`Document::preview` with `draft` false), whose
//!   cost is its tested layers (a landscape's folds on f and its wall curves,
//!   a sphere-traced surface's strokes): `restLayers` names the layers a frame
//!   at rest would lay out, `restLayout` lays some out as `preview::lay_out`
//!   does, and `preview` given those layouts serves the frame under the
//!   serving engine's own keys without laying them out again.
//!
//! A share computed by an engine holding another document would be wrong
//! without failing, so every share carries the fingerprint of the document
//! it was asked of (`Document::layout_fingerprint` for the frame at rest,
//! `bake_fingerprint` for the bake), and an engine holding another refuses
//! it as such (`Refused::another`), which tells a pool to bring that engine
//! to its document again. An assembly names its plan's fingerprint too, and
//! one put together after the document moved on says so (`snapshot`).

use std::collections::{BTreeMap, BTreeSet};

use kurven_bake::BakeOptions;
use kurven_bake::pieces::{self, Baked, Piece};
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::CameraPreset;
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::spaces::P3;
use kurven_core::svg::{self, Strokes, Style};

use crate::document::Document;
use crate::preview::{Laid, content_key, curves_key, fnv, is_tested_exactly, lay_out, rest_key};
use crate::protocol::Array;

/// About how many vertices of a landscape's layer one piece of a bake tests:
/// about a quarter of a second of cn's contours natively, so cn's bake is
/// some eighty pieces for a pool to share out as each engine comes free
/// (Swift hands its threads chunks as they come free too), and the longest
/// piece is the layer that does not split, its fold curves.
pub const VERTICES_PER_PIECE: usize = 1024;

/// A reply's JSON and arrays, or why not.
pub type Shared = Result<(JsonObject, Vec<(String, Array)>), Refused>;

/// Why a share was refused, and whether because it was asked of another
/// document than this engine holds (`another`): the asking engine's
/// fingerprint is not this one's. A pool then brings the engine to its
/// document again and asks once more; any other refusal is the work's own.
#[derive(Debug, Clone, PartialEq)]
pub struct Refused {
    pub message: String,
    pub another: bool,
}

impl From<String> for Refused {
    fn from(message: String) -> Refused {
        Refused { message, another: false }
    }
}

impl From<&str> for Refused {
    fn from(message: &str) -> Refused {
        Refused { message: message.to_string(), another: false }
    }
}

/// Refused unless `o`'s `fingerprint` is `ours`: a share asked of another
/// document.
fn of_this_document(o: &JsonObject, ours: Option<String>, what: &str) -> Result<(), Refused> {
    let theirs = o.get("fingerprint").and_then(JsonValue::as_str);
    if theirs.is_none() || theirs.map(str::to_string) != ours {
        return Err(Refused { message: format!("{what} is of another document"), another: true });
    }
    Ok(())
}

impl Document {
    /// The scene a bake draws and how, refused as `Document::bake` refuses
    /// before it bakes -- `bake` asks this too, so there is one rule: the
    /// message, and whether it is said as the bake's status.
    fn bake_inputs(&self) -> Result<(Scene, BakeOptions), (String, bool)> {
        let Some(scene) = self.framed_scene() else { return Err(("nothing to bake".to_string(), false)) };
        if let Some(why) = CameraPreset::unsupported_buffer(self.bake_resolution) {
            return Err((format!("the resolution {why}"), true));
        }
        if !self.can_bake() {
            return Err((
                "switch off perspective to bake: the plates are orthographic, and a perspective bake has no oracle"
                    .to_string(),
                true,
            ));
        }
        let mut options = BakeOptions::new(self.bake_resolution);
        options.depth_tolerance = Some(self.depth_tolerance);
        Ok((scene, options))
    }

    /// `bake_inputs`, a refusal said in the bake's status as `bake` says it.
    pub(crate) fn bake_inputs_said(&mut self) -> Result<(Scene, BakeOptions), String> {
        self.bake_inputs().map_err(|(why, said)| {
            if said {
                self.bake_status = Some(why.clone());
            }
            why
        })
    }

    /// The pieces the bake of the framed scene splits into
    /// (`kurven_bake::pieces::plan`), or the refusal `bake` would give, said
    /// in the bake's status as `bake` says it.
    pub fn bake_plan(&mut self) -> Result<Vec<Piece>, String> {
        let (scene, options) = self.bake_inputs_said()?;
        pieces::plan(&scene, &options, VERTICES_PER_PIECE)
            .map_err(|error| self.bake_failed(format!("bake failed: {error}")))
    }

    /// One piece's strokes. Says nothing in the document: the engine that
    /// assembles the bake says what came of it.
    pub fn bake_piece(&self, piece: Piece) -> Result<Baked, String> {
        let (scene, options) = self.bake_inputs().map_err(|(why, _)| why)?;
        pieces::bake_piece(&scene, &options, piece).map_err(|error| format!("bake failed: {error}"))
    }

    /// A bake that failed in one of its pieces, said in the bake's status.
    pub fn bake_failed(&mut self, status: String) -> String {
        self.bake_status = Some(status.clone());
        status
    }

    /// The tested layers a frame at rest would lay out (`preview` with
    /// `draft` false, holding `have`), by their place in the shown scene:
    /// each shown layer whose layout is the camera's (`is_tested_exactly`)
    /// and whose key at rest (`preview::rest_key`, the key `preview` sends
    /// it under) the page does not hold. Keys alone: nothing is laid out,
    /// built or remembered.
    pub fn rest_layers(&self, nonce: &str, have: &BTreeSet<String>) -> Vec<usize> {
        let Some(scene) = self.shown_scene() else { return Vec::new() };
        let shown = self.shown_visible_layers();
        let curves = curves_key(&content_key(nonce, &scene), &scene);
        (0..scene.layers().len())
            .filter(|&i| {
                shown.contains(&i)
                    && is_tested_exactly(&scene.layers()[i], &scene)
                    && !have.contains(&rest_key(&scene, &curves, i, have))
            })
            .collect()
    }

    /// The layouts at rest of the shown scene's layers `layers`, each as
    /// `lay_out` lays it out; a place the scene has no tested layer at is
    /// refused.
    pub fn rest_laid(&self, layers: &[usize]) -> Result<Vec<(usize, Laid)>, String> {
        let scene = self.shown_scene().ok_or("nothing shown")?;
        layers
            .iter()
            .map(|&i| {
                let layer = scene.layers().get(i).ok_or_else(|| format!("no layer {i} shown"))?;
                if !is_tested_exactly(layer, &scene) {
                    return Err(format!("layer {i} ({}) is not tested for the camera", layer.spec.name));
                }
                Ok((i, lay_out(layer, &scene, false)))
            })
            .collect()
    }

    /// What a frame at rest's layouts are a function of, hashed: the shown
    /// scene's camera, its depth tolerance and the resolution its runs' ends
    /// are placed to, every layer's curves, and the surface they are tested
    /// against (a landscape's grid and window, a parametric surface's
    /// positions, a solid's vertices). Two engines holding the same document
    /// give the same; a layout given with another is of another document.
    pub fn layout_fingerprint(&self) -> Option<String> {
        let scene = self.shown_scene()?;
        Some(self.fingerprinted(0, &scene, &[]))
    }

    /// What the bake is a function of, hashed likewise: the framed scene (no
    /// picked face's boundary), the depth tolerance it is tested at and its
    /// resolution. A piece or an assembly asked with another is of another
    /// document.
    pub fn bake_fingerprint(&self) -> Option<String> {
        let scene = self.framed_scene()?;
        Some(self.fingerprinted(1, &scene, &[self.bake_resolution as u64, self.depth_tolerance.to_bits()]))
    }

    /// `scene`'s fingerprint with `extra`, remembered in slot `slot` for what
    /// it was hashed for: the scene's content and curves (counted per
    /// process, and new whenever its geometry or curves are), its camera,
    /// depth tolerance and resolution, and `extra`. A pool asks for it every
    /// rest and every piece; hashing cn's grid and curves each time was the
    /// cost of the ask.
    fn fingerprinted(&self, slot: usize, scene: &Scene, extra: &[u64]) -> String {
        let mut key = vec![scene.content().raw(), scene.curves().raw()];
        key.extend(scene.camera.view.m.columns.iter().flat_map(|c| c.to_array()).map(f64::to_bits));
        key.push(u64::from(scene.camera.is_perspective()));
        key.extend([scene.depth_tolerance.to_bits(), scene.resolution.to_bits()]);
        key.extend_from_slice(extra);
        let mut memo = self.preview_memo.fingerprints.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((k, f)) = &memo[slot]
            && *k == key
        {
            return f.clone();
        }
        let f = fingerprint(scene, extra);
        memo[slot] = Some((key, f.clone()));
        f
    }
}

/// A scene's fingerprint, hashed afresh.
fn fingerprint(scene: &Scene, extra: &[u64]) -> String {
    let mut words: Vec<u64> = Vec::new();
    words.extend(scene.camera.view.m.columns.iter().flat_map(|c| c.to_array()).map(f64::to_bits));
    words.push(u64::from(scene.camera.is_perspective()));
    words.push(scene.depth_tolerance.to_bits());
    words.push(scene.resolution.to_bits());
    words.extend_from_slice(extra);
    for layer in scene.layers() {
        words.push(fnv(layer.spec.name.bytes().map(u64::from)));
        words.push(u64::from(layer.spec.clipped));
        words.push(layer.spec.width.to_bits());
        words.push(fnv(layer.spec.color.bytes().map(u64::from)));
        words.push(layer.paths.vertices.len() as u64);
        words.extend(layer.paths.offsets.iter().map(|&o| o as u64));
        words.extend(layer.paths.vertices.iter().flat_map(|p| [p.x(), p.y(), p.z()]).map(f64::to_bits));
    }
    match scene.geometry() {
        SceneGeometry::Heightfield(h) => {
            let g = &*h.surface.height;
            words.extend([g.width as u64, g.height as u64]);
            words.extend([g.domain.real.lo, g.domain.real.hi, g.domain.imag.lo, g.domain.imag.hi].map(f64::to_bits));
            words.extend(g.values.iter().map(|&v| u64::from(v.to_bits())));
            words.extend(h.occluder.vertices.iter().flat_map(|p| [p.x(), p.y(), p.z()]).map(f64::to_bits));
        }
        SceneGeometry::Parametric(s) => {
            words.extend(s.positions.iter().flat_map(|p| [p.x(), p.y(), p.z()]).map(f64::to_bits));
        }
        SceneGeometry::Solid(s) => {
            words.extend(s.vertices.iter().flat_map(|p| [p.x(), p.y(), p.z()]).map(f64::to_bits));
        }
    }
    format!("{:016x}", fnv(words))
}

// MARK: - the shares as messages

/// A piece as a request names it: `{"whole": true}`, `{"layer": i}`, or
/// `{"layer": i, "from": a, "to": b}`; with `weight`, how many vertices it
/// tests, for sharing the pieces out (a whole layer derived for the camera
/// weighs what its scene's largest layer does, its curves being unknown
/// until derived).
pub fn piece_json(piece: &Piece, weight: usize) -> JsonValue {
    let mut o = JsonObject::new();
    match *piece {
        Piece::Whole => {
            o.insert("whole".to_string(), JsonValue::Bool(true));
        }
        Piece::Layer(i) => {
            o.insert("layer".to_string(), JsonValue::Int(i as i64));
        }
        Piece::Paths { layer, from, to } => {
            o.insert("layer".to_string(), JsonValue::Int(layer as i64));
            o.insert("from".to_string(), JsonValue::Int(from as i64));
            o.insert("to".to_string(), JsonValue::Int(to as i64));
        }
    }
    o.insert("weight".to_string(), JsonValue::Int(weight as i64));
    JsonValue::Object(o)
}

/// A piece from a request's JSON.
pub fn piece_of(v: &JsonValue) -> Result<Piece, String> {
    let index = |key: &str| -> Result<Option<usize>, String> {
        match v.get(key) {
            None => Ok(None),
            Some(x) => x
                .as_i64()
                .and_then(|i| usize::try_from(i).ok())
                .map(Some)
                .ok_or_else(|| format!("a piece's '{key}' is an index")),
        }
    };
    if v.get("whole").and_then(JsonValue::as_bool) == Some(true) {
        return Ok(Piece::Whole);
    }
    match (index("layer")?, index("from")?, index("to")?) {
        (Some(layer), Some(from), Some(to)) => Ok(Piece::Paths { layer, from, to }),
        (Some(layer), None, None) => Ok(Piece::Layer(layer)),
        _ => Err("a piece is {whole}, {layer} or {layer, from, to}".to_string()),
    }
}

/// The plan, each piece with its weight.
pub fn bake_plan(d: &mut Document) -> Shared {
    let plan = d.bake_plan()?;
    let scene = d.framed_scene().ok_or("nothing to bake")?;
    let largest = scene.layers().iter().map(|l| l.paths.vertices.len()).max().unwrap_or(0);
    let weight = |p: &Piece| match *p {
        Piece::Paths { layer, from, to } => {
            let o = &scene.layers()[layer].paths.offsets;
            o[to] - o[from]
        }
        Piece::Layer(i) => {
            let l = &scene.layers()[i];
            if l.spec.clipped { l.paths.vertices.len().max(largest) } else { 0 }
        }
        Piece::Whole => largest,
    };
    let mut out = JsonObject::new();
    out.insert("pieces".to_string(), JsonValue::Array(plan.iter().map(|p| piece_json(p, weight(p))).collect()));
    out.insert("fingerprint".to_string(), d.bake_fingerprint().map_or(JsonValue::Null, |f| json::string(&f)));
    out.insert("frame".to_string(), frame_of(d));
    Ok((out, Vec::new()))
}

/// The frame that maps a page point to the viewport's pixel, as the
/// preview's does: what `bake` answers with.
fn frame_of(d: &Document) -> JsonValue {
    d.navigator.map(|n| n.framing.frame(d.viewport)).as_ref().map_or(JsonValue::Null, crate::protocol::frame_json)
}

/// One piece's strokes: each stroke layer's colour in `colors`, its width in
/// the array `piece.widths`, its page points (x, y, z) in `piece.K.points`
/// and its offsets in `piece.K.offsets`, doubles and indices as the bake
/// holds them, for `bakeAssemble` to take back to the bit. Refused, as of
/// another document, unless its `fingerprint` is this document's bake's.
pub fn bake_piece(d: &Document, o: &JsonObject) -> Shared {
    let piece = piece_of(o.get("piece").ok_or("wants a 'piece'")?)?;
    of_this_document(o, d.bake_fingerprint(), "the piece")?;
    let baked = d.bake_piece(piece)?;
    let mut arrays = vec![("piece.widths".to_string(), Array::f64(baked.iter().map(|(s, _)| s.width)))];
    for (k, (_, paths)) in baked.iter().enumerate() {
        arrays
            .push((format!("piece.{k}.points"), Array::f64(paths.vertices.iter().flat_map(|p| [p.x(), p.y(), p.z()]))));
        arrays.push((format!("piece.{k}.offsets"), Array::u32(paths.offsets.iter().map(|&o| o as u32))));
    }
    let mut out = JsonObject::new();
    out.insert("colors".to_string(), JsonValue::Array(baked.iter().map(|(s, _)| json::string(&s.color)).collect()));
    Ok((out, arrays))
}

/// The bake from its pieces: `plan` the pieces in order, `colors` each one's
/// stroke colours as `bakePiece` gave them, and its arrays given as inputs
/// under `P.widths`, `P.K.points` and `P.K.offsets` for piece P. Answers as
/// `bake` does. Or `failed`: a piece's refusal, said as the bake's status
/// and given back.
///
/// `fingerprint` and `frame` are the plan's. The pieces are all of the
/// plan's document (each was refused otherwise), so the strokes are that
/// document's bake whatever this engine holds now. Where it holds that
/// document still, the reply is `bake`'s; where it has moved on since (an
/// edit or a move after Export was asked), the strokes are still the plan's
/// document's, consistently: the frame is the plan's, the status says so,
/// and the reply says `snapshot`, as one engine's bake, asked before the
/// edit, baked the document as it was.
pub fn bake_assemble(d: &mut Document, o: &JsonObject, mut inputs: BTreeMap<String, Vec<u8>>) -> Shared {
    if let Some(failed) = o.get("failed").and_then(JsonValue::as_str) {
        return Err(d.bake_failed(failed.to_string()).into());
    }
    let theirs = o.get("fingerprint").and_then(JsonValue::as_str).ok_or("wants the plan's 'fingerprint'")?;
    let snapshot = d.bake_fingerprint().as_deref() != Some(theirs);
    let plan: Vec<Piece> = o
        .get("plan")
        .and_then(JsonValue::as_array)
        .ok_or("wants a 'plan'")?
        .iter()
        .map(piece_of)
        .collect::<Result<_, _>>()?;
    let colors = o.get("colors").and_then(JsonValue::as_array).ok_or("wants 'colors'")?;
    if colors.len() != plan.len() {
        return Err(format!("{} pieces planned, colours for {}", plan.len(), colors.len()).into());
    }
    let mut baked = Vec::with_capacity(plan.len());
    for (p, colors) in colors.iter().enumerate() {
        let colors = colors.as_array().ok_or("a piece's colours are a list")?;
        let widths = doubles(take(&mut inputs, &format!("{p}.widths"))?)?;
        if widths.len() != colors.len() {
            return Err(format!("piece {p} gives {} widths for {} colours", widths.len(), colors.len()).into());
        }
        let mut layers: Baked = Vec::with_capacity(colors.len());
        for (k, (color, width)) in colors.iter().zip(widths).enumerate() {
            let color = color.as_str().ok_or("a colour is a string")?.to_string();
            let points = doubles(take(&mut inputs, &format!("{p}.{k}.points"))?)?;
            let offsets = indices(take(&mut inputs, &format!("{p}.{k}.offsets"))?)?;
            layers.push((Style { color, width }, polylines(points, offsets)?));
        }
        baked.push(layers);
    }
    let strokes = pieces::assemble(&plan, baked)?;
    let (mut out, arrays) = bake_reply(d, &strokes, o.get("svg").and_then(JsonValue::as_bool) == Some(true));
    if snapshot {
        d.bake_status =
            Some(format!("{} paths, of the document as it was when the bake was asked", strokes.path_count()));
        out.insert("frame".to_string(), o.get("frame").cloned().unwrap_or(JsonValue::Null));
        out.insert("snapshot".to_string(), JsonValue::Bool(true));
    } else {
        d.bake_status = Some(format!("{} paths", strokes.path_count()));
    }
    Ok((out, arrays))
}

/// What `bake` answers: each stroke layer's page points (x, y) and offsets,
/// its style, the SVG when asked, and the frame that maps a page point to
/// the viewport's pixel, as the preview's does.
pub fn bake_reply(d: &Document, strokes: &Strokes, with_svg: bool) -> (JsonObject, Vec<(String, Array)>) {
    let mut arrays = Vec::new();
    let mut styles = Vec::new();
    for (i, (style, paths)) in strokes.layers.iter().enumerate() {
        let points = paths.vertices.iter().flat_map(|p| [p.x(), p.y()]);
        let offsets = paths.offsets.iter().map(|&o| o as u32);
        arrays.push((format!("stroke.{i}.points"), Array::f64(points)));
        arrays.push((format!("stroke.{i}.offsets"), Array::u32(offsets)));
        styles.push(json::object([("width", JsonValue::Double(style.width)), ("color", json::string(&style.color))]));
    }
    if with_svg {
        let text = svg::render(strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS);
        arrays.push(("svg".to_string(), Array::u8(text.into_bytes())));
    }
    let mut out = JsonObject::new();
    out.insert("strokes".to_string(), JsonValue::Array(styles));
    out.insert("frame".to_string(), frame_of(d));
    (out, arrays)
}

/// The layers a frame at rest would lay out, holding `have`, each with its
/// weight (its vertices; the fold curves, derived for the camera, weigh
/// what the scene's largest layer does), and the document's fingerprint.
pub fn rest_layers(d: &Document, nonce: &str, have: &BTreeSet<String>) -> Shared {
    let layers = d.rest_layers(nonce, have);
    let scene = d.shown_scene().ok_or("nothing shown")?;
    let largest = scene.layers().iter().map(|l| l.paths.vertices.len()).max().unwrap_or(0);
    let weights: Vec<i64> = layers
        .iter()
        .map(|&i| {
            let l = &scene.layers()[i];
            let derived = matches!(l.spec.source, kurven_core::manifest::LayerSource::FoldCurves);
            (if derived { largest.max(l.paths.vertices.len()) } else { l.paths.vertices.len() }) as i64
        })
        .collect();
    let mut out = JsonObject::new();
    out.insert("layers".to_string(), json::ints(&layers.iter().map(|&i| i as i64).collect::<Vec<_>>()));
    out.insert("weights".to_string(), json::ints(&weights));
    out.insert("fingerprint".to_string(), d.layout_fingerprint().map_or(JsonValue::Null, |f| json::string(&f)));
    Ok((out, Vec::new()))
}

/// The layouts at rest of `layers`: each one's world points (x, y, z) in
/// `laid.I.vertices` and offsets in `laid.I.offsets`, its refusal in
/// `laid`, and the document's fingerprint, for `preview` to be given.
/// Refused, as of another document, unless its `fingerprint` is this
/// document's.
pub fn rest_layout(d: &Document, o: &JsonObject) -> Shared {
    of_this_document(o, d.layout_fingerprint(), "the layout asked for")?;
    let layers: Vec<usize> = o
        .get("layers")
        .and_then(JsonValue::as_array)
        .ok_or("wants 'layers'")?
        .iter()
        .map(|v| v.as_i64().and_then(|i| usize::try_from(i).ok()).ok_or("a layer is an index"))
        .collect::<Result<_, _>>()?;
    let mut arrays = Vec::new();
    let mut described = Vec::new();
    for (i, laid) in d.rest_laid(&layers)? {
        let v = laid.paths.vertices.iter().flat_map(|p| [p.x(), p.y(), p.z()]);
        arrays.push((format!("laid.{i}.vertices"), Array::f64(v)));
        arrays.push((format!("laid.{i}.offsets"), Array::u32(laid.paths.offsets.iter().map(|&o| o as u32))));
        described.push(json::object([
            ("layer", JsonValue::Int(i as i64)),
            ("refusal", laid.refusal.as_deref().map_or(JsonValue::Null, json::string)),
        ]));
    }
    let mut out = JsonObject::new();
    out.insert("laid".to_string(), JsonValue::Array(described));
    out.insert("fingerprint".to_string(), d.layout_fingerprint().map_or(JsonValue::Null, |f| json::string(&f)));
    Ok((out, arrays))
}

/// The layouts a `preview` request is given (`laid`, as `restLayout`
/// described them, their arrays as inputs), checked to be of this document
/// (`fingerprint`); None when it is given none.
pub fn given_layout(
    d: &Document,
    o: &JsonObject,
    mut inputs: BTreeMap<String, Vec<u8>>,
) -> Result<Option<BTreeMap<usize, Laid>>, Refused> {
    let Some(laid) = o.get("laid") else { return Ok(None) };
    let laid = laid.as_array().ok_or("'laid' is a list")?;
    of_this_document(o, d.layout_fingerprint(), "the layout given")?;
    let mut given = BTreeMap::new();
    for entry in laid {
        let i = entry.get("layer").and_then(JsonValue::as_i64).and_then(|i| usize::try_from(i).ok());
        let i = i.ok_or("a layout names its layer")?;
        let vertices = doubles(take(&mut inputs, &format!("laid.{i}.vertices"))?)?;
        let offsets = indices(take(&mut inputs, &format!("laid.{i}.offsets"))?)?;
        let refusal = entry.get("refusal").and_then(JsonValue::as_str).map(str::to_string);
        given.insert(i, Laid { paths: polylines(vertices, offsets)?, refusal });
    }
    Ok(Some(given))
}

fn take(inputs: &mut BTreeMap<String, Vec<u8>>, name: &str) -> Result<Vec<u8>, String> {
    inputs.remove(name).ok_or_else(|| format!("wants the input '{name}'"))
}

fn doubles(bytes: Vec<u8>) -> Result<Vec<f64>, String> {
    let (whole, rest) = bytes.as_chunks::<8>();
    if !rest.is_empty() {
        return Err("an array of doubles is a multiple of eight bytes".to_string());
    }
    Ok(whole.iter().map(|&b| f64::from_le_bytes(b)).collect())
}

fn indices(bytes: Vec<u8>) -> Result<Vec<usize>, String> {
    let (whole, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err("an array of indices is a multiple of four bytes".to_string());
    }
    Ok(whole.iter().map(|&b| u32::from_le_bytes(b) as usize).collect())
}

/// Paths from their points, three doubles each, and offsets, checked as a
/// polyline set holds them.
fn polylines<S>(points: Vec<f64>, offsets: Vec<usize>) -> Result<PolylineSet<S>, String> {
    let (whole, rest) = points.as_chunks::<3>();
    if !rest.is_empty() {
        return Err("points are three doubles each".to_string());
    }
    let n = whole.len();
    let ordered = offsets.windows(2).all(|w| w[0] <= w[1]);
    if offsets.first() != Some(&0) || offsets.last() != Some(&n) || !ordered {
        return Err(format!("offsets that do not cut {n} points into paths"));
    }
    let vertices = whole.iter().map(|c| P3::new(c[0], c[1], c[2])).collect();
    Ok(PolylineSet::new(vertices, offsets, None))
}
