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
use kurven_core::hatching::FaceSelector;
use kurven_core::json::{self, JsonObject, JsonValue};
use kurven_core::manifest::CameraPreset;
use kurven_core::manifest::LayerSource;
use kurven_core::polyline::PolylineSet;
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::spaces::P3;
use kurven_core::svg::{self, Strokes, Style};

use crate::document::Document;
use crate::preview::{Laid, content_key, curves_key, is_tested_exactly, lay_out, rest_key};
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

    /// `scene`'s fingerprint with `extra`: its camera, depth tolerance,
    /// resolution and mode, `extra`, and the hash of everything else the
    /// bake and the layouts read (`content_hash`). That hash is the cost, and
    /// is remembered in slot `slot` for what it was hashed for: the scene's
    /// content and curves -- counted per process, and new whenever a scene's
    /// geometry or curves are, since a scene is only ever made anew
    /// (`Scene::heightfield_scene`, `drawing`, `looking` keeping both but
    /// changing only the camera) -- and the open bundle's manifest, which
    /// names the function behind the refiner. A camera move rehashes nothing.
    fn fingerprinted(&self, slot: usize, scene: &Scene, extra: &[u64]) -> String {
        let manifest = self.open_manifest();
        let mut key = Hasher::new();
        key.text(&manifest);
        let key = vec![scene.content().raw(), scene.curves().raw(), key.0];
        let content = {
            let mut memo = self.preview_memo.fingerprints.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            match &memo[slot] {
                Some((k, f)) if *k == key => f.clone(),
                _ => {
                    let f = format!("{:016x}", content_hash(scene, &manifest));
                    memo[slot] = Some((key, f.clone()));
                    f
                }
            }
        };
        let mut h = Hasher::new();
        h.f64s(scene.camera.view.m.columns.iter().flat_map(|c| c.to_array()));
        h.debug(&scene.camera.projection);
        h.f64s([scene.depth_tolerance, scene.resolution]);
        h.debug(&scene.mode);
        for &w in extra {
            h.word(w);
        }
        h.text(&content);
        format!("{:016x}", h.0)
    }

    /// The open bundle's manifest, canonical, with whether a refiner is
    /// attached to it: what the scene's function, which no value of the
    /// scene shows (a refiner is closures), is built from.
    fn open_manifest(&self) -> String {
        if let Some(b) = self.bundle() {
            return format!("landscape {} {}", b.refine.is_some(), b.manifest.json().canonical());
        }
        if let Some(b) = self.surface_bundle() {
            return format!("surface {}", b.manifest.json().canonical());
        }
        if let Some(b) = self.polyhedron_bundle() {
            return format!("polyhedron {}", b.manifest.json().canonical());
        }
        String::new()
    }
}

/// FNV-1a over bytes (the preview's keys' hash, `preview::fnv`, over words),
/// fed values, text and `Debug` renderings without building them whole.
struct Hasher(u64);

impl Hasher {
    fn new() -> Hasher {
        Hasher(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn word(&mut self, w: u64) {
        self.bytes(&w.to_le_bytes());
    }

    fn f64s(&mut self, xs: impl IntoIterator<Item = f64>) {
        for x in xs {
            self.word(x.to_bits());
        }
    }

    /// Text, its length first, so two texts run together hash apart.
    fn text(&mut self, s: &str) {
        self.word(s.len() as u64);
        self.bytes(s.as_bytes());
    }

    /// A value's `Debug` rendering: Rust writes a double's shortest
    /// round trip, so it is the value to the bit.
    fn debug(&mut self, v: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let _ = write!(self, "{v:?}");
        self.bytes(b"|");
    }

    fn grid(&mut self, g: &kurven_core::grid::Grid2D<f32>) {
        self.word(g.width as u64);
        self.word(g.height as u64);
        self.f64s([g.domain.real.lo, g.domain.real.hi, g.domain.imag.lo, g.domain.imag.hi]);
        for &v in g.values.iter() {
            self.bytes(&v.to_bits().to_le_bytes());
        }
    }

    fn points<S>(&mut self, ps: &[P3<S>]) {
        self.word(ps.len() as u64);
        self.f64s(ps.iter().flat_map(|p| [p.x(), p.y(), p.z()]));
    }
}

impl std::fmt::Write for Hasher {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.bytes(s.as_bytes());
        Ok(())
    }
}

/// Everything of a scene the bake and the layouts read but its camera, its
/// depth tolerance and its resolution, with the open bundle's manifest
/// (`Document::open_manifest`), hashed: each layer whole -- its spec as a
/// manifest writes it (its source: the field and levels, a hatching's
/// spacing, angle, tone and the faces it lies on, the clip), its curves and
/// their surface coordinates -- and the geometry whole: a landscape's grids,
/// truncation, occluder, tiles, region, step, walls and refiner; a
/// parametric surface's grid, axes, outward side and neighbourhood; a
/// solid as it is.
fn content_hash(scene: &Scene, manifest: &str) -> u64 {
    let mut h = Hasher::new();
    h.text(manifest);
    h.word(scene.layers().len() as u64);
    for layer in scene.layers() {
        h.text(&layer.spec.json().canonical());
        h.points(&layer.paths.vertices);
        h.word(layer.paths.offsets.len() as u64);
        for &o in &layer.paths.offsets {
            h.word(o as u64);
        }
        match &layer.paths.coords {
            Some(c) => {
                h.word(c.len() as u64 + 1);
                h.f64s(c.iter().flat_map(|q| [q.x(), q.y()]));
            }
            None => h.word(0),
        }
    }
    match scene.geometry() {
        SceneGeometry::Heightfield(hf) => {
            h.bytes(b"heightfield");
            h.grid(&hf.surface.height);
            match &hf.surface.phase {
                Some(p) => {
                    h.word(1);
                    h.grid(p);
                }
                None => h.word(0),
            }
            h.text(&hf.surface.truncation.json().canonical());
            h.word(u64::from(hf.surface.cached));
            h.points(&hf.occluder.vertices);
            h.word(hf.occluder.triangles.len() as u64);
            for t in &hf.occluder.triangles {
                h.bytes(&[t[0].to_le_bytes(), t[1].to_le_bytes(), t[2].to_le_bytes()].concat());
            }
            h.debug(&hf.tiles);
            h.debug(&hf.region);
            h.word(hf.step as u64);
            h.debug(&hf.walls);
            match &hf.refine {
                Some(r) => {
                    h.word(1);
                    h.f64s([r.tolerance]);
                    let parts = [r.rings.is_some(), r.magnitude.is_some(), r.phase.is_some(), r.zero.is_some()];
                    for part in parts {
                        h.word(u64::from(part));
                    }
                }
                None => h.word(0),
            }
        }
        SceneGeometry::Parametric(s) => {
            h.bytes(b"parametric");
            h.points(&s.positions);
            h.debug(s);
            h.word(u64::from(s.second_order_jet.is_some()));
        }
        SceneGeometry::Solid(s) => {
            h.bytes(b"solid");
            h.debug(s);
        }
    }
    h.0
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
        Piece::Paths { layer, from, to, of } => {
            o.insert("layer".to_string(), JsonValue::Int(layer as i64));
            o.insert("from".to_string(), JsonValue::Int(from as i64));
            o.insert("to".to_string(), JsonValue::Int(to as i64));
            o.insert("of".to_string(), JsonValue::Int(of as i64));
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
    match (index("layer")?, index("from")?, index("to")?, index("of")?) {
        (Some(layer), Some(from), Some(to), Some(of)) => Ok(Piece::Paths { layer, from, to, of }),
        (Some(layer), None, None, None) => Ok(Piece::Layer(layer)),
        _ => Err("a piece is {whole}, {layer} or {layer, from, to, of}".to_string()),
    }
}

/// The plan, each piece with its weight.
///
/// With `bakeAlone`, a plan of one piece (a surface, a solid, an outline
/// asked for: nothing a pool can share) is not answered as a plan but baked
/// here, the reply `bake`'s with `baked` true: a helper would replay the
/// document, bake the same piece and send it back to be put together, for
/// nothing.
pub fn bake_plan(d: &mut Document, o: &JsonObject) -> Shared {
    let plan = d.bake_plan()?;
    if plan.len() == 1 && o.get("bakeAlone").and_then(JsonValue::as_bool) == Some(true) {
        let strokes = d.bake()?;
        let (mut out, arrays) = bake_reply(d, &strokes, o.get("svg").and_then(JsonValue::as_bool) == Some(true));
        out.insert("baked".to_string(), JsonValue::Bool(true));
        return Ok((out, arrays));
    }
    let scene = d.framed_scene().ok_or("nothing to bake")?;
    let weight = |p: &Piece| match *p {
        Piece::Paths { layer, from, to, .. } => weight(&scene, layer, Some((from, to))),
        Piece::Layer(i) => weight(&scene, i, None),
        Piece::Whole => scene.layers().iter().map(|l| l.paths.vertices.len()).sum(),
    };
    let mut out = JsonObject::new();
    out.insert("pieces".to_string(), JsonValue::Array(plan.iter().map(|p| piece_json(p, weight(p))).collect()));
    out.insert("fingerprint".to_string(), d.bake_fingerprint().map_or(JsonValue::Null, |f| json::string(&f)));
    out.insert("frame".to_string(), frame_of(d));
    Ok((out, Vec::new()))
}

/// How much testing a share of layer `i` is, for handing out the heaviest
/// first, one rule for the bake's pieces and the frame at rest's layers:
/// the vertices of paths `range`, or of the whole layer; a layer whose
/// curves are derived for the camera (the fold curves, a hatching on the
/// graph), whose curves are not known until derived, weighs at least what
/// the scene's largest layer does; an unclipped layer, drawn as it comes,
/// nothing.
fn weight(scene: &Scene, i: usize, range: Option<(usize, usize)>) -> usize {
    let layer = &scene.layers()[i];
    let offsets = &layer.paths.offsets;
    if let Some((from, to)) = range {
        return offsets[to] - offsets[from];
    }
    if !layer.spec.clipped {
        return 0;
    }
    let derived = matches!(
        layer.spec.source,
        LayerSource::FoldCurves | LayerSource::Hatching { faces: FaceSelector::Graph { .. }, .. }
    );
    let own = layer.paths.vertices.len();
    if derived { own.max(scene.layers().iter().map(|l| l.paths.vertices.len()).max().unwrap_or(0)) } else { own }
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
    if !snapshot {
        // Of this document: every layer of its framed scene drawn.
        let layers = d.framed_scene().map(|s| s.layers().len());
        pieces::covers(&plan, layers)?;
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
    let styles = crate::protocol::stroke_arrays(strokes, &mut arrays);
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
    let weights: Vec<i64> = layers.iter().map(|&i| weight(&scene, i, None) as i64).collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use kurven_core::manifest::{LayerSource, Region};
    use kurven_core::polyline::Layer;

    /// gamma at 160 samples, framed: a landscape with hatching, walls, a
    /// region, tiles and a refiner.
    fn gamma() -> Scene {
        let mut s = crate::protocol::Session::new();
        for r in [r#"{"kind":"create","preset":"gamma"}"#, r#"{"kind":"set","key":"res","value":160}"#] {
            assert!(s.call(r).json.contains(r#""ok":true"#), "{r}");
        }
        s.document.framed_scene().unwrap()
    }

    fn rebuilt(scene: &Scene, change: impl FnOnce(&mut kurven_core::scene::Heightfield)) -> Scene {
        let mut h = scene.heightfield().unwrap().clone();
        change(&mut h);
        Scene::heightfield_scene(
            h.surface,
            h.occluder,
            h.tiles,
            h.region,
            h.step,
            scene.layers().to_vec(),
            scene.camera,
            scene.mode,
            scene.depth_tolerance,
            h.refine,
            h.walls,
        )
    }

    /// Each input the bake reads, changed alone, changes the content's hash;
    /// the same scene rebuilt, with new content and curves, does not.
    #[test]
    fn every_input_of_the_bake_changes_the_fingerprint() {
        let scene = gamma();
        let base = content_hash(&scene, "");
        assert_eq!(content_hash(&rebuilt(&scene, |_| {}), ""), base, "the same inputs, the same hash");
        assert_ne!(content_hash(&scene, "another manifest"), base, "the manifest");
        let hatched =
            scene.layers().iter().position(|l| matches!(l.spec.source, LayerSource::Hatching { .. })).unwrap();
        let with = |change: &dyn Fn(&mut Layer)| {
            let mut layers = scene.layers().to_vec();
            change(&mut layers[hatched]);
            content_hash(&scene.drawing(layers), "")
        };
        let changed: Vec<(&str, u64)> = vec![
            (
                "a hatching's spacing, its paths kept",
                with(&|l| {
                    if let LayerSource::Hatching { hatching, .. } = &mut l.spec.source {
                        hatching.spacing *= 1.5;
                    }
                }),
            ),
            (
                "a hatching's angle",
                with(&|l| {
                    if let LayerSource::Hatching { hatching, .. } = &mut l.spec.source {
                        hatching.angle += 0.1;
                    }
                }),
            ),
            ("a layer's clip", with(&|l| l.spec.clipped = !l.spec.clipped)),
            (
                "surface coordinates on the curves",
                with(&|l| {
                    let n = l.paths.vertices.len();
                    l.paths.coords = Some(vec![kurven_core::spaces::P2::new(0.0, 0.0); n]);
                }),
            ),
            ("no refiner", content_hash(&rebuilt(&scene, |h| h.refine = None), "")),
            ("no walls", content_hash(&rebuilt(&scene, |h| h.walls = kurven_core::manifest::Walls::None), "")),
            ("a tile more", content_hash(&rebuilt(&scene, |h| h.tiles.push(h.tiles[0])), "")),
            (
                "a tile moved",
                content_hash(
                    &rebuilt(&scene, |h| {
                        h.tiles[0].tx += 1.0;
                    }),
                    "",
                ),
            ),
            (
                "another region",
                content_hash(
                    &rebuilt(&scene, |h| {
                        h.region = match (&h.region, &h.walls) {
                            (Region::Full, kurven_core::manifest::Walls::Perimeter { perimeter, .. }) => {
                                Region::Inside(perimeter.clone())
                            }
                            _ => Region::Full,
                        }
                    }),
                    "",
                ),
            ),
            ("the step", content_hash(&rebuilt(&scene, |h| h.step += 1), "")),
            ("the cache flag", content_hash(&rebuilt(&scene, |h| h.surface.cached = !h.surface.cached), "")),
        ];
        for (what, hash) in changed {
            assert_ne!(hash, base, "{what}");
        }
    }
}
