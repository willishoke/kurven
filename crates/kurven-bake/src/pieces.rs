//! A bake in pieces, for a pool of engines that share no memory: the page's
//! Workers, or a desktop host's threads.
//!
//! Swift spreads a bake over its cores inside `Heightfield.visibleRuns`: every
//! vertex of a layer is tested on a thread (`parallelMap(_:chunk:worker:_:)`,
//! four vertices at a time, each thread with a traversal forked for it), then
//! every fold crossing, then every place a run turns, and the results are put
//! back in the order the items went in. Each of those items belongs to one
//! path, and no path's runs read another's: the runs of a layer are its
//! paths' runs, path after path. So a layer of a landscape splits at path
//! boundaries into pieces whose runs, put end to end in the paths' order, are
//! the layer's -- the same seam Swift's threads take, gathered into pieces a
//! Worker can be sent whole, since Workers cannot hand each other a vertex at
//! a time. A worker's forked tests answer as the ones they were forked from,
//! which is what lets Swift's split vary with the thread count and the
//! drawing stay the same; here it lets the pieces vary with the pool.
//!
//! What does not split, a piece takes whole: a layer whose curves are
//! derived for the camera (the fold curves, a hatching on the graph), whose
//! derivation is serial between its stages; an unclipped layer, drawn as it
//! comes; a clipped layer with no vertices. A scene that is not a landscape,
//! or an outline asked for, is one piece: the bake itself.
//!
//! `bake_piece` of every piece of `plan`, gathered by `assemble` in the
//! plan's order, is `bake(...).strokes` to the bit, however many engines
//! computed the pieces and in whatever order they finished.

use std::fmt;

use kurven_core::hatching::FaceSelector;
use kurven_core::hidden_line::HiddenLine;
use kurven_core::manifest::LayerSource;
use kurven_core::parametric::PortRefusal;
use kurven_core::polyline::{Layer, PolylineSet};
use kurven_core::scene::{Scene, SceneGeometry};
use kurven_core::spaces::PageSpace;
use kurven_core::svg::{Strokes, Style};

use crate::{BakeError, BakeOptions, bake};

/// A share of a bake's work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    /// The whole bake (`bake`): a scene that is not a landscape, or one with
    /// an outline asked for.
    Whole,
    /// One layer of a landscape, tested whole (`Scene::visible`).
    Layer(usize),
    /// Paths `from..to` of one layer of a landscape: their runs.
    Paths { layer: usize, from: usize, to: usize },
}

/// The stroke layers a piece draws, in order: the whole bake's for `Whole`,
/// a layer's (one per width of a hatching) for `Layer`, and one for `Paths`,
/// its share of its layer's strokes.
pub type Baked = Vec<(Style, PolylineSet<PageSpace>)>;

/// The pieces a bake of `scene` splits into, in the order their strokes are
/// drawn: each landscape layer that splits, in pieces of about
/// `vertices_per_piece` vertices of whole paths (never a piece of none), and
/// every other layer whole. Refuses what `bake` refuses before it tests a
/// curve -- a perspective camera, a scene that projects to nothing -- with
/// `bake`'s error.
pub fn plan(scene: &Scene, options: &BakeOptions, vertices_per_piece: usize) -> Result<Vec<Piece>, BakeError> {
    if scene.camera.is_perspective() {
        return Err(BakeError::Perspective);
    }
    if options.outline.is_some() || scene.heightfield().is_none() {
        return Ok(vec![Piece::Whole]);
    }
    if scene.projects_to_nothing()? {
        return Err(BakeError::EmptyScene);
    }
    let budget = vertices_per_piece.max(1);
    let mut pieces = Vec::new();
    for (i, layer) in scene.layers().iter().enumerate() {
        let paths = &layer.paths;
        if !splits(scene, layer) || paths.vertices.is_empty() {
            pieces.push(Piece::Layer(i));
            continue;
        }
        // Whole paths to a piece until it holds the budget; a piece of no
        // vertices would be drawn as its paths come (`visible_curves` passes
        // an empty set through), where the layer's runs drop them, so
        // vertexless paths go with the piece before them, or at the start,
        // after them.
        let mut from = 0;
        let mut held = 0;
        for p in 0..paths.count() {
            held += paths.offsets[p + 1] - paths.offsets[p];
            if held >= budget {
                pieces.push(Piece::Paths { layer: i, from, to: p + 1 });
                from = p + 1;
                held = 0;
            }
        }
        if from < paths.count() {
            if held > 0 {
                pieces.push(Piece::Paths { layer: i, from, to: paths.count() });
            } else if let Some(Piece::Paths { to, .. }) = pieces.last_mut() {
                // Only vertexless paths left: the layer's last piece takes them.
                *to = paths.count();
            }
        }
    }
    Ok(pieces)
}

/// Whether a landscape layer's visibility is its paths' runs, path after
/// path (`Scene::visible`'s heightfield arm, `Heightfield::visible_curves`):
/// a clipped layer whose curves are carried, not derived for the camera.
fn splits(scene: &Scene, layer: &Layer) -> bool {
    matches!(scene.geometry(), SceneGeometry::Heightfield(_))
        && layer.spec.clipped
        && !matches!(
            layer.spec.source,
            LayerSource::FoldCurves | LayerSource::Hatching { faces: FaceSelector::Graph { .. }, .. }
        )
}

/// Why a piece did not bake: the bake's own refusal, or a piece the scene
/// has no such share for (a layer or paths it does not have, a layer that
/// does not split): a plan made of another scene.
#[derive(Debug, Clone, PartialEq)]
pub enum PieceError {
    Bake(BakeError),
    NotOfThisScene(String),
}

impl fmt::Display for PieceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PieceError::Bake(error) => write!(f, "{error}"),
            PieceError::NotOfThisScene(why) => write!(f, "{why}"),
        }
    }
}

impl From<BakeError> for PieceError {
    fn from(error: BakeError) -> Self {
        PieceError::Bake(error)
    }
}

impl From<PortRefusal> for PieceError {
    fn from(refusal: PortRefusal) -> Self {
        PieceError::Bake(BakeError::Port(refusal))
    }
}

/// The strokes one piece draws, as `bake` draws them.
pub fn bake_piece(scene: &Scene, options: &BakeOptions, piece: Piece) -> Result<Baked, PieceError> {
    let depth_tolerance = Some(options.depth_tolerance.unwrap_or(scene.depth_tolerance));
    let tested = |layer: &Layer| -> Result<Baked, PieceError> {
        Ok(scene
            .visible(layer, depth_tolerance, None)?
            .iter()
            .map(|(layer, seen)| (Style::of(&layer.spec), HiddenLine::pass(seen)))
            .collect())
    };
    match piece {
        Piece::Whole => Ok(bake(scene, options)?.strokes.layers),
        Piece::Layer(i) => tested(layer_at(scene, i)?),
        Piece::Paths { layer, from, to } => {
            let whole = layer_at(scene, layer)?;
            if !splits(scene, whole) || from >= to || to > whole.paths.count() {
                return Err(PieceError::NotOfThisScene(format!("layer {layer} has no piece of paths {from}..{to}")));
            }
            tested(&Layer { spec: whole.spec.clone(), paths: sub_paths(&whole.paths, from, to) })
        }
    }
}

fn layer_at(scene: &Scene, i: usize) -> Result<&Layer, PieceError> {
    scene
        .layers()
        .get(i)
        .ok_or_else(|| PieceError::NotOfThisScene(format!("no layer {i}: the scene has {}", scene.layers().len())))
}

/// Paths `from..to` of a set, with their coordinates.
fn sub_paths<S: Copy>(paths: &PolylineSet<S>, from: usize, to: usize) -> PolylineSet<S> {
    let (lo, hi) = (paths.offsets[from], paths.offsets[to]);
    PolylineSet::new(
        paths.vertices[lo..hi].to_vec(),
        paths.offsets[from..=to].iter().map(|&o| o - lo).collect(),
        paths.coords.as_ref().map(|c| c[lo..hi].to_vec()),
    )
}

/// The drawing from its pieces' strokes, given in the plan's order: each
/// split layer's pieces put end to end as one stroke layer, every other
/// piece's layers as they come. `baked[k]` is piece `plan[k]`'s.
pub fn assemble(plan: &[Piece], baked: Vec<Baked>) -> Result<Strokes, String> {
    if plan.len() != baked.len() {
        return Err(format!("{} pieces planned, {} baked", plan.len(), baked.len()));
    }
    let mut layers: Baked = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for (piece, mut strokes) in plan.iter().zip(baked) {
        match *piece {
            Piece::Paths { layer, from, to } => {
                let [(style, paths)] = <[_; 1]>::try_from(std::mem::take(&mut strokes))
                    .map_err(|s: Baked| format!("a piece of paths drew {} layers, not one", s.len()))?;
                match (open, layers.last_mut()) {
                    (Some((l, end)), Some((_, set))) if l == layer && end == from => append(set, &paths),
                    _ if from == 0 => layers.push((style, paths)),
                    _ => return Err(format!("layer {layer}'s pieces are not in order at path {from}")),
                }
                open = Some((layer, to));
            }
            Piece::Layer(_) | Piece::Whole => {
                layers.extend(strokes);
                open = None;
            }
        }
    }
    Ok(Strokes { layers })
}

/// `more`'s paths after `set`'s.
fn append(set: &mut PolylineSet<PageSpace>, more: &PolylineSet<PageSpace>) {
    let base = set.vertices.len();
    set.vertices.extend_from_slice(&more.vertices);
    set.offsets.extend(more.offsets.iter().skip(1).map(|&o| o + base));
}
