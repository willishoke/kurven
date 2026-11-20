//! The picker's pictures: the port of `Sources/KurvenApp/Thumbnails.swift`'s
//! engine half, each catalog entry drawn by the path that draws it full size.
//!
//! Swift samples each function small, builds each surface on a coarser grid,
//! fits the camera it opens at to a 440 by 300 viewport and renders one
//! preview frame with Metal into a PNG it caches on disk. Here the engine
//! builds the same scene and navigator ([`draw`]) and answers the frame the
//! window's preview would draw of it (`preview::preview_scene`); the page
//! draws that on its GPU, or, without one, the judged strokes, and caches the
//! picture under [`key`]. Nothing of the open document is touched: a
//! thumbnail is a scene of its own.

use kurven_core::json::{self, JsonValue};
use kurven_core::navigation::{Framing, Navigator, Orbit, Viewport};
use kurven_core::scene::Scene;
use kurven_core::spaces::{P2, P3};
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_landscape::catalog::{Catalog, FunctionPreset, LandscapeRequest};
use kurven_landscape::landscape::NativeLandscape;
use kurven_surfaces::drawing::{SurfaceDrawing, SurfacePreset};

use crate::document::QUICK_BOUNDS;

/// The size they are drawn at: twice what the picker shows, so the curves
/// are halved rather than aliased (`Thumbnails.size`).
pub const SIZE: Viewport = Viewport { width: 440, height: 300 };
/// Samples along a function's longer side: a tenth of the work of the real
/// thing and, at 200 points across, indistinguishable from it.
pub const SAMPLES: i64 = 220;
/// Grid points around a surface's `u`.
pub const GRID: i64 = 256;
/// Bumped when the styling a landscape is given by default changes, which
/// changes the picture without changing anything the key is made of
/// (`Thumbnails.style`). Swift's is 2; the port's pictures start their own
/// count, being drawn by another renderer.
pub const STYLE: i64 = 1;

/// One cell of the picker: a function drawn as a landscape, or a surface
/// (`GalleryEntry`).
#[derive(Debug, Clone)]
pub enum Entry {
    Function(FunctionPreset),
    Surface(SurfacePreset),
}

impl Entry {
    /// Every entry, the functions first and then the surfaces, in catalog
    /// order: the picker's order and the order Swift warms them in.
    pub fn all() -> Vec<Entry> {
        let functions = Catalog::native().presets.iter().cloned().map(Entry::Function);
        functions.chain(SurfacePreset::catalog().into_iter().map(Entry::Surface)).collect()
    }

    /// Unique across both kinds: a function and a surface may share a name.
    pub fn id(&self) -> String {
        match self {
            Entry::Function(p) => p.name.clone(),
            Entry::Surface(p) => format!("surface:{}", p.name),
        }
    }

    /// The entry an id names.
    pub fn named(id: &str) -> Option<Entry> {
        Entry::all().into_iter().find(|e| e.id() == id)
    }

    /// The picker's cell, as the page shows it: its id, name, label, the
    /// line under the label (the expression, or the surface's formula), the
    /// notes, and the key its picture is cached under.
    pub fn json(&self) -> JsonValue {
        let (kind, name, label, formula, notes) = match self {
            Entry::Function(p) => ("function", &p.name, &p.label, &p.expression, &p.notes),
            Entry::Surface(p) => ("surface", &p.name, &p.label, &p.formula, &p.notes),
        };
        json::object([
            ("id", json::string(&self.id())),
            ("kind", json::string(kind)),
            ("name", json::string(name)),
            ("label", json::string(label)),
            ("formula", json::string(formula)),
            ("notes", json::string(notes)),
            ("thumbnail", json::string(&key(self))),
        ])
    }
}

/// What a surface's thumbnail is built from: its own request, coarser. A
/// trajectory as long as the window's, at a quarter the width, is grey
/// moiré: a quarter of it keeps its linear curves as far apart.
pub fn surface_request(preset: &SurfacePreset) -> SurfaceRequest {
    let mut request = preset.request.clone();
    request.grid = GRID;
    if let Shape::Forced { turns, .. } = &mut request.shape {
        *turns = (*turns / 4).max(1);
    }
    request
}

/// What a function's thumbnail is sampled from: its preset at `SAMPLES`,
/// hatched at about twenty strokes along the longest edge -- the default is
/// right for a bake and a black rectangle here.
pub fn landscape_request(preset: &FunctionPreset) -> LandscapeRequest {
    let mut request = LandscapeRequest::from_preset(preset, SAMPLES);
    let d = preset.domain;
    request.spacing = Some(kurven_math::swift::max(d.real.length().abs(), d.imag.length().abs()) / 22.0);
    request
}

/// Where an entry's picture is kept, keyed by everything that would change
/// it: the id, then FNV-1a of the expression, the window, the truncation,
/// the size and samples and the style (a surface: its request spelled out,
/// which is every number its drawing is built from), as Swift's
/// `cache(for:)` hashes them. Not a process-seeded hash: a cache keyed on
/// one misses every launch.
pub fn key(entry: &Entry) -> String {
    let text = match entry {
        Entry::Function(p) => {
            let d = p.domain;
            let truncation = p.truncation_height.map_or("none".to_string(), |h| format!("{h:?}"));
            format!(
                "{}|{:?},{:?},{:?},{:?}|{truncation}|{}x{}@{SAMPLES}|style{STYLE}",
                p.expression, d.real.lo, d.real.hi, d.imag.lo, d.imag.hi, SIZE.width, SIZE.height
            )
        }
        Entry::Surface(p) => {
            format!("surface|{}|{}x{}|style{STYLE}", surface_request(p).json().canonical(), SIZE.width, SIZE.height)
        }
    };
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{}-{hash:016x}", entry.id().replace(':', "-"))
}

/// A thumbnail's scene and the navigator it is drawn from, at `SIZE`.
pub struct Thumbnail {
    pub scene: Scene,
    pub navigator: Navigator,
}

/// Build an entry's scene and fit its camera, as Swift's `draw` does.
///
/// A function: sampled at `SAMPLES` and fitted to its exact bounds, not the
/// subsampled ones the window fits with -- a pole spire is a few samples
/// wide, and a coarse subsample can miss its tip. A surface: built from its
/// coarser request and drawn from the camera it opens at.
pub fn draw(entry: &Entry) -> Result<Thumbnail, String> {
    match entry {
        Entry::Function(preset) => {
            let bundle = NativeLandscape::build(&landscape_request(preset), "native", true)
                .map_err(|e| format!("{}: {e}", preset.name))?;
            let camera_preset =
                bundle.manifest.presets.first().cloned().ok_or("the landscape declares no camera presets")?;
            let mut navigator = Navigator::new(
                Orbit::matching(&camera_preset.projection, P3::new(0.0, 0.0, 0.0)),
                Framing::new(P2::new(0.0, 0.0), 1.0),
            );
            let mut scene = Scene::from_bundle(&bundle, &camera_preset).looking(navigator.camera());
            let bounds = scene.view_bounds().ok().flatten().or_else(|| scene.quick_bounds(QUICK_BOUNDS));
            if let Some(bounds) = bounds {
                navigator.framing = Framing::fitting(&bounds, SIZE, Framing::FIT_MARGIN);
                scene = scene.looking(navigator.camera());
            }
            Ok(Thumbnail { scene, navigator })
        }
        Entry::Surface(preset) => {
            let request = surface_request(preset);
            let drawing = SurfaceDrawing::build(&request, None, false).map_err(|e| format!("{}: {e}", preset.name))?;
            let orbit = Orbit::matching(&SurfaceRequest::PROJECTION, P3::new(0.0, 0.0, 0.0));
            let scene = drawing.scene(orbit.camera());
            let bounds = scene.quick_bounds(QUICK_BOUNDS).ok_or("the surface has no extent")?;
            let navigator = Navigator::new(orbit, Framing::fitting(&bounds, SIZE, Framing::FIT_MARGIN));
            Ok(Thumbnail { scene, navigator })
        }
    }
}
