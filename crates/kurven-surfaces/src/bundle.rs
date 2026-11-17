//! A parametric surface's `.kurven` bundle: the port of
//! `Sources/KurvenDynamics/SurfaceBundle.swift`.

use std::path::{Path, PathBuf};

use kurven_core::bundle::{BundleError, BundleFiles, Source, holds_key_in, write_files};
use kurven_core::camera::Camera;
use kurven_core::json::JsonObject;
use kurven_core::manifest::{CameraPreset, LayerSpec, ManifestError, Provenance, SchemaError, SurfaceManifest};
use kurven_core::scene::{PreviewMode, Scene};
use kurven_core::surface_request::{Shape, SurfaceRequest};
use kurven_math::format;

use crate::drawing::{DrawingError, SurfaceDrawing};

/// Why a surface's bundle cannot be read or written.
#[derive(Debug)]
pub enum SurfaceBundleError {
    Bundle(BundleError),
    Schema(SchemaError),
    Drawing(DrawingError),
    Io(String),
}

impl std::fmt::Display for SurfaceBundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurfaceBundleError::Bundle(e) => write!(f, "{e}"),
            SurfaceBundleError::Schema(e) => write!(f, "{e}"),
            SurfaceBundleError::Drawing(e) => write!(f, "{e}"),
            SurfaceBundleError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SurfaceBundleError {}

impl From<SchemaError> for SurfaceBundleError {
    fn from(e: SchemaError) -> SurfaceBundleError {
        SurfaceBundleError::Schema(e)
    }
}
impl From<ManifestError> for SurfaceBundleError {
    fn from(e: ManifestError) -> SurfaceBundleError {
        SurfaceBundleError::Schema(SchemaError::Manifest(e))
    }
}
impl From<kurven_core::json::JsonError> for SurfaceBundleError {
    fn from(e: kurven_core::json::JsonError) -> SurfaceBundleError {
        SurfaceBundleError::Schema(SchemaError::Json(e))
    }
}
impl From<BundleError> for SurfaceBundleError {
    fn from(e: BundleError) -> SurfaceBundleError {
        match e {
            BundleError::Schema(e) => SurfaceBundleError::Schema(e),
            other => SurfaceBundleError::Bundle(other),
        }
    }
}
impl From<DrawingError> for SurfaceBundleError {
    fn from(e: DrawingError) -> SurfaceBundleError {
        SurfaceBundleError::Drawing(e)
    }
}

/// A parametric surface's `.kurven` bundle: its manifest, and the drawing its
/// request builds.
///
/// The bundle is the manifest alone (`SurfaceManifest`): the request, which
/// `SurfaceDrawing::build` turns into the same drawing `kurven-cli surface`
/// draws, and the layers as they were kept, which restyle it
/// (`SurfaceDrawing::restyled`). A landscape's bundle carries its grids
/// because f was sampled once; a parametric surface's map is code, and costs
/// less to run again than to store.
#[derive(Debug, Clone)]
pub struct SurfaceBundle {
    pub url: PathBuf,
    pub manifest: SurfaceManifest,
    pub drawing: SurfaceDrawing,
}

impl SurfaceBundle {
    /// The bundle that keeps `drawing`: its request, its layers as drawn, and
    /// the camera presets it opens at. Swift's defaults: the request's one
    /// preset, and `gitSha` "native".
    pub fn new(url: &Path, drawing: SurfaceDrawing, presets: Vec<CameraPreset>, git_sha: &str) -> SurfaceBundle {
        let manifest = SurfaceManifest::new(
            drawing.request.clone(),
            drawing.layers.iter().map(|l| l.spec.clone()).collect(),
            presets,
            Provenance {
                function: SurfaceBundle::function(&drawing.request),
                example: "surface".to_string(),
                params: JsonObject::new(),
                cpu_count: 1,
                git_sha: git_sha.to_string(),
            },
        );
        SurfaceBundle { url: url.to_path_buf(), manifest, drawing }
    }

    /// The same bundle drawing `layers` in place of its own: its request,
    /// presets and provenance kept, the surface and its identity too, so
    /// only the curves are new. The curves the request draws itself are the
    /// request's, as `layers` lists them or not (`SurfaceDrawing::keeping`):
    /// layers asked for while the request was being edited name its curves
    /// as they were before the edit.
    pub fn restyled(&self, layers: &[LayerSpec]) -> Result<SurfaceBundle, DrawingError> {
        let drawing = self.drawing.keeping(layers, None)?;
        let mut manifest = self.manifest.clone();
        manifest.layers = drawing.layers.iter().map(|l| l.spec.clone()).collect();
        Ok(SurfaceBundle { url: self.url.clone(), manifest, drawing })
    }

    /// The same bundle recording `git_sha` as the build that made it: what an
    /// edit undone, or a drag let go where it began, puts back.
    pub fn recording(&self, git_sha: &str) -> SurfaceBundle {
        let mut manifest = self.manifest.clone();
        manifest.provenance.git_sha = git_sha.to_string();
        SurfaceBundle { url: self.url.clone(), manifest, drawing: self.drawing.clone() }
    }

    /// The mathematics a request draws, for the provenance.
    pub fn function(request: &SurfaceRequest) -> String {
        match &request.shape {
            Shape::Torus { .. } => "torus".to_string(),
            Shape::Sn { modulus, .. } | Shape::Relief { modulus, .. } => {
                format!("sn(z, {})", format::description(*modulus))
            }
            Shape::Periodic { expression, .. } => expression.clone(),
            Shape::Forced { system, .. } => system.clone(),
        }
    }

    /// Whether the bundle at `url` holds a parametric surface rather than a
    /// landscape: its manifest has a `surface` (`BundleManifest`). False
    /// where it has no manifest, for the landscape's reader to say so.
    pub fn holds_surface(url: &Path) -> Result<bool, SurfaceBundleError> {
        Ok(holds_key_in(&Source::Directory(url), "surface")?)
    }

    /// Read a bundle and build its drawing: the request built, then drawn
    /// with the manifest's layers.
    pub fn read(url: &Path) -> Result<SurfaceBundle, SurfaceBundleError> {
        SurfaceBundle::read_from(&Source::Directory(url))
    }

    /// `read`, wherever the bundle's files are: a directory, or in memory.
    pub fn read_from(source: &Source) -> Result<SurfaceBundle, SurfaceBundleError> {
        let manifest = SurfaceManifest::from_json(&source.manifest_json()?)?;
        if manifest.presets.is_empty() {
            return Err(ManifestError::Unsupported(
                "a parametric surface with no camera preset: it has no view to open at or to bake from".to_string(),
            )
            .into());
        }
        let drawing = SurfaceDrawing::build(&manifest.surface, None, false)?.restyled(&manifest.layers)?;
        Ok(SurfaceBundle { url: source.url(), manifest, drawing })
    }

    /// Write the bundle: a directory holding its manifest, with `layers` in
    /// place of its own when given -- the layers as they are drawn. The
    /// manifest is written to a temporary file and renamed into place, as
    /// Swift's `atomically: true` does.
    pub fn write(&self, url: &Path, layers: Option<&[LayerSpec]>) -> Result<(), SurfaceBundleError> {
        Ok(write_files(&self.files(layers), url)?)
    }

    /// The files `write` writes: the manifest alone.
    pub fn files(&self, layers: Option<&[LayerSpec]>) -> BundleFiles {
        let mut manifest = self.manifest.clone();
        if let Some(layers) = layers {
            manifest.layers = layers.to_vec();
        }
        BundleFiles::from([("manifest.json".to_string(), manifest.canonical_json().into_bytes())])
    }

    /// The scene the bundle describes under one of its presets.
    pub fn scene(&self, preset: &CameraPreset) -> Scene {
        Scene::parametric_scene(
            self.drawing.surface.clone(),
            self.drawing.layers.clone(),
            Camera::oblique(&preset.projection),
            PreviewMode::Drawing,
            preset.depth_tolerance,
        )
    }
}
