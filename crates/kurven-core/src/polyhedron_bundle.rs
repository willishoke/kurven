//! A polyhedron's `.kurven` bundle: the port of
//! `Sources/KurvenCore/PolyhedronBundle.swift`.
//!
//! The bundle is the manifest alone (`PolyhedronManifest`): the vertices and
//! loops, from which the solid is built as `kurven-cli surface solid` builds
//! it, and the layers, its edges and the hatching laid on its faces. Reading
//! it draws what the CLI draws from the same solid, byte for byte.

use std::path::{Path, PathBuf};

use crate::bundle::BundleError;
use crate::camera::Camera;
use crate::hatching::FaceSelector;
use crate::json::{self, JsonObject};
use crate::manifest::{CameraPreset, LayerSource, LayerSpec, ManifestError, PolyhedronManifest, Provenance};
use crate::polyline::Layer;
use crate::scene::{PreviewMode, Scene};
use crate::solid::Solid;

/// A polyhedron's bundle: its manifest, and the layers drawn on its solid.
#[derive(Debug, Clone)]
pub struct PolyhedronBundle {
    pub url: PathBuf,
    pub manifest: PolyhedronManifest,
    /// The layers as drawn on the solid, before a camera: every edge, and
    /// every hatch line.
    pub layers: Vec<Layer>,
}

impl PolyhedronBundle {
    pub fn solid(&self) -> &Solid {
        &self.manifest.solid
    }

    /// The bundle that keeps `solid` with `layers` drawn on it, and the
    /// camera presets it opens at. The layers are those a polyhedron's
    /// manifest takes (`PolyhedronManifest::unsupported`). Swift's `gitSha`
    /// defaults to "native".
    pub fn new(
        url: PathBuf,
        solid: Solid,
        layers: Vec<LayerSpec>,
        presets: Vec<CameraPreset>,
        function: &str,
        git_sha: &str,
    ) -> PolyhedronBundle {
        let provenance = Provenance {
            function: function.to_string(),
            example: "polyhedron".to_string(),
            params: JsonObject::new(),
            cpu_count: 1,
            git_sha: git_sha.to_string(),
        };
        let drawn = layers.iter().map(|spec| PolyhedronBundle::drawn(spec, &solid)).collect();
        let manifest = PolyhedronManifest::new(solid, layers, presets, provenance);
        PolyhedronBundle { url, manifest, layers: drawn }
    }

    /// One layer drawn on the solid: its edges, or its hatch lines, as the
    /// CLI draws them. The scene tests both for its camera (`Scene::visible`).
    pub fn drawn(spec: &LayerSpec, solid: &Solid) -> Layer {
        match &spec.source {
            LayerSource::Hatching { hatching, faces: FaceSelector::Solid } => {
                Layer { spec: spec.clone(), paths: solid.hatch_paths(hatching) }
            }
            _ => Layer { spec: spec.clone(), paths: solid.edge_paths() },
        }
    }

    /// The same solid with `layers` drawn on it, each kept as it is drawn
    /// here when its source is unchanged, so a restyle of one hatching lays
    /// that hatching and nothing else; or why the solid cannot take them.
    /// The layers keep their places and presets: a restyle, not a new
    /// polyhedron.
    pub fn restyled(&self, layers: Vec<LayerSpec>) -> Result<PolyhedronBundle, ManifestError> {
        if let Some(why) = PolyhedronManifest::unsupported(&layers, self.solid()) {
            return Err(ManifestError::Invalid(why));
        }
        let drawn = layers
            .iter()
            .map(|spec| match self.layers.iter().find(|l| l.spec.source == spec.source) {
                Some(kept) => Layer { spec: spec.clone(), paths: kept.paths.clone() },
                None => PolyhedronBundle::drawn(spec, self.solid()),
            })
            .collect();
        let mut manifest = self.manifest.clone();
        manifest.layers = layers;
        Ok(PolyhedronBundle { url: self.url.clone(), manifest, layers: drawn })
    }

    /// Whether the bundle at `url` holds a polyhedron: its manifest has a
    /// `polyhedron` (`BundleManifest`). False where it has no manifest, for
    /// the landscape's reader to say so.
    pub fn holds_polyhedron(url: impl AsRef<Path>) -> Result<bool, BundleError> {
        let file = url.as_ref().join("manifest.json");
        if !file.exists() {
            return Ok(false);
        }
        let text = std::fs::read_to_string(&file).map_err(|e| BundleError::Io(format!("{}: {e}", file.display())))?;
        Ok(json::parse(&text)
            .map_err(|e| BundleError::Schema(e.into()))?
            .object("Manifest")
            .map_err(|e| BundleError::Schema(e.into()))?
            .contains_key("polyhedron"))
    }

    /// Read a bundle: the solid built from its vertices and loops, checked,
    /// and its layers drawn on it.
    pub fn read(url: impl AsRef<Path>) -> Result<PolyhedronBundle, BundleError> {
        let url = url.as_ref();
        let file = url.join("manifest.json");
        if !file.exists() {
            return Err(BundleError::MissingFile {
                file: "manifest.json".to_string(),
                bundle: url.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            });
        }
        let text = std::fs::read_to_string(&file).map_err(|e| BundleError::Io(format!("{}: {e}", file.display())))?;
        let manifest = PolyhedronManifest::from_json(&json::parse(&text).map_err(|e| BundleError::Schema(e.into()))?)?;
        if manifest.presets.is_empty() {
            return Err(BundleError::Schema(
                ManifestError::Unsupported(
                    "a polyhedron with no camera preset: it has no view to open at or to bake from".to_string(),
                )
                .into(),
            ));
        }
        let layers = manifest.layers.iter().map(|spec| PolyhedronBundle::drawn(spec, &manifest.solid)).collect();
        Ok(PolyhedronBundle { url: url.to_path_buf(), manifest, layers })
    }

    /// Write the bundle: a directory holding its manifest, written to a
    /// temporary file and renamed into place (Swift's `atomically: true`).
    pub fn write(&self, url: impl AsRef<Path>) -> Result<(), BundleError> {
        let url = url.as_ref();
        std::fs::create_dir_all(url).map_err(|e| BundleError::Io(format!("{}: {e}", url.display())))?;
        let path = url.join("manifest.json");
        let temporary = url.join(".manifest.json.tmp");
        std::fs::write(&temporary, self.manifest.canonical_json())
            .and_then(|()| std::fs::rename(&temporary, &path))
            .map_err(|e| BundleError::Io(format!("{}: {e}", path.display())))
    }

    /// The scene the bundle describes under one of its presets.
    pub fn scene(&self, preset: &CameraPreset) -> Scene {
        Scene::solid_scene(
            self.solid().clone(),
            self.layers.clone(),
            Camera::oblique(&preset.projection),
            PreviewMode::Drawing,
            preset.depth_tolerance,
        )
    }
}
