//! Reading a `.kurven` bundle into values, restyling it, and writing it:
//! the port of `Sources/KurvenCore/Bundle.swift`.
//!
//! A described layer's curves come from `Surface::curves`, which dispatches
//! to `derive` (`surface.rs`), the face contours (`face_contour.rs`) and the
//! hatchings, the wall edges and the truncation edge (`hatch.rs`).

use std::fmt;
use std::path::{Path, PathBuf};

use crate::grid::Grid2D;
use crate::manifest::{GridRef, LayerSource, LayerSpec, Manifest, ManifestError, SchemaError, Walls};
use crate::mesh::Mesh;
use crate::npy::{self, NpyDType, NpyError};
use crate::polyline::{Layer, PolylineSet};
use crate::spaces::{P3, WorldSpace};
use crate::surface::{ContourRefine, Surface};

/// Why a bundle cannot be read or written.
#[derive(Debug, Clone, PartialEq)]
pub enum BundleError {
    MissingFile {
        file: String,
        bundle: String,
    },
    ShapeMismatch {
        file: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    BadOffsets {
        layer: String,
        last: i64,
        vertices: usize,
    },
    /// Offsets that do not start at 0 or that decrease somewhere: the
    /// paths they name overlap or begin past the first vertex. Swift checks
    /// only the last offset, and an array like [1, 5] or [0, 4, 2, 5] then
    /// traps in `PolylineSet`; the port refuses it here.
    UnorderedOffsets {
        layer: String,
        index: usize,
        offset: i64,
    },
    MissingWalls(String),
    /// The manifest could not be read or decoded.
    Schema(SchemaError),
    /// An array could not be read or written.
    Npy(NpyError),
    /// The file system refused a write.
    Io(String),
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BundleError::MissingFile { file, bundle } => write!(f, "bundle {bundle}: no file {file}"),
            BundleError::ShapeMismatch { file, expected, found } => {
                write!(f, "bundle: {file} has shape {found:?}, manifest says {expected:?}")
            }
            BundleError::BadOffsets { layer, last, vertices } => {
                write!(f, "bundle: layer {layer} offsets end at {last} but there are {vertices} vertices")
            }
            BundleError::UnorderedOffsets { layer, index, offset } => {
                write!(
                    f,
                    "bundle: layer {layer} offsets must start at 0 and never decrease, but offset {index} is {offset}"
                )
            }
            BundleError::MissingWalls(file) => {
                write!(f, "bundle: manifest names a wall mesh at {file}, which is absent")
            }
            BundleError::Schema(e) => write!(f, "{e}"),
            BundleError::Npy(e) => write!(f, "{e}"),
            BundleError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BundleError {}

impl From<SchemaError> for BundleError {
    fn from(e: SchemaError) -> BundleError {
        BundleError::Schema(e)
    }
}

impl From<NpyError> for BundleError {
    fn from(e: NpyError) -> BundleError {
        BundleError::Npy(e)
    }
}

/// A decoded bundle: the manifest, the grids, and the curves.
///
/// Arrays are eager. A bundle is a value the whole point of which is that it is
/// already computed; lazily faulting parts of it in would trade the one property
/// it has for nothing.
#[derive(Debug, Clone)]
pub struct KurvenBundle {
    pub url: PathBuf,
    pub manifest: Manifest,
    pub surface: Surface,
    pub layers: Vec<Layer>,
    /// Present when the manifest carries `Walls::Mesh`; `None` when the walls
    /// are derived from a perimeter instead.
    pub wall_mesh: Option<Mesh<WorldSpace>>,
    /// Present when the function behind the grids is available: every described
    /// contour layer is derived through it, now and after every restyle, so a
    /// truncation slider re-derives refined curves rather than falling back to the grid's.
    /// A bundle read from disk has none until a consumer that knows the function
    /// attaches one (`refined`).
    pub refine: Option<ContourRefine>,
}

impl KurvenBundle {
    pub fn new(
        url: PathBuf,
        manifest: Manifest,
        surface: Surface,
        layers: Vec<Layer>,
        wall_mesh: Option<Mesh<WorldSpace>>,
        refine: Option<ContourRefine>,
    ) -> KurvenBundle {
        KurvenBundle { url, manifest, surface, layers, wall_mesh, refine }
    }

    /// A bundle whose every layer is described: the curves are derived from the
    /// grids, as `read` derives it for a `--derived` bundle on disk.
    pub fn described(
        url: PathBuf,
        manifest: Manifest,
        surface: Surface,
        refine: Option<ContourRefine>,
    ) -> KurvenBundle {
        let layers = manifest
            .layers
            .iter()
            .map(|spec| Layer { spec: spec.clone(), paths: surface.curves(spec, &manifest.occluder, refine.as_ref()) })
            .collect();
        KurvenBundle { url, manifest, surface, layers, wall_mesh: None, refine }
    }

    /// The same bundle deriving its contour layers through `refine` -- or,
    /// with `None`, through the grids alone. Dumped layers are untouched.
    pub fn refined(&self, refine: Option<ContourRefine>) -> KurvenBundle {
        let base = KurvenBundle {
            url: self.url.clone(),
            manifest: self.manifest.clone(),
            surface: self.surface.clone(),
            layers: Vec::new(),
            wall_mesh: self.wall_mesh.clone(),
            refine: refine.clone(),
        };
        let layers = self
            .manifest
            .layers
            .iter()
            .map(|spec| if spec.files().is_none() { base.redrawn(spec) } else { self.redrawn(spec) })
            .collect();
        KurvenBundle { layers, ..base }
    }

    pub fn layer(&self, name: &str) -> Result<&Layer, ManifestError> {
        self.layers.iter().find(|l| l.spec.name == name).ok_or_else(|| ManifestError::NoSuchLayer {
            name: name.to_string(),
            have: self.layers.iter().map(|l| l.spec.name.clone()).collect(),
        })
    }

    /// Everything the depth pass draws besides the heightfield itself: the
    /// walls, the pieces of the truncated surface the heightfield's own
    /// triangles cut off (`Mesh::truncation_edge_mesh`), and the risers where
    /// two copies meet at a tile boundary with different tops
    /// (`Mesh::tile_boundary_risers`).
    pub fn occluder(&self) -> Mesh<WorldSpace> {
        let o = &self.manifest.occluder;
        let step = o.step.max(0) as usize;
        Mesh::concat(&[
            self.walls(),
            Mesh::truncation_edge_mesh(&self.surface, step, &o.region, &o.tiles),
            Mesh::tile_boundary_risers(&self.surface, step, &o.region, &o.tiles),
        ])
    }

    /// The occluding walls, whether dumped or described.
    pub fn walls(&self) -> Mesh<WorldSpace> {
        match &self.manifest.occluder.walls {
            Walls::None => Mesh::empty(),
            Walls::Mesh { .. } => self.wall_mesh.clone().unwrap_or_else(Mesh::empty),
            Walls::Perimeter { perimeter, base } => {
                Mesh::walls(perimeter, &self.surface, *base, &self.manifest.occluder.tiles)
            }
        }
    }

    /// Read a bundle directory.
    pub fn read(url: impl AsRef<Path>) -> Result<KurvenBundle, BundleError> {
        let url = url.as_ref();
        let name = url.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let file = |rel: &str| -> Result<PathBuf, BundleError> {
            let u = url.join(rel);
            if !u.exists() {
                return Err(BundleError::MissingFile { file: rel.to_string(), bundle: name.clone() });
            }
            Ok(u)
        };

        let manifest = Manifest::read(file("manifest.json")?)?;

        let grid = |r: &GridRef| -> Result<Grid2D<f32>, BundleError> {
            let a = npy::read(file(&r.file)?)?;
            let expected = vec![r.shape.0.max(0) as usize, r.shape.1.max(0) as usize];
            if a.shape != expected || r.shape.0 < 0 || r.shape.1 < 0 {
                return Err(BundleError::ShapeMismatch { file: r.file.clone(), expected, found: a.shape.clone() });
            }
            Ok(Grid2D::new(expected[1], expected[0], manifest.domain, a.floats()?))
        };

        let height = grid(&manifest.height)?;
        let phase = match &manifest.phase {
            Some(r) => Some(grid(r)?),
            None => None,
        };

        // A bundle written from a cached grid has no evaluator behind it, so its
        // heights were looked up by nearest pixel. `provenance.function` naming
        // a cache is the only signal of that in the file; treat a bundle whose
        // params mention a cache as cached.
        let cached = manifest.provenance.params.contains_key("cache");

        let surface = Surface::new(height, phase, manifest.truncation.clone(), cached);

        let mut layers: Vec<Layer> = Vec::new();
        for spec in &manifest.layers {
            let Some((vertices, offsets)) = spec.files() else {
                // Described, not dumped: derive the curves from the grids the
                // bundle already carries. This is what a `--derived` bundle
                // trades its layer files for, and what makes the levels, the
                // hatch spacing and the truncation editable.
                layers.push(Layer { spec: spec.clone(), paths: surface.curves(spec, &manifest.occluder, None) });
                continue;
            };
            let v = npy::read(file(vertices)?)?;
            let o = npy::read(file(offsets)?)?;
            let verts: Vec<P3<WorldSpace>> = v.rows3()?.into_iter().map(|r| P3::new(r[0], r[1], r[2])).collect();
            let offsets = o.ints()?;
            let last = offsets.last().copied().unwrap_or(0);
            if last != verts.len() as i64 || offsets.iter().any(|&k| k < 0) {
                return Err(BundleError::BadOffsets { layer: spec.name.clone(), last, vertices: verts.len() });
            }
            let mut previous = 0;
            for (index, &offset) in offsets.iter().enumerate() {
                if (index == 0 && offset != 0) || offset < previous {
                    return Err(BundleError::UnorderedOffsets { layer: spec.name.clone(), index, offset });
                }
                previous = offset;
            }
            let offsets: Vec<usize> = offsets.into_iter().map(|k| k as usize).collect();
            layers.push(Layer { spec: spec.clone(), paths: PolylineSet::new(verts, offsets, None) });
        }

        let mut wall_mesh: Option<Mesh<WorldSpace>> = None;
        if let Walls::Mesh { vertices, triangles } = &manifest.occluder.walls {
            let v = npy::read(file(vertices)?)?;
            let t = npy::read(file(triangles)?)?;
            wall_mesh = Some(Mesh::new(
                v.rows3()?.into_iter().map(|r| P3::new(r[0], r[1], r[2])).collect(),
                t.rows3i()?.into_iter().map(|r| r.map(|i| i as i32)).collect(),
            ));
        }

        Ok(KurvenBundle { url: url.to_path_buf(), manifest, surface, layers, wall_mesh, refine: None })
    }

    /// Re-derive one described layer at different levels.
    ///
    /// Only meaningful for `LayerSource::Contour`: a dumped layer is an answer
    /// with no question behind it, and this returns it unchanged. That
    /// asymmetry is the point of describing layers at all -- a bundle exported
    /// with `--derived` has editable level sets and one exported without does
    /// not, and the type says which you have.
    pub fn layer_at_levels(&self, spec: &LayerSpec, levels: &[f64]) -> Layer {
        let LayerSource::Contour { field, restriction, faces, light, .. } = &spec.source else {
            return self.redrawn(spec);
        };
        let mut edited = spec.clone();
        edited.source = LayerSource::Contour {
            field: *field,
            levels: levels.to_vec(),
            restriction: restriction.clone(),
            faces: faces.clone(),
            light: *light,
        };
        self.redrawn(&edited)
    }

    /// One layer, drawn from whatever its spec now says.
    ///
    /// A described layer is derived again; a dumped one keeps the curves it was
    /// read with, because the bundle carries no question that could produce it
    /// a second time.
    pub fn redrawn(&self, spec: &LayerSpec) -> Layer {
        if spec.files().is_some() {
            let existing = self.layers.iter().find(|l| l.spec.name == spec.name);
            return Layer { spec: spec.clone(), paths: existing.map_or_else(PolylineSet::empty, |l| l.paths.clone()) };
        }
        Layer { spec: spec.clone(), paths: self.surface.curves(spec, &self.manifest.occluder, self.refine.as_ref()) }
    }

    /// The same grids under a different manifest.
    ///
    /// The landscape does not change -- the samples are the samples -- but the
    /// truncation, the levels, the hatch spacing and the walls all can, and all of them
    /// are derived from the manifest. This is the edit path for every one of
    /// them, and the reason a truncation slider costs milliseconds instead of a round
    /// trip to Python: only the *curves* and the wall curtains are rebuilt, from
    /// grids that never left memory.
    ///
    /// It is a restyling, not a resampling: passing a manifest whose grids,
    /// domain or shape differ from this bundle's would describe a landscape
    /// these arrays are not, so those fields are taken from this bundle and the
    /// rest from the argument.
    pub fn restyled(&self, manifest: &Manifest) -> KurvenBundle {
        let mut honest = manifest.clone();
        honest.domain = self.manifest.domain;
        honest.height = self.manifest.height.clone();
        honest.phase = self.manifest.phase.clone();
        let restyled_surface = Surface {
            height: self.surface.height.clone(),
            phase: self.surface.phase.clone(),
            truncation: honest.truncation.clone(),
            cached: self.surface.cached,
        };
        let bundle = KurvenBundle {
            url: self.url.clone(),
            manifest: honest.clone(),
            surface: restyled_surface.clone(),
            layers: Vec::new(),
            wall_mesh: self.wall_mesh.clone(),
            refine: self.refine.clone(),
        };
        // A layer's curves are its spec drawn on the truncated surface past the
        // occluder, so a layer none of those moved for is the curves already
        // here: a hatch step draws the hatching, not every phase line again.
        let unmoved = honest.truncation == self.manifest.truncation && honest.occluder == self.manifest.occluder;
        let layers = honest
            .layers
            .iter()
            .map(|spec| {
                if unmoved && let Some(kept) = self.layers.iter().find(|l| l.spec == *spec) {
                    return kept.clone();
                }
                if spec.files().is_none() { bundle.redrawn(spec) } else { self.redrawn(spec) }
            })
            .collect();
        KurvenBundle { layers, ..bundle }
    }

    /// The levels a described layer was exported with, or `None` when it is
    /// dumped.
    pub fn levels(spec: &LayerSpec) -> Option<Vec<f64>> {
        match &spec.source {
            LayerSource::Contour { levels, .. } => Some(levels.clone()),
            _ => None,
        }
    }

    /// A contour's levels resampled to `count`, evenly over its own range, as
    /// a level slider sets them: never fewer than two, so a family saved at
    /// them keeps its range to be resampled again. `None` for a layer with
    /// fewer than two levels, which has no range.
    pub fn levels_resampled(spec: &LayerSpec, count: usize) -> Option<Vec<f64>> {
        let base = KurvenBundle::levels(spec)?;
        if base.len() < 2 {
            return None;
        }
        // Swift's `min()` and `max()`: the first of the least and of the
        // greatest, by `<`.
        let (mut lo, mut hi) = (base[0], base[0]);
        for &v in &base[1..] {
            if v < lo {
                lo = v;
            }
            if hi < v {
                hi = v;
            }
        }
        let n = count.max(2);
        Some((0..n).map(|k| lo + (hi - lo) * k as f64 / (n - 1) as f64).collect())
    }

    /// Write a described bundle: the manifest and the two grids.
    ///
    /// Every layer of a landscape is a description, so this is the whole of
    /// it. A bundle carrying dumped curves would need its layer files written too,
    /// and a manifest naming files that are not there is worse than refusing.
    pub fn write(&self, url: impl AsRef<Path>) -> Result<(), BundleError> {
        let url = url.as_ref();
        if !self.manifest.layers.iter().all(|l| l.files().is_none()) {
            return Err(BundleError::MissingFile {
                file: "layers/*.npy (this bundle has dumped layers)".to_string(),
                bundle: url.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            });
        }
        std::fs::create_dir_all(url).map_err(|e| BundleError::Io(format!("{}: {e}", url.display())))?;
        let height = &*self.surface.height;
        npy::write_f32(&height.values, &[height.height, height.width], url.join("height.npy"))?;
        if let Some(phase) = &self.surface.phase {
            npy::write_f32(&phase.values, &[phase.height, phase.width], url.join("phase.npy"))?;
        }
        let mut written = self.manifest.clone();
        written.height = GridRef {
            file: "height.npy".to_string(),
            shape: (height.height as i64, height.width as i64),
            dtype: NpyDType::Float32,
        };
        if let Some(phase) = &self.surface.phase {
            written.phase = Some(GridRef {
                file: "phase.npy".to_string(),
                shape: (phase.height as i64, phase.width as i64),
                dtype: NpyDType::Float32,
            });
        }
        // The manifest goes last, so a bundle whose manifest exists is a
        // bundle whose arrays are complete. Swift writes it atomically; so
        // does this, by a temporary file renamed into place.
        let path = url.join("manifest.json");
        let temporary = url.join(".manifest.json.tmp");
        std::fs::write(&temporary, written.canonical_json())
            .and_then(|()| std::fs::rename(&temporary, &path))
            .map_err(|e| BundleError::Io(format!("{}: {e}", path.display())))
    }
}
