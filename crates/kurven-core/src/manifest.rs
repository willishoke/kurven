//! The bundle schema: the one definition of it. The port of
//! `Sources/KurvenCore/Manifest.swift`.
//!
//! It began as the mirror of the Python package's dataclasses and is now the
//! only copy. `tests/fixtures/contract/*.kurven` are decoded here and
//! re-encoded, and the canonical JSON must come back byte for byte, so a
//! change to the schema is a change to those golden files, made on purpose.
//!
//! Every sum type here is an `enum` with an exhaustive `match` at each use
//! site, so "a truncation kind nobody handled" is a compile error rather than a
//! silently ignored branch.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;

use crate::hatching::{FaceKind, FaceSelector, Hatching, Tone, Turing};
use crate::json::{self, JsonError, JsonObject, JsonValue, Members, object, string};
use crate::npy::{NpyDType, NpyError};
use crate::simd::{Vec2, Vec3};
use crate::solid::{Solid, SolidError};
use crate::spaces::{P2, P3, Transform, WorldSpace};
use crate::surface_request::SurfaceRequest;
use crate::text;

#[derive(Debug, Clone, PartialEq)]
pub enum ManifestError {
    UnsupportedSchema {
        schema: i64,
        supported: i64,
    },
    UnexpectedAxes {
        axes: Vec<String>,
        expected: Vec<String>,
    },
    UnknownKind {
        kind: String,
        of: String,
        known: Vec<String>,
    },
    BadShape {
        what: String,
        shape: Vec<i64>,
    },
    NoSuchPreset {
        name: String,
        have: Vec<String>,
    },
    NoSuchLayer {
        name: String,
        have: Vec<String>,
    },
    Unsupported(String),
    /// A value the schema can spell that no drawing can be built from.
    Invalid(String),
}

impl ManifestError {
    pub fn unknown_kind(kind: &str, of: &str, known: &[&str]) -> ManifestError {
        ManifestError::UnknownKind {
            kind: kind.to_string(),
            of: of.to_string(),
            known: known.iter().map(|k| k.to_string()).collect(),
        }
    }

    pub fn bad_shape(what: &str, shape: &[i64]) -> ManifestError {
        ManifestError::BadShape { what: what.to_string(), shape: shape.to_vec() }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestError::UnsupportedSchema { schema, supported } => {
                write!(f, "manifest: schema {schema} is not the supported {supported}")
            }
            ManifestError::UnexpectedAxes { axes, expected } => write!(
                f,
                "manifest: axes {} are not {}; this bundle is in a different coordinate order",
                text::strings(axes),
                text::strings(expected)
            ),
            ManifestError::UnknownKind { kind, of, known } => {
                write!(f, "manifest: {of} kind '{kind}' is not one of {}", text::strings(known))
            }
            ManifestError::BadShape { what, shape } => write!(f, "manifest: {what} has shape {}", text::ints(shape)),
            ManifestError::NoSuchPreset { name, have } => {
                write!(f, "manifest: no preset '{name}' (have {})", text::strings(have))
            }
            ManifestError::NoSuchLayer { name, have } => {
                write!(f, "manifest: no layer '{name}' (have {})", text::strings(have))
            }
            ManifestError::Unsupported(what) | ManifestError::Invalid(what) => write!(f, "manifest: {what}"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// Whatever a decode of the schema can refuse with. Swift throws any
/// `Error` from these initializers, and its description is the message;
/// this is that union, its `Display` the same message.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaError {
    Json(JsonError),
    Manifest(ManifestError),
    Npy(NpyError),
    Solid(SolidError),
    /// The file could not be read.
    Io(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::Json(e) => e.fmt(f),
            SchemaError::Manifest(e) => e.fmt(f),
            SchemaError::Npy(e) => e.fmt(f),
            SchemaError::Solid(e) => e.fmt(f),
            SchemaError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SchemaError {}

impl From<JsonError> for SchemaError {
    fn from(e: JsonError) -> SchemaError {
        SchemaError::Json(e)
    }
}
impl From<ManifestError> for SchemaError {
    fn from(e: ManifestError) -> SchemaError {
        SchemaError::Manifest(e)
    }
}
impl From<NpyError> for SchemaError {
    fn from(e: NpyError) -> SchemaError {
        SchemaError::Npy(e)
    }
}
impl From<SolidError> for SchemaError {
    fn from(e: SolidError) -> SchemaError {
        SchemaError::Solid(e)
    }
}

/// Schema 3 names the face a contour or a hatching lies on (`FaceSelector`),
/// risers among them, and contours a face's coordinates and height as well
/// as |f| and arg f. Schema 2 took the standard names (truncation,
/// restriction, step, coordinate curves, ...) and one hatching kind for
/// every face. Older schemas are refused, not translated: there are no
/// bundles outside this repository.
///
/// A manifest with a `surface` holds a parametric surface by the request that
/// builds it (`SurfaceManifest`), and one with a `polyhedron` a polyhedron by
/// its vertices and loops (`PolyhedronManifest`), not a landscape. Both were
/// added to schema 3 without a bump: no key of a landscape's manifest
/// changed, so every bundle that read before reads and writes as it did, and
/// nothing outside this repository writes schema 3. So were isophotes, a
/// field and a light on a contour, a layer of integral curves
/// (`LayerSource::IntegralCurves`), a kind of its own, and the principal
/// directions, a direction field it can follow, on a parametric surface.
pub const KURVEN_SCHEMA: i64 = 3;
/// The axis convention every bundle states about itself. A bundle that says
/// anything else is not readable by this code, and saying so up front is the
/// whole point of writing it into the file.
pub const KURVEN_AXES: [&str; 3] = ["real", "imag", "magnitude"];

fn kurven_axes() -> Vec<String> {
    KURVEN_AXES.iter().map(|s| s.to_string()).collect()
}

fn read_json(path: &Path) -> Result<JsonValue, SchemaError> {
    let text = std::fs::read_to_string(path).map_err(|e| SchemaError::Io(format!("{}: {e}", path.display())))?;
    Ok(json::parse(&text)?)
}

// MARK: - leaves

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const fn new(lo: f64, hi: f64) -> Interval {
        Interval { lo, hi }
    }
    pub fn length(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn from_json(json: &JsonValue) -> Result<Interval, SchemaError> {
        let o = json.object("Interval")?;
        Ok(Interval::new(o.double("lo", "Interval")?, o.double("hi", "Interval")?))
    }
    pub fn json(&self) -> JsonValue {
        object([("lo", JsonValue::Double(self.lo)), ("hi", JsonValue::Double(self.hi))])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Domain {
    pub real: Interval,
    pub imag: Interval,
}

impl Domain {
    pub const fn new(real: Interval, imag: Interval) -> Domain {
        Domain { real, imag }
    }

    pub fn from_json(json: &JsonValue) -> Result<Domain, SchemaError> {
        let o = json.object("Domain")?;
        Ok(Domain::new(
            Interval::from_json(o.value("real", "Domain")?)?,
            Interval::from_json(o.value("imag", "Domain")?)?,
        ))
    }
    pub fn json(&self) -> JsonValue {
        object([("real", self.real.json()), ("imag", self.imag.json())])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridRef {
    pub file: String,
    /// `(ny, nx)` -- rows index imag, columns index real.
    pub shape: (i64, i64),
    pub dtype: NpyDType,
}

impl GridRef {
    pub fn from_json(json: &JsonValue) -> Result<GridRef, SchemaError> {
        let o = json.object("GridRef")?;
        let s = o.ints("shape", "GridRef")?;
        if s.len() != 2 {
            return Err(ManifestError::bad_shape("GridRef", &s).into());
        }
        let d = o.string("dtype", "GridRef")?;
        let dtype = NpyDType::ALL.into_iter().find(|t| t.descr() == d).ok_or_else(|| NpyError::UnsupportedDType {
            found: d.clone(),
            expected: NpyDType::ALL.iter().map(|t| t.descr().to_string()).collect(),
        })?;
        Ok(GridRef { file: o.string("file", "GridRef")?, shape: (s[0], s[1]), dtype })
    }
    pub fn json(&self) -> JsonValue {
        object([
            ("file", string(&self.file)),
            ("dtype", string(self.dtype.descr())),
            ("shape", json::ints(&[self.shape.0, self.shape.1])),
        ])
    }
}

/// A 2x3 affine map on the world (x, y) plane. Elliptic's eighteen reflected
/// copies are eighteen of these, and the base tile is the identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2 {
    pub a: f64,
    pub b: f64,
    pub tx: f64,
    pub c: f64,
    pub d: f64,
    pub ty: f64,
}

impl Affine2 {
    pub const IDENTITY: Affine2 = Affine2 { a: 1.0, b: 0.0, tx: 0.0, c: 0.0, d: 1.0, ty: 0.0 };

    /// The map of a point: Swift's `callAsFunction`.
    pub fn apply(&self, p: P3<WorldSpace>) -> P3<WorldSpace> {
        P3::new(self.a * p.x() + self.b * p.y() + self.tx, self.c * p.x() + self.d * p.y() + self.ty, p.z())
    }

    /// The map back. `None` for a degenerate tile, which a manifest should not
    /// contain and which this refuses to guess at.
    pub fn inverse(&self) -> Option<Affine2> {
        let det = self.a * self.d - self.b * self.c;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let (ia, ib) = (self.d / det, -self.b / det);
        let (ic, id) = (-self.c / det, self.a / det);
        Some(Affine2 {
            a: ia,
            b: ib,
            tx: -(ia * self.tx + ib * self.ty),
            c: ic,
            d: id,
            ty: -(ic * self.tx + id * self.ty),
        })
    }

    /// As a `Transform`, for folding into a camera or handing to a shader.
    pub fn transform(&self) -> Transform<WorldSpace, WorldSpace> {
        Transform::from_rows(
            (Vec3::new(self.a, self.b, 0.0), Vec3::new(self.c, self.d, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(self.tx, self.ty, 0.0),
        )
    }

    pub fn from_json(json: &JsonValue) -> Result<Affine2, SchemaError> {
        let o = json.object("Affine2")?;
        let m = o.doubles("m", "Affine2")?;
        if m.len() != 6 {
            return Err(ManifestError::bad_shape("Affine2", &[m.len() as i64]).into());
        }
        Ok(Affine2 { a: m[0], b: m[1], tx: m[2], c: m[3], d: m[4], ty: m[5] })
    }
    pub fn json(&self) -> JsonValue {
        object([("m", json::doubles(&[self.a, self.b, self.tx, self.c, self.d, self.ty]))])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerimeterEdge {
    pub start: P2<WorldSpace>,
    pub end: P2<WorldSpace>,
    pub density: i64,
}

impl PerimeterEdge {
    pub fn from_json(json: &JsonValue) -> Result<PerimeterEdge, SchemaError> {
        let o = json.object("Edge")?;
        let s = o.doubles("start", "Edge")?;
        let e = o.doubles("end", "Edge")?;
        if s.len() != 2 || e.len() != 2 {
            return Err(ManifestError::bad_shape("Edge", &[s.len() as i64, e.len() as i64]).into());
        }
        Ok(PerimeterEdge { start: P2::new(s[0], s[1]), end: P2::new(e[0], e[1]), density: o.int("density", "Edge")? })
    }
    pub fn json(&self) -> JsonValue {
        object([
            ("start", json::doubles(&[self.start.x(), self.start.y()])),
            ("end", json::doubles(&[self.end.x(), self.end.y()])),
            ("density", JsonValue::Int(self.density)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPerimeter {
    edges: Vec<PerimeterEdge>,
    /// The area the edges' starts enclose, by the shoelace formula, signed:
    /// positive when the perimeter runs counter-clockwise seen from above,
    /// negative clockwise. Kept with the edges, so `inward` costs one
    /// edge.
    signed_area: f64,
}

impl BoundaryPerimeter {
    pub fn new(edges: Vec<PerimeterEdge>) -> BoundaryPerimeter {
        let signed_area = BoundaryPerimeter::signed_area_of(&edges);
        BoundaryPerimeter { edges, signed_area }
    }

    pub fn edges(&self) -> &[PerimeterEdge] {
        &self.edges
    }

    /// Replace the edges, keeping the area with them (Swift's `didSet`).
    pub fn set_edges(&mut self, edges: Vec<PerimeterEdge>) {
        *self = BoundaryPerimeter::new(edges);
    }

    pub fn signed_area(&self) -> f64 {
        self.signed_area
    }

    pub fn from_json(json: &JsonValue) -> Result<BoundaryPerimeter, SchemaError> {
        let o = json.object("Perimeter")?;
        let edges = o.array("edges", "Perimeter")?.iter().map(PerimeterEdge::from_json).collect::<Result<_, _>>()?;
        Ok(BoundaryPerimeter::new(edges))
    }
    pub fn json(&self) -> JsonValue {
        object([("edges", JsonValue::Array(self.edges.iter().map(PerimeterEdge::json).collect()))])
    }

    fn signed_area_of(edges: &[PerimeterEdge]) -> f64 {
        let corners: Vec<Vec2> = edges.iter().map(|e| e.start.v).collect();
        let n = corners.len();
        0.5 * (0..n).fold(0.0, |a, k| {
            let (p, q) = (corners[k], corners[(k + 1) % n]);
            a + p.x * q.y - q.x * p.y
        })
    }

    /// Edge `k`'s unit normal into the footprint the perimeter bounds, in
    /// the ground plane: to the left of the edge's direction when the
    /// perimeter runs counter-clockwise, to the right when it runs
    /// clockwise. Zero for an edge of no length.
    pub fn inward(&self, k: usize) -> Vec2 {
        let d = self.edges[k].end.v - self.edges[k].start.v;
        let length = d.length();
        if !(length > 0.0) {
            return Vec2::ZERO;
        }
        let left = Vec2::new(-d.y, d.x) / length;
        if self.signed_area < 0.0 { -left } else { left }
    }
}

// MARK: - truncation

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealBand {
    /// The band is `x < below`, less the bands before it.
    pub below: f64,
    pub height: f64,
}

impl RealBand {
    fn from_json(json: &JsonValue) -> Result<RealBand, SchemaError> {
        let o = json.object("RealBand")?;
        Ok(RealBand { below: o.double("below", "RealBand")?, height: o.double("height", "RealBand")? })
    }
    fn json(&self) -> JsonValue {
        object([("below", JsonValue::Double(self.below)), ("height", JsonValue::Double(self.height))])
    }
}

/// Which side of a point along x a one-sided limit approaches it from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// From lesser x.
    Left,
    /// From greater x.
    Right,
}

/// One band's part of a stretch of Re z (`Truncation::bands`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandSpan {
    /// The band, as `band_at` numbers it.
    pub band: usize,
    pub height: f64,
    /// Where it holds the stretch: from `lo` up to, not including, `hi`.
    pub lo: f64,
    pub hi: f64,
}

/// The truncation of the surface from above: the drawn surface is
/// `min(|f|, h(x))` for the truncation height `h`.
///
/// A truncation is a property of the model, not the camera: it is what turns
/// a pole into a visible plateau.
#[derive(Debug, Clone, PartialEq)]
pub enum Truncation {
    None,
    Uniform(f64),
    /// Gamma's per-spire truncation: a height per band of Re z. `beyond` is
    /// the height where no band matches -- a separate field rather than a
    /// final band with a huge threshold, because "everything else" is what
    /// it means.
    RealBands {
        bands: Vec<RealBand>,
        beyond: f64,
    },
}

impl Truncation {
    /// The truncation height at world x (= Re z); `+inf` where there is
    /// none. On a band boundary, the band that starts there: the truncation
    /// is right-continuous, and this is its limit from the right.
    pub fn height_at(&self, x: f64) -> f64 {
        self.height_at_from(x, Side::Right)
    }

    /// The one-sided limit of the truncation height at x from `side`: the
    /// band just left of x or just right of it. The two differ only on a
    /// band boundary, where the band from the left is the one that ends
    /// there.
    pub fn height_at_from(&self, x: f64, side: Side) -> f64 {
        match self {
            Truncation::None => f64::INFINITY,
            Truncation::Uniform(h) => *h,
            Truncation::RealBands { bands, beyond } => {
                let band = match side {
                    Side::Right => bands.iter().find(|b| x < b.below),
                    Side::Left => bands.iter().find(|b| x <= b.below),
                };
                band.map_or(*beyond, |b| b.height)
            }
        }
    }

    /// The band world x lies in: the index of the first listed band it is
    /// under, the count of them beyond the last, and 0 for a uniform
    /// truncation's one band. `None` where there is no truncation. On a band
    /// boundary, the band that starts there.
    pub fn band_at(&self, x: f64) -> Option<usize> {
        self.band_at_from(x, Side::Right)
    }

    /// The band x lies in as a one-sided limit from `side`, as
    /// `height_at_from` reads it.
    pub fn band_at_from(&self, x: f64, side: Side) -> Option<usize> {
        match self {
            Truncation::None => None,
            Truncation::Uniform(_) => Some(0),
            Truncation::RealBands { bands, .. } => Some(
                match side {
                    Side::Right => bands.iter().position(|b| x < b.below),
                    Side::Left => bands.iter().position(|b| x <= b.below),
                }
                .unwrap_or(bands.len()),
            ),
        }
    }

    /// The bands that meet the stretch of Re z from `lo` to `hi`, in order
    /// along it, each with the part of the stretch it holds. The truncation is
    /// constant on each. Nothing where there is no truncation.
    pub fn bands(&self, lo: f64, hi: f64) -> Vec<BandSpan> {
        match self {
            Truncation::None => Vec::new(),
            Truncation::Uniform(h) => {
                if lo < hi {
                    vec![BandSpan { band: 0, height: *h, lo, hi }]
                } else {
                    Vec::new()
                }
            }
            Truncation::RealBands { bands, beyond } => {
                // A band is x < below, less the bands before it: from the
                // greatest boundary before it to its own.
                let mut out = Vec::new();
                let mut start = f64::NEG_INFINITY;
                for k in 0..=bands.len() {
                    let end = if k < bands.len() { bands[k].below } else { f64::INFINITY };
                    let (a, b) = (start.max(lo), end.min(hi));
                    if a < b {
                        let height = if k < bands.len() { bands[k].height } else { *beyond };
                        out.push(BandSpan { band: k, height, lo: a, hi: b });
                    }
                    if k < bands.len() {
                        start = start.max(end);
                    }
                }
                // Stable, as Swift's sort is.
                out.sort_by(|x, y| x.lo.total_cmp(&y.lo));
                out
            }
        }
    }

    /// The greatest truncation height anywhere; `+inf` where there is
    /// none.
    pub fn highest(&self) -> f64 {
        match self {
            Truncation::None => f64::INFINITY,
            Truncation::Uniform(h) => *h,
            Truncation::RealBands { bands, beyond } => bands.iter().map(|b| b.height).fold(*beyond, f64::max),
        }
    }

    pub fn from_json(json: &JsonValue) -> Result<Truncation, SchemaError> {
        let o = json.object("Truncation")?;
        let kind = o.string("kind", "Truncation")?;
        Ok(match kind.as_str() {
            "none" => Truncation::None,
            "uniform" => Truncation::Uniform(o.double("height", "Truncation.uniform")?),
            "realBands" => Truncation::RealBands {
                bands: o
                    .array("bands", "Truncation.realBands")?
                    .iter()
                    .map(RealBand::from_json)
                    .collect::<Result<_, _>>()?,
                beyond: o.optional_double("beyond").unwrap_or(f64::INFINITY),
            },
            other => {
                return Err(ManifestError::unknown_kind(other, "Truncation", &["none", "uniform", "realBands"]).into());
            }
        })
    }

    pub fn json(&self) -> JsonValue {
        match self {
            Truncation::None => object([("kind", string("none"))]),
            Truncation::Uniform(h) => object([("kind", string("uniform")), ("height", JsonValue::Double(*h))]),
            Truncation::RealBands { bands, beyond } => {
                let bands = JsonValue::Array(bands.iter().map(RealBand::json).collect());
                if beyond.is_finite() {
                    object([("kind", string("realBands")), ("bands", bands), ("beyond", JsonValue::Double(*beyond))])
                } else {
                    object([("kind", string("realBands")), ("bands", bands)])
                }
            }
        }
    }
}

// MARK: - occluder

/// The cut-face curtains: dumped as a mesh, or described as a perimeter.
///
/// A dumped mesh has its top edges at the *analytic* surface height, which is what
/// the Python plates draw. A derived one re-evaluates the top edge by bilinear
/// lookup on `height.npy`, so it differs by the grid's interpolation error --
/// a difference worth measuring rather than assuming away.
#[derive(Debug, Clone, PartialEq)]
pub enum Walls {
    None,
    Mesh { vertices: String, triangles: String },
    Perimeter { perimeter: BoundaryPerimeter, base: f64 },
}

impl Walls {
    fn from_json(json: &JsonValue) -> Result<Walls, SchemaError> {
        let o = json.object("Walls")?;
        let kind = o.string("kind", "Walls")?;
        Ok(match kind.as_str() {
            "none" => Walls::None,
            "mesh" => Walls::Mesh {
                vertices: o.string("vertices", "Walls.mesh")?,
                triangles: o.string("triangles", "Walls.mesh")?,
            },
            "perimeter" => Walls::Perimeter {
                perimeter: BoundaryPerimeter::from_json(o.value("perimeter", "Walls.perimeter")?)?,
                base: o.double("base", "Walls.perimeter")?,
            },
            other => return Err(ManifestError::unknown_kind(other, "Walls", &["none", "mesh", "perimeter"]).into()),
        })
    }
    fn json(&self) -> JsonValue {
        match self {
            Walls::None => object([("kind", string("none"))]),
            Walls::Mesh { vertices, triangles } => {
                object([("kind", string("mesh")), ("vertices", string(vertices)), ("triangles", string(triangles))])
            }
            Walls::Perimeter { perimeter, base } => object([
                ("kind", string("perimeter")),
                ("perimeter", perimeter.json()),
                ("base", JsonValue::Double(*base)),
            ]),
        }
    }
}

/// The footprint the heightfield is rasterized over.
///
/// zeta's landscape is a staircase, not a rectangle: the notch in front of the
/// s = 1 pole is where the drawing cuts away to show the cross-section. It is an
/// absence of geometry, not a mask applied afterwards, so a cell contributes
/// triangles only when all four of its corners are inside -- no bridging, and
/// nothing to draw over.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    Full,
    Inside(BoundaryPerimeter),
}

impl Region {
    /// Whether a world point is inside. `Full` admits everything.
    pub fn contains(&self, p: P2<WorldSpace>) -> bool {
        match self {
            Region::Full => true,
            Region::Inside(perimeter) => {
                let edges = perimeter.edges();
                point_in_polygon_by(p, edges.len(), |i| edges[i].start)
            }
        }
    }

    fn from_json(json: &JsonValue) -> Result<Region, SchemaError> {
        let o = json.object("Region")?;
        let kind = o.string("kind", "Region")?;
        Ok(match kind.as_str() {
            "full" => Region::Full,
            "inside" => Region::Inside(BoundaryPerimeter::from_json(o.value("perimeter", "Region.inside")?)?),
            other => return Err(ManifestError::unknown_kind(other, "Region", &["full", "inside"]).into()),
        })
    }
    fn json(&self) -> JsonValue {
        match self {
            Region::Full => object([("kind", string("full"))]),
            Region::Inside(p) => object([("kind", string("inside")), ("perimeter", p.json())]),
        }
    }
}

/// Even-odd ray crossing, scanning along world x (= real).
///
/// The scan axis is part of the contract, not an implementation detail: a test
/// that rays along the other axis agrees everywhere except on the boundary, and
/// the boundary is exactly where a staircase cutout lives. `kurven.bundle`'s
/// `point_in_polygon` is this function, and both `Perimeter` types on the Python
/// side delegate to it for the same reason.
pub fn point_in_polygon(p: P2<WorldSpace>, corners: &[P2<WorldSpace>]) -> bool {
    point_in_polygon_by(p, corners.len(), |i| corners[i])
}

/// `point_in_polygon` over `n` corners read by index, so a perimeter's
/// edges are scanned in place, with nothing allocated per point.
fn point_in_polygon_by(p: P2<WorldSpace>, n: usize, corner: impl Fn(usize) -> P2<WorldSpace>) -> bool {
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (corner(i), corner((i + 1) % n));
        if a.x() == b.x() {
            continue;
        }
        let crossing = (b.y() - a.y()) * (p.x() - a.x()) / (b.x() - a.x()) + a.y();
        if (a.x() > p.x()) != (b.x() > p.x()) && p.y() < crossing {
            inside = !inside;
        }
    }
    inside
}

/// What blocks lines of sight.
///
/// The heightfield is never triangles on disk: it is `height.npy` plus a `step`,
/// instanced once per entry in `tiles`. A renderer rasterizes it from the
/// texture at whatever detail it is drawing at, which is the only way the
/// elliptic plate's nineteen tiles and gamma's 20000-square bake are affordable.
///
/// `tiles` always contains the identity as its first entry, and `region` is the
/// footprint each tile is clipped to.
#[derive(Debug, Clone, PartialEq)]
pub struct Occluder {
    pub step: i64,
    pub tiles: Vec<Affine2>,
    pub walls: Walls,
    pub region: Region,
    pub base: f64,
}

impl Occluder {
    fn from_json(json: &JsonValue) -> Result<Occluder, SchemaError> {
        let o = json.object("Occluder")?;
        Ok(Occluder {
            step: o.int("step", "Occluder")?,
            tiles: o.array("tiles", "Occluder")?.iter().map(Affine2::from_json).collect::<Result<_, _>>()?,
            walls: Walls::from_json(o.value("walls", "Occluder")?)?,
            region: Region::from_json(o.value("region", "Occluder")?)?,
            base: o.optional_double("base").unwrap_or(0.0),
        })
    }
    fn json(&self) -> JsonValue {
        object([
            ("step", JsonValue::Int(self.step)),
            ("tiles", JsonValue::Array(self.tiles.iter().map(Affine2::json).collect())),
            ("walls", self.walls.json()),
            ("region", self.region.json()),
            ("base", JsonValue::Double(self.base)),
        ])
    }
}

// MARK: - layers

/// A plain enumeration with the names a manifest spells: `raw` is Swift's
/// `rawValue`, `parse` its `init(rawValue:)`, `ALL` its `allCases`.
macro_rules! raw_enum {
    ($(#[$doc:meta])* $name:ident { $($(#[$vdoc:meta])* $case:ident = $raw:literal),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vdoc])* $case,)+
        }

        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$case,)+];

            pub fn raw(self) -> &'static str {
                match self {
                    $($name::$case => $raw,)+
                }
            }

            pub fn parse(raw: &str) -> Option<$name> {
                match raw {
                    $($raw => Some($name::$case),)+
                    _ => None,
                }
            }

            pub fn names() -> Vec<&'static str> {
                $name::ALL.iter().map(|c| c.raw()).collect()
            }
        }
    };
}

raw_enum!(LayerRole { Magnitude = "magnitude", Phase = "phase", Scaffold = "scaffold", Outline = "outline" });

raw_enum!(
    /// How a layer's z was chosen. Recorded so a consumer that re-contours the grid
    /// itself can reproduce the lift rather than guess it.
    HeightPolicy { Surface = "surface", Level = "level", Magnitude = "magnitude" }
);

raw_enum!(
    /// Which field of its faces a contour layer is a level set of: |f| or arg f
    /// on the graph, or on any face a coordinate of its parameter rectangle, `u`
    /// and `v`, or its height. On the graph and a plateau u and v are Re z and
    /// Im z, and the height |f| and h(x); on a wall and a riser, (s, z), the
    /// distance along the line it stands on and the height, which is z again.
    /// A parametric surface and a polyhedron face expose their own. `isophote`
    /// is n · l, the outward unit normal against the unit direction toward a
    /// light the layer names, on a curved face that bounds a solid: its level
    /// sets are the isophotes (Koenderink and van Doorn 1980).
    ContourField {
        Magnitude = "magnitude",
        Phase = "phase",
        U = "u",
        V = "v",
        Height = "height",
        Isophote = "isophote",
    }
);

impl ContourField {
    /// Whether this is one of f's fields, placed by the refiner.
    pub fn is_field_of_f(self) -> bool {
        self == ContourField::Magnitude || self == ContourField::Phase
    }

    /// The function this field is on `faces`, by the name of one field that
    /// is it: on the graph the height is |f|, so `Height` is `Magnitude`
    /// there; any other field is itself.
    pub fn function_on(self, faces: &FaceSelector) -> ContourField {
        if self == ContourField::Height && faces.kind() == FaceKind::Graph { ContourField::Magnitude } else { self }
    }

    /// What the field's values are computed from (`FieldClass`).
    pub fn field_class(self) -> FieldClass {
        match self {
            ContourField::U | ContourField::V => FieldClass::Coordinate,
            ContourField::Height | ContourField::Isophote => FieldClass::Embedding,
            ContourField::Magnitude | ContourField::Phase => FieldClass::Function,
        }
    }

    fn parse_or_refuse(text: &str) -> Result<ContourField, ManifestError> {
        ContourField::parse(text)
            .ok_or_else(|| ManifestError::unknown_kind(text, "ContourField", &ContourField::names()))
    }
}

raw_enum!(
    /// One of a surface's two principal curvatures at each point, by order: the
    /// greater or the lesser, measured against the normal into the solid the
    /// surface bounds, so a convex surface's are positive
    /// (`Face.principalCurvatures(at:on:)`). On the round torus the greater is
    /// the tube's, 1/r, and its lines of curvature are the meridians; the
    /// lesser's are the parallels.
    PrincipalCurvature { Greater = "greater", Lesser = "lesser" }
);

/// A direction field on a face, by name: what a layer of integral curves
/// follows (`LayerSource::IntegralCurves`). Each is a vector field (D9),
/// whose integral curves are traced both ways from each seed
/// (`VectorField`): signed, as a slope is, or standing for a field of line
/// elements, unsigned, as the principal directions are, which the
/// integrator orients as it goes (`VectorField.signed`). There is no
/// line-field type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DirectionField {
    /// The gradient of a scalar field in the face's own metric: its integral
    /// curves are the field's slope lines (Cayley 1859, "On contour and slope
    /// lines"), which cross its level sets at right angles on the face. The
    /// slope lines of |f| on the graph are the lines of steepest ascent and
    /// descent, and where f is analytic and not zero they are its phase
    /// contours, since log f is analytic too.
    Slope(ContourField),
    /// The principal direction of one principal curvature, the greater or
    /// the lesser (do Carmo §3-2): its integral curves are the lines of
    /// curvature of that family, and the two families cross at right angles
    /// in space. Unsigned, and undefined at an umbilic, where the two
    /// curvatures are equal. A layer draws one family; both are two layers.
    PrincipalDirection(PrincipalCurvature),
}

impl DirectionField {
    fn from_json(json: &JsonValue) -> Result<DirectionField, SchemaError> {
        let o = json.object("DirectionField")?;
        let kind = o.string("kind", "DirectionField")?;
        Ok(match kind.as_str() {
            "slope" => DirectionField::Slope(ContourField::parse_or_refuse(&o.string("of", "DirectionField.slope")?)?),
            "principalDirection" => {
                let text = o.string("of", "DirectionField.principalDirection")?;
                let curvature = PrincipalCurvature::parse(&text).ok_or_else(|| {
                    ManifestError::unknown_kind(&text, "PrincipalCurvature", &PrincipalCurvature::names())
                })?;
                DirectionField::PrincipalDirection(curvature)
            }
            other => {
                return Err(
                    ManifestError::unknown_kind(other, "DirectionField", &["slope", "principalDirection"]).into()
                );
            }
        })
    }

    pub fn json(&self) -> JsonValue {
        match self {
            DirectionField::Slope(field) => object([("kind", string("slope")), ("of", string(field.raw()))]),
            DirectionField::PrincipalDirection(c) => {
                object([("kind", string("principalDirection")), ("of", string(c.raw()))])
            }
        }
    }
}

/// How the seeds are placed along the level set.
#[derive(Debug, Clone, PartialEq)]
pub enum Spacing {
    /// Where `field` takes each of the values along it: arg f at listed
    /// angles along a contour of |f|, say.
    Values(ContourField, Vec<f64>),
    /// About `spacing` apart along each of its curves, measured in space:
    /// a curve of length L takes n = max(1, round(L / spacing)) seeds, at
    /// the middles of its n equal stretches.
    ArcLength(f64),
}

/// Where a layer's integral curves start (`LayerSource::IntegralCurves`): on
/// the level set of `field` at `level`, at the points along it where a
/// second field takes listed values, or evenly spaced by arc length along
/// it. Cayley's construction: slope lines seeded along a contour.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeds {
    pub field: ContourField,
    pub level: f64,
    pub spacing: Spacing,
}

impl Seeds {
    fn from_json(json: &JsonValue) -> Result<Seeds, SchemaError> {
        let o = json.object("Seeds")?;
        let on = o.value("on", "Seeds")?.object("Seeds.on")?;
        let field = ContourField::parse_or_refuse(&on.string("field", "Seeds.on")?)?;
        let level = on.double("level", "Seeds.on")?;
        let at = o.value("at", "Seeds")?.object("Seeds.at")?;
        let kind = at.string("kind", "Seeds.at")?;
        let spacing = match kind.as_str() {
            "values" => Spacing::Values(
                ContourField::parse_or_refuse(&at.string("field", "Seeds.at")?)?,
                at.doubles("values", "Seeds.at")?,
            ),
            "arcLength" => Spacing::ArcLength(at.double("spacing", "Seeds.at")?),
            other => return Err(ManifestError::unknown_kind(other, "Seeds.at", &["values", "arcLength"]).into()),
        };
        Ok(Seeds { field, level, spacing })
    }

    pub fn json(&self) -> JsonValue {
        let at = match &self.spacing {
            Spacing::Values(of, values) => {
                object([("kind", string("values")), ("field", string(of.raw())), ("values", json::doubles(values))])
            }
            Spacing::ArcLength(spacing) => {
                object([("kind", string("arcLength")), ("spacing", JsonValue::Double(*spacing))])
            }
        };
        object([
            ("on", object([("field", string(self.field.raw())), ("level", JsonValue::Double(self.level))])),
            ("at", at),
        ])
    }

    /// Why these seeds cannot be placed on `faces`, or `None` when they can: a
    /// level or a value that is not a number, a level set of |f| at 0 or
    /// below, or of the height on the graph, where it is |f|, which is a set
    /// of points or empty (a parametric surface's height takes any sign), no values, a
    /// spacing that is not positive, or under `cell`, the grid's cell where
    /// it is known, a field the faces do not expose, an isophote, whose light
    /// a layer of integral curves does not keep, and values of the function
    /// whose level set the seeds lie on, under its name or another
    /// (`ContourField::function_on`), which takes one value there.
    pub fn unsupported(&self, faces: &FaceSelector, cell: Option<f64>) -> Option<String> {
        let positive = |f: ContourField| f.function_on(faces) == ContourField::Magnitude;
        let mut fields = vec![self.field];
        if !self.level.is_finite() {
            return Some(format!("seeds on a level set at {}: a level is a number", text::double(self.level)));
        }
        if positive(self.field) && self.level <= 0.0 {
            return Some(format!(
                "seeds on the level set {} = {}: it is f's zeros or nothing, no curve to seed along",
                self.field.raw(),
                text::double(self.level)
            ));
        }
        match &self.spacing {
            Spacing::Values(of, values) => {
                fields.push(*of);
                if of.function_on(faces) == self.field.function_on(faces) {
                    return Some(format!(
                        "seeds at values of {} along a level set of {}: on the {} they are one function, which takes \
                         one value there",
                        of.raw(),
                        self.field.raw(),
                        faces.name()
                    ));
                }
                if values.is_empty() {
                    return Some(format!("seeds at no values of {}", of.raw()));
                }
                if let Some(&bad) = values.iter().find(|&&v| !v.is_finite() || (positive(*of) && v <= 0.0)) {
                    return Some(format!(
                        "seeds where {} is {}: {}",
                        of.raw(),
                        text::double(bad),
                        if bad.is_finite() { "it is positive" } else { "a value is a number" }
                    ));
                }
            }
            Spacing::ArcLength(spacing) => {
                if !(spacing.is_finite() && *spacing > 0.0) {
                    return Some(format!("seeds spaced {} apart: a spacing is positive", text::double(*spacing)));
                }
                if let Some(cell) = cell
                    && *spacing < cell
                {
                    return Some(format!(
                        "seeds spaced {} apart, closer than a cell of the grid, {}",
                        text::double(*spacing),
                        text::double(cell)
                    ));
                }
            }
        }
        for f in fields {
            if f == ContourField::Isophote {
                return Some("seeds placed by an isophote, whose light integral curves do not keep".to_string());
            }
            if let Some(why) = faces.unsupported_field(f) {
                return Some(why);
            }
        }
        None
    }
}

/// A class of scalar fields, by what their values are computed from, which
/// decides what an edit of a surface redoes for their contours. A
/// coordinate is the parameter rectangle's own, so its contours are kept in
/// parameter coordinates while the rectangle and its samples are, and only
/// carried into space again when the surface moves. The height is read off
/// the map, so its contours are found again whenever the surface moves. |f|
/// and arg f come from f's samples, which a new function or sampling takes
/// again, and which a moved surface only carries into space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldClass {
    Coordinate,
    Embedding,
    Function,
}

raw_enum!(
    /// An axis of the domain: the real or the imaginary.
    DomainAxis { Real = "real", Imag = "imag" }
);

impl DomainAxis {
    /// The angle of a line along the axis, in radians from the real axis.
    pub fn angle(self) -> f64 {
        if self == DomainAxis::Real { 0.0 } else { core::f64::consts::FRAC_PI_2 }
    }

    /// The axis a line at `angle` runs along, to within
    /// `Hatching::ANGLE_TOLERANCE`, or `None` when it runs along neither.
    pub fn from_angle(angle: f64) -> Option<DomainAxis> {
        DomainAxis::ALL.iter().copied().find(|a| (angle - a.angle()).abs() <= Hatching::ANGLE_TOLERANCE)
    }
}

/// The part of the domain a contour is restricted to.
///
/// A small closed vocabulary, not an expression language: inside the
/// occluder's footprint, below the truncation, within a band of one axis,
/// and intersections of these. It can be evaluated without an interpreter.
/// Curves that need more than this should be stored, and `LayerSource::File`
/// is how the schema says so.
#[derive(Debug, Clone, PartialEq)]
pub enum Restriction {
    All,
    /// Inside `occluder.region`: the same polygon the heightfield is cut to,
    /// referenced rather than repeated.
    Footprint,
    /// Where |f| is below the truncation: membership in the graph, off the
    /// plateaus. It also carries a contour of |f| or arg f that leaves over
    /// the truncation to the truncation edge (`Surface.derive`).
    BelowTruncation,
    Band {
        axis: DomainAxis,
        lo: f64,
        hi: f64,
    },
    Intersection(Vec<Restriction>),
}

impl Restriction {
    /// Whether lying below the truncation is among the conditions.
    pub fn requires_below_truncation(&self) -> bool {
        match self {
            Restriction::BelowTruncation => true,
            Restriction::Intersection(all) => all.iter().any(Restriction::requires_below_truncation),
            Restriction::All | Restriction::Footprint | Restriction::Band { .. } => false,
        }
    }

    /// Whether lying in the footprint is among the conditions.
    pub fn requires_footprint(&self) -> bool {
        match self {
            Restriction::Footprint => true,
            Restriction::Intersection(all) => all.iter().any(Restriction::requires_footprint),
            Restriction::All | Restriction::BelowTruncation | Restriction::Band { .. } => false,
        }
    }

    fn from_json(json: &JsonValue) -> Result<Restriction, SchemaError> {
        let o = json.object("Restriction")?;
        let kind = o.string("kind", "Restriction")?;
        Ok(match kind.as_str() {
            "all" => Restriction::All,
            "footprint" => Restriction::Footprint,
            "belowTruncation" => Restriction::BelowTruncation,
            "band" => {
                let axis_text = o.string("axis", "Restriction.band")?;
                let axis = DomainAxis::parse(&axis_text).ok_or_else(|| {
                    ManifestError::unknown_kind(&axis_text, "Restriction.band axis", &DomainAxis::names())
                })?;
                Restriction::Band {
                    axis,
                    lo: o.double("lo", "Restriction.band")?,
                    hi: o.double("hi", "Restriction.band")?,
                }
            }
            "intersection" => Restriction::Intersection(
                o.array("of", "Restriction.intersection")?
                    .iter()
                    .map(Restriction::from_json)
                    .collect::<Result<_, _>>()?,
            ),
            other => {
                return Err(ManifestError::unknown_kind(
                    other,
                    "Restriction",
                    &["all", "footprint", "belowTruncation", "band", "intersection"],
                )
                .into());
            }
        })
    }

    pub fn json(&self) -> JsonValue {
        match self {
            Restriction::All => object([("kind", string("all"))]),
            Restriction::Footprint => object([("kind", string("footprint"))]),
            Restriction::BelowTruncation => object([("kind", string("belowTruncation"))]),
            Restriction::Band { axis, lo, hi } => object([
                ("kind", string("band")),
                ("axis", string(axis.raw())),
                ("lo", JsonValue::Double(*lo)),
                ("hi", JsonValue::Double(*hi)),
            ]),
            Restriction::Intersection(of) => object([
                ("kind", string("intersection")),
                ("of", JsonValue::Array(of.iter().map(Restriction::json).collect())),
            ]),
        }
    }
}

/// The geometry a manifest holds, which decides the layers it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    Landscape,
    Surface,
    Polyhedron,
}

/// Where a layer's geometry comes from: a file, or a description.
///
/// A dumped layer is the answer; a described one is the question. Describing it
/// is what lets a consumer move the levels and get a new answer, and it is what
/// shrinks a bundle from megabytes of vertices to a list of numbers.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerSource {
    File {
        vertices: String,
        offsets: String,
    },
    /// Level sets of `field` at `levels` on `faces`, restricted by
    /// `restriction`, and replicated once per occluder tile where the faces
    /// are `tiled`. |f| and arg f lie on the graph and are lifted by the
    /// layer's height policy, over the truncation onto a plateau unless the
    /// restriction keeps them below it; a coordinate or the height lies on
    /// its face, where the face puts it
    /// (`Surface.contours(of:_:on:restriction:in:refine:light:)`). `light`,
    /// the direction toward a light, is an isophote's, and only an
    /// isophote's: not normalized, as a half-Lambert tone keeps its own.
    Contour {
        field: ContourField,
        levels: Vec<f64>,
        restriction: Restriction,
        faces: FaceSelector,
        light: Option<Vec3>,
    },
    /// Hatching (`Hatching`) laid on `faces`: a landscape's walls, risers
    /// or plateaus, or every face of a solid, which a polyhedron's manifest
    /// takes (`PolyhedronManifest`). The landscape's hatchings are
    /// descriptions rather than stored curves for the same reason contours
    /// are: dragging the truncation moves every plateau, and a plateau that
    /// moves is a different set of hatch lines. `Hatch.swift` derives them,
    /// and `tests/fixtures/hatch` pins them.
    Hatching {
        hatching: Hatching,
        faces: FaceSelector,
    },
    /// The edges of the listed walls: each one's top and bottom edges, and a
    /// vertical edge at each distinct corner, its two ends.
    WallEdges {
        edges: Vec<i64>,
        base: f64,
    },
    /// The truncation edge: where the graph meets the plateaus, the level
    /// set |f| = h(x).
    TruncationEdge {
        tiled: bool,
    },
    /// The coordinate curves of a parametric surface: `u` curves along which
    /// u is constant and `v` along which v is, evenly spaced. Each vertex
    /// carries its surface coordinate, so visibility is tested where on the
    /// surface it lies.
    CoordinateCurves {
        u: i64,
        v: i64,
    },
    /// Linear curves on a torus: `count` straight lines through its
    /// parameter rectangle at `slope` turns of `v` per turn of `u`, each
    /// drawn `turns` turns round -- or until it closes, when the slope is
    /// rational and it is a (p, q) torus curve. Each vertex carries its
    /// surface coordinate.
    LinearCurves {
        slope: f64,
        turns: i64,
        count: i64,
    },
    /// The fold curves of a parametric surface, where it turns edge-on to
    /// the projection direction; their image is the outline. They depend on
    /// the camera and are derived for it.
    FoldCurves,
    /// A trajectory of a flow, integrated and carried with its coordinates
    /// on the invariant surface it winds round.
    Trajectory,
    /// Integral curves of `field` on `faces`, one through each of `seeds`,
    /// traced both ways from it
    /// (`Face.integralCurve(of:through:on:limits:returningTo:)`) and
    /// replicated once per occluder tile where the faces are `tiled`. Slope
    /// lines lie on the graph, lifted to |f|
    /// (`Surface.integralCurves(of:seeds:on:in:refine:)`); lines of
    /// curvature on a parametric surface, each once round where it does not
    /// close (`ParametricSurface.integralCurves(of:seeds:limits:)`).
    /// `LinearCurves` and `Trajectory` are integral curves too, of a
    /// constant field on a torus, in closed form, and keep their own kinds,
    /// as D2 keeps the special-case kinds: defined by that expansion in code
    /// (`VectorField.constant`).
    IntegralCurves {
        field: DirectionField,
        seeds: Seeds,
        faces: FaceSelector,
    },
    /// The edges of a solid, tested per edge by its two faces and cut
    /// exactly where something else hides them (`Solid.visibleEdges`).
    Edges,
}

impl LayerSource {
    /// A layer of a landscape's manifest.
    pub fn from_json(json: &JsonValue) -> Result<LayerSource, SchemaError> {
        LayerSource::from_json_on(json, Geometry::Landscape)
    }

    /// A layer of a manifest holding `geometry`. A landscape's takes any but
    /// the faces of another geometry. A parametric surface's
    /// (`SurfaceManifest`) takes a contour on its one face, its coordinate
    /// curves, linear curves, fold curves or a trajectory; which fields and
    /// restrictions the surface admits its request decides. A polyhedron's
    /// (`PolyhedronManifest`) takes its edges and hatching on its faces.
    pub fn from_json_on(json: &JsonValue, geometry: Geometry) -> Result<LayerSource, SchemaError> {
        let o = json.object("LayerSource")?;
        let kind = o.string("kind", "LayerSource")?;
        let on_surface = geometry == Geometry::Surface;
        let unsupported = |why: String| -> SchemaError { ManifestError::Unsupported(why).into() };
        if geometry == Geometry::Polyhedron {
            match kind.as_str() {
                "edges" | "hatching" => {}
                "file" => {
                    return Err(unsupported(
                        "a dumped layer on a polyhedron: its edges and hatching are drawn from its faces, not stored"
                            .to_string(),
                    ));
                }
                "contour" => {
                    return Err(unsupported(
                        "a contour on a polyhedron's faces: a polyhedron draws its edges and hatching".to_string(),
                    ));
                }
                _ => {
                    return Err(unsupported(format!(
                        "a {kind} layer on a polyhedron: its curves are its edges and hatching on its faces"
                    )));
                }
            }
        }
        if on_surface {
            match kind.as_str() {
                "contour" | "coordinateCurves" | "linearCurves" | "foldCurves" | "trajectory" | "hatching"
                | "integralCurves" => {}
                "file" => {
                    return Err(unsupported(
                        "a dumped layer on a parametric surface: its curves are drawn from its request, not stored"
                            .to_string(),
                    ));
                }
                _ => {
                    return Err(unsupported(format!(
                        "a {kind} layer on a parametric surface: it is one face, with no walls, truncation edge or \
                         edges of a solid"
                    )));
                }
            }
        }
        Ok(match kind.as_str() {
            "file" => LayerSource::File {
                vertices: o.string("vertices", "LayerSource.file")?,
                offsets: o.string("offsets", "LayerSource.file")?,
            },
            "contour" => {
                let field = ContourField::parse_or_refuse(&o.string("field", "LayerSource.contour")?)?;
                let faces = if on_surface {
                    LayerSource::surface_faces(o, "LayerSource.contour")?
                } else {
                    LayerSource::landscape_faces(o, "LayerSource.contour")?
                };
                if let Some(why) = faces.unsupported_field(field) {
                    return Err(unsupported(why));
                }
                let restriction = Restriction::from_json(o.value("restriction", "LayerSource.contour")?)?;
                if on_surface && restriction != Restriction::All && restriction != Restriction::BelowTruncation {
                    return Err(unsupported(
                        "a contour on a parametric surface kept to a footprint or a band: the surface has neither"
                            .to_string(),
                    ));
                }
                if restriction.requires_below_truncation() && faces.kind() != FaceKind::Graph && !on_surface {
                    return Err(unsupported(format!(
                        "belowTruncation on the {}: it is membership in the graph, and holds nothing of another face",
                        faces.name()
                    )));
                }
                let light = if field == ContourField::Isophote {
                    let l = o.doubles("light", "LayerSource.contour")?;
                    if l.len() != 3 || !Tone::has_direction(Vec3::new(l[0], l[1], l[2])) {
                        return Err(JsonError::wrong_type(
                            "light",
                            "three numbers, not all zero",
                            "LayerSource.contour",
                        )
                        .into());
                    }
                    Some(Vec3::new(l[0], l[1], l[2]))
                } else {
                    if o.contains_key("light") {
                        return Err(unsupported(format!(
                            "a light on a contour of {}: only an isophote has one",
                            field.raw()
                        )));
                    }
                    None
                };
                LayerSource::Contour {
                    field,
                    levels: o.doubles("levels", "LayerSource.contour")?,
                    restriction,
                    faces,
                    light,
                }
            }
            "hatching" if geometry == Geometry::Polyhedron => {
                let faces = FaceSelector::from_json(o.value("faces", "LayerSource.hatching")?)?;
                if faces != FaceSelector::Solid {
                    return Err(unsupported(format!(
                        "hatching on the {} of a polyhedron: it is laid on the solid's faces, `solid`",
                        faces.name()
                    )));
                }
                let hatching = Hatching::from_fields(o)?;
                if let Some(why) = faces.unsupported_hatching(&hatching) {
                    return Err(unsupported(why));
                }
                LayerSource::Hatching { hatching, faces }
            }
            "hatching" if on_surface => {
                let faces = FaceSelector::from_json(o.value("faces", "LayerSource.hatching")?)?;
                if faces != FaceSelector::Surface {
                    return Err(unsupported(format!(
                        "hatching on the {} of a parametric surface: the surface is one face, `surface`",
                        faces.name()
                    )));
                }
                if o.contains_key("step") {
                    return Err(LayerSource::curved_step().into());
                }
                let hatching = Hatching::from_fields(o)?;
                if let Some(why) = faces.unsupported_hatching(&hatching) {
                    return Err(unsupported(why));
                }
                LayerSource::Hatching { hatching, faces }
            }
            "hatching" => {
                let faces = LayerSource::landscape_faces(o, "LayerSource.hatching")?;
                if o.contains_key("step") {
                    return Err(LayerSource::landscape_step().into());
                }
                let hatching = Hatching::from_fields(o)?;
                if let Some(why) = faces.unsupported_hatching(&hatching) {
                    return Err(unsupported(why));
                }
                LayerSource::Hatching { hatching, faces }
            }
            "wallEdges" => {
                if o.contains_key("step") {
                    return Err(LayerSource::wall_edges_step().into());
                }
                LayerSource::WallEdges {
                    edges: o.ints("edges", "LayerSource.wallEdges")?,
                    base: o.double("base", "LayerSource.wallEdges").unwrap_or(0.0),
                }
            }
            "truncationEdge" => {
                LayerSource::TruncationEdge { tiled: o.bool("tiled", "LayerSource.truncationEdge", Some(false))? }
            }
            "foldCurves" => LayerSource::FoldCurves,
            "trajectory" => LayerSource::Trajectory,
            "integralCurves" => {
                let faces = if on_surface {
                    LayerSource::surface_faces(o, "LayerSource.integralCurves")?
                } else {
                    LayerSource::landscape_faces(o, "LayerSource.integralCurves")?
                };
                let field = DirectionField::from_json(o.value("field", "LayerSource.integralCurves")?)?;
                let seeds = Seeds::from_json(o.value("seeds", "LayerSource.integralCurves")?)?;
                if let Some(why) = LayerSource::unsupported_integral_curves(&field, &seeds, &faces, None) {
                    return Err(unsupported(why));
                }
                LayerSource::IntegralCurves { field, seeds, faces }
            }
            "edges" => LayerSource::Edges,
            "coordinateCurves" => LayerSource::CoordinateCurves {
                u: o.int("u", "LayerSource.coordinateCurves")?,
                v: o.int("v", "LayerSource.coordinateCurves")?,
            },
            "linearCurves" => LayerSource::LinearCurves {
                slope: o.double("slope", "LayerSource.linearCurves")?,
                turns: o.int("turns", "LayerSource.linearCurves")?,
                count: o.int("count", "LayerSource.linearCurves")?,
            },
            other => {
                return Err(ManifestError::unknown_kind(
                    other,
                    "LayerSource",
                    &[
                        "file",
                        "contour",
                        "hatching",
                        "wallEdges",
                        "truncationEdge",
                        "coordinateCurves",
                        "linearCurves",
                        "foldCurves",
                        "trajectory",
                        "integralCurves",
                        "edges",
                    ],
                )
                .into());
            }
        })
    }

    /// Why integral curves of `field` from `seeds` cannot lie on `faces`, or
    /// `None` when they can. Slope lines lie on the graph alone so far, the
    /// slope of a field the graph exposes but an isophote, whose light they
    /// do not keep. Lines of curvature lie on a parametric surface alone,
    /// whose map has second derivatives in closed form: on the graph |f|'s
    /// would be its central differences differenced again. Seeds are placed
    /// by fields the faces expose, no closer than `cell` where it is known
    /// (`Seeds::unsupported`), and on a surface by its coordinates
    /// and its height, not by f's fields, which its map does not carry.
    pub fn unsupported_integral_curves(
        field: &DirectionField,
        seeds: &Seeds,
        faces: &FaceSelector,
        cell: Option<f64>,
    ) -> Option<String> {
        match field {
            DirectionField::Slope(ContourField::Isophote) => {
                if faces.kind() != FaceKind::Graph {
                    return Some(format!("slope lines on the {}: they lie on the graph so far", faces.name()));
                }
                return Some("the slope of an isophote: integral curves keep no light to read n · l by".to_string());
            }
            DirectionField::Slope(of) => {
                if faces.kind() != FaceKind::Graph {
                    return Some(format!("slope lines on the {}: they lie on the graph so far", faces.name()));
                }
                if let Some(why) = faces.unsupported_field(*of) {
                    return Some(why);
                }
            }
            DirectionField::PrincipalDirection(_) => {
                if *faces != FaceSelector::Surface {
                    return Some(format!(
                        "lines of curvature on the {}: they lie on a parametric surface, whose map has second \
                         derivatives in closed form",
                        faces.name()
                    ));
                }
                let mut fields = vec![seeds.field];
                if let Spacing::Values(of, _) = &seeds.spacing {
                    fields.push(*of);
                }
                if let Some(f) = fields.iter().find(|f| f.is_field_of_f()) {
                    return Some(format!(
                        "seeds placed by {} on a parametric surface: lines of curvature are seeded by its \
                         coordinates or its height",
                        f.raw()
                    ));
                }
            }
        }
        seeds.unsupported(faces, cell)
    }

    /// The face a contour on a parametric surface names: its one face.
    fn surface_faces(o: &JsonObject, what: &str) -> Result<FaceSelector, SchemaError> {
        let faces = FaceSelector::from_json(o.value("faces", what)?)?;
        if faces != FaceSelector::Surface {
            let layer = if what == "LayerSource.contour" { "a contour" } else { "integral curves" };
            return Err(ManifestError::Unsupported(format!(
                "{layer} on the {} of a parametric surface: the surface is one face, `surface`",
                faces.name()
            ))
            .into());
        }
        Ok(faces)
    }

    /// The faces a contour or a hatching in a manifest names: a landscape's.
    fn landscape_faces(o: &JsonObject, what: &str) -> Result<FaceSelector, SchemaError> {
        let faces = FaceSelector::from_json(o.value("faces", what)?)?;
        match faces {
            FaceSelector::Graph { .. }
            | FaceSelector::Plateaus { .. }
            | FaceSelector::Walls { .. }
            | FaceSelector::Risers { .. } => Ok(faces),
            FaceSelector::Surface | FaceSelector::Solid => Err(ManifestError::Unsupported(format!(
                "a layer on a {}'s faces: this manifest holds a landscape, with a graph, plateaus, walls and risers",
                faces.name()
            ))
            .into()),
        }
    }

    /// A wall's or a riser's hatch line is its two ends, and a plateau's has
    /// a vertex at each grid line, so there is no step along one to write
    /// (schema 3); nor along a wall's edges.
    fn landscape_step() -> ManifestError {
        ManifestError::Unsupported(
            "a step on a landscape's hatching: a line is its two ends on a wall or a riser, and has a vertex at \
             each grid line on a plateau, so schema 3 writes none"
                .to_string(),
        )
    }

    /// A hatch line on a curved face has a vertex wherever it crosses a
    /// sample of the face's parameter rectangle, as a contour of the
    /// coordinate across it does, and its tone is read there, so there is no
    /// step to write either.
    fn curved_step() -> ManifestError {
        ManifestError::Unsupported(
            "a step on a curved face's hatching: a line has a vertex wherever it crosses a sample of the face's \
             parameter rectangle, so schema 3 writes none"
                .to_string(),
        )
    }

    fn wall_edges_step() -> ManifestError {
        ManifestError::Unsupported(
            "a step on wall edges: an edge up a corner is its two ends, and schema 3 writes none".to_string(),
        )
    }

    pub fn json(&self) -> JsonValue {
        match self {
            LayerSource::File { vertices, offsets } => {
                object([("kind", string("file")), ("vertices", string(vertices)), ("offsets", string(offsets))])
            }
            LayerSource::Contour { field, levels, restriction, faces, light } => {
                let mut o = object([
                    ("kind", string("contour")),
                    ("field", string(field.raw())),
                    ("levels", json::doubles(levels)),
                    ("restriction", restriction.json()),
                    ("faces", faces.json()),
                ]);
                if let (Some(l), JsonValue::Object(m)) = (light, &mut o) {
                    m.insert("light".to_string(), json::doubles(&l.to_array()));
                }
                o
            }
            LayerSource::Hatching { hatching, faces } => {
                let mut m = hatching.fields();
                m.insert("kind".to_string(), string("hatching"));
                m.insert("faces".to_string(), faces.json());
                match faces {
                    FaceSelector::Solid => {}
                    _ => {
                        m.remove("step");
                    }
                }
                JsonValue::Object(m)
            }
            LayerSource::WallEdges { edges, base } => object([
                ("kind", string("wallEdges")),
                ("edges", json::ints(edges)),
                ("base", JsonValue::Double(*base)),
            ]),
            LayerSource::TruncationEdge { tiled } => {
                object([("kind", string("truncationEdge")), ("tiled", JsonValue::Bool(*tiled))])
            }
            LayerSource::CoordinateCurves { u, v } => {
                object([("kind", string("coordinateCurves")), ("u", JsonValue::Int(*u)), ("v", JsonValue::Int(*v))])
            }
            LayerSource::LinearCurves { slope, turns, count } => object([
                ("kind", string("linearCurves")),
                ("slope", JsonValue::Double(*slope)),
                ("turns", JsonValue::Int(*turns)),
                ("count", JsonValue::Int(*count)),
            ]),
            LayerSource::FoldCurves => object([("kind", string("foldCurves"))]),
            LayerSource::Trajectory => object([("kind", string("trajectory"))]),
            LayerSource::IntegralCurves { field, seeds, faces } => object([
                ("kind", string("integralCurves")),
                ("field", field.json()),
                ("seeds", seeds.json()),
                ("faces", faces.json()),
            ]),
            LayerSource::Edges => object([("kind", string("edges"))]),
        }
    }

    /// Vertical hatching on the walls along `edges`, a line every `spacing`
    /// along each wall, each line its two ends. (Swift's defaults: no end
    /// lines, base and top offset 0.)
    pub fn wall_hatching(edges: Vec<i64>, spacing: f64, end_lines: bool, base: f64, top_offset: f64) -> LayerSource {
        LayerSource::Hatching {
            hatching: Hatching::plain(FaceSelector::WALL_ANGLE, spacing),
            faces: FaceSelector::Walls { edges, end_lines, base, top_offset },
        }
    }

    /// Vertical hatching on the risers, as on the walls: a line every
    /// `spacing` along each, from its bottom edge to its top.
    pub fn riser_hatching(spacing: f64, tiled: bool, end_lines: bool) -> LayerSource {
        LayerSource::Hatching {
            hatching: Hatching::plain(FaceSelector::WALL_ANGLE, spacing),
            faces: FaceSelector::Risers { tiled, end_lines },
        }
    }

    /// Hatching on the plateaus along one axis of the domain, a line every
    /// `spacing` across the other.
    pub fn plateau_hatching(axis: DomainAxis, spacing: f64, tiled: bool) -> LayerSource {
        LayerSource::Hatching {
            hatching: Hatching::plain(axis.angle(), spacing),
            faces: FaceSelector::Plateaus { tiled },
        }
    }

    /// Its kind as a manifest names it, an isophote and a half-Lambert tone
    /// by what they are.
    pub fn kind(&self) -> &'static str {
        match self {
            LayerSource::File { .. } => "file",
            LayerSource::Contour { field: ContourField::Isophote, .. } => "isophotes",
            LayerSource::Contour { .. } => "contour",
            LayerSource::Hatching { hatching, .. } => {
                if let Tone::HalfLambert { .. } = hatching.tone {
                    "half-Lambert tone"
                } else {
                    "hatching"
                }
            }
            LayerSource::WallEdges { .. } => "wallEdges",
            LayerSource::TruncationEdge { .. } => "truncationEdge",
            LayerSource::CoordinateCurves { .. } => "coordinateCurves",
            LayerSource::LinearCurves { .. } => "linearCurves",
            LayerSource::FoldCurves => "foldCurves",
            LayerSource::Trajectory => "trajectory",
            LayerSource::IntegralCurves { field: DirectionField::Slope(_), .. } => "slope lines",
            LayerSource::IntegralCurves { field: DirectionField::PrincipalDirection(_), .. } => "lines of curvature",
            LayerSource::Edges => "edges",
        }
    }

    /// The direction toward the light the layer is read under: an
    /// isophote's, or a hatching's half-Lambert tone's. `None` for any other.
    pub fn light(&self) -> Option<Vec3> {
        match self {
            LayerSource::Contour { field: ContourField::Isophote, light, .. } => *light,
            LayerSource::Hatching { hatching, .. } => match hatching.tone {
                Tone::HalfLambert { light } => Some(light),
                _ => None,
            },
            _ => None,
        }
    }

    /// The same layer under a light toward `light`, or `None` for a layer read
    /// under none (`light`).
    pub fn lit(&self, light: Vec3) -> Option<LayerSource> {
        match self {
            LayerSource::Contour { field: ContourField::Isophote, levels, restriction, faces, light: Some(_) } => {
                Some(LayerSource::Contour {
                    field: ContourField::Isophote,
                    levels: levels.clone(),
                    restriction: restriction.clone(),
                    faces: faces.clone(),
                    light: Some(light),
                })
            }
            LayerSource::Hatching { hatching, faces } => {
                let Tone::HalfLambert { .. } = hatching.tone else { return None };
                let mut hatching = *hatching;
                hatching.tone = Tone::HalfLambert { light };
                Some(LayerSource::Hatching { hatching, faces: faces.clone() })
            }
            _ => None,
        }
    }

    /// The axis plateau hatching runs along, or `None` for any other source.
    pub fn plateau_hatch_axis(&self) -> Option<DomainAxis> {
        match self {
            LayerSource::Hatching { hatching, faces: FaceSelector::Plateaus { .. } } => {
                DomainAxis::from_angle(hatching.angle)
            }
            _ => None,
        }
    }

    /// True when the geometry is a rule rather than a file -- the layers whose
    /// parameters a consumer can edit and re-derive.
    pub fn is_derived(&self) -> bool {
        !matches!(self, LayerSource::File { .. })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    pub name: String,
    pub role: LayerRole,
    pub source: LayerSource,
    /// A matplotlib line width in points, carried into the SVG stroke width.
    pub width: f64,
    pub height_policy: HeightPolicy,
    pub color: String,
    /// False for curves that lie *in* the occluding geometry -- a cut-face hatch,
    /// which a depth test would half erase.
    pub clipped: bool,
}

impl LayerSpec {
    /// A layer in black, clipped: Swift's defaults.
    pub fn new(name: &str, role: LayerRole, source: LayerSource, width: f64, height_policy: HeightPolicy) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            role,
            source,
            width,
            height_policy,
            color: "#000000".to_string(),
            clipped: true,
        }
    }

    /// `(vertices, offsets)` when this layer is dumped, else `None`.
    pub fn files(&self) -> Option<(&str, &str)> {
        match &self.source {
            LayerSource::File { vertices, offsets } => Some((vertices, offsets)),
            _ => None,
        }
    }

    pub fn from_json(json: &JsonValue) -> Result<LayerSpec, SchemaError> {
        LayerSpec::from_json_on(json, Geometry::Landscape)
    }

    /// A layer of a manifest holding `geometry` (`LayerSource::from_json_on`).
    pub fn from_json_on(json: &JsonValue, geometry: Geometry) -> Result<LayerSpec, SchemaError> {
        let o = json.object("LayerSpec")?;
        let role_text = o.string("role", "LayerSpec")?;
        let role = LayerRole::parse(&role_text)
            .ok_or_else(|| ManifestError::unknown_kind(&role_text, "LayerRole", &LayerRole::names()))?;
        let policy_text = o.string("heightPolicy", "LayerSpec")?;
        let height_policy = HeightPolicy::parse(&policy_text)
            .ok_or_else(|| ManifestError::unknown_kind(&policy_text, "HeightPolicy", &HeightPolicy::names()))?;
        let width = o.double("width", "LayerSpec")?;
        // A polyhedron's manifest refuses one in its own words
        // (`PolyhedronManifest::unsupported`).
        if !(geometry == Geometry::Polyhedron || (width.is_finite() && width >= 0.0)) {
            return Err(ManifestError::Invalid(format!(
                "layer '{}': a width must be a number not below 0, not {}",
                o.string("name", "LayerSpec").unwrap_or_default(),
                text::double(width)
            ))
            .into());
        }
        Ok(LayerSpec {
            name: o.string("name", "LayerSpec")?,
            role,
            source: LayerSource::from_json_on(o.value("source", "LayerSpec")?, geometry)?,
            width,
            height_policy,
            color: o.string("color", "LayerSpec").unwrap_or_else(|_| "#000000".to_string()),
            clipped: o.bool("clipped", "LayerSpec", Some(true))?,
        })
    }

    pub fn json(&self) -> JsonValue {
        object([
            ("name", string(&self.name)),
            ("role", string(self.role.raw())),
            ("source", self.source.json()),
            ("width", JsonValue::Double(self.width)),
            ("heightPolicy", string(self.height_policy.raw())),
            ("color", string(&self.color)),
            ("clipped", JsonValue::Bool(self.clipped)),
        ])
    }
}

// MARK: - camera

/// The Python `Projection` parameter set, verbatim.
///
/// Column 0 of the array `Projection.apply` consumes is imag and column 1 is
/// real, so `flipX` flips imag and `yScale` scales real. `Camera::oblique` folds
/// the axis exchange into one matrix; these numbers cross unchanged so the two
/// cameras can be compared numerically instead of by inspection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObliqueProjection {
    pub shear: f64,
    pub x_angle: f64,
    pub z_angle: f64,
    pub flip_x: bool,
    pub y_scale: Option<f64>,
}

impl ObliqueProjection {
    /// The plates' three projections, as Swift's camera golden set
    /// (`Fixtures.swift`) and the navigation suites name them.
    pub const PLATES: [(&'static str, ObliqueProjection); 3] = [
        ("recip", ObliqueProjection { shear: 0.5, x_angle: -55.0, z_angle: -90.0, flip_x: true, y_scale: None }),
        ("elliptic", ObliqueProjection { shear: 0.51, x_angle: -63.0, z_angle: -90.0, flip_x: true, y_scale: None }),
        ("zeta", ObliqueProjection { shear: -0.18, x_angle: -79.5, z_angle: -90.0, flip_x: true, y_scale: Some(0.75) }),
    ];

    pub fn from_json(json: &JsonValue) -> Result<ObliqueProjection, SchemaError> {
        let o = json.object("ObliqueProjection")?;
        Ok(ObliqueProjection {
            shear: o.double("shear", "ObliqueProjection")?,
            x_angle: o.double("xAngle", "ObliqueProjection")?,
            z_angle: o.double("zAngle", "ObliqueProjection")?,
            flip_x: o.bool("flipX", "ObliqueProjection", None)?,
            y_scale: o.optional_double("yScale"),
        })
    }
    pub fn json(&self) -> JsonValue {
        object([
            ("shear", JsonValue::Double(self.shear)),
            ("xAngle", JsonValue::Double(self.x_angle)),
            ("zAngle", JsonValue::Double(self.z_angle)),
            ("flipX", JsonValue::Bool(self.flip_x)),
            ("yScale", self.y_scale.map_or(JsonValue::Null, JsonValue::Double)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CameraPreset {
    pub name: String,
    pub projection: ObliqueProjection,
    /// How far in view depth a curve may lie behind the surface and still be
    /// drawn: the hidden-line test's tolerance.
    pub depth_tolerance: f64,
    /// The depth buffer's resolution, in pixels along the longer side.
    pub buffer: i64,
}

impl CameraPreset {
    /// The finest depth buffer a preset asks for, along its longer side:
    /// gamma's published bake, whose stitched buffer is 1.6 GB.
    pub const MAX_BUFFER: i64 = 20_000;

    /// Why `buffer` is no depth buffer's resolution, or `None` when it is one:
    /// at least a pixel, at most `MAX_BUFFER`.
    pub fn unsupported_buffer(buffer: i64) -> Option<String> {
        (!(1..=CameraPreset::MAX_BUFFER).contains(&buffer))
            .then(|| format!("must be 1 to {} pixels, not {buffer}", CameraPreset::MAX_BUFFER))
    }

    fn from_json(json: &JsonValue) -> Result<CameraPreset, SchemaError> {
        let o = json.object("CameraPreset")?;
        let buffer = o.int("buffer", "CameraPreset")?;
        if let Some(why) = CameraPreset::unsupported_buffer(buffer) {
            return Err(ManifestError::Invalid(format!("a preset's buffer {why}")).into());
        }
        Ok(CameraPreset {
            name: o.string("name", "CameraPreset")?,
            projection: ObliqueProjection::from_json(o.value("projection", "CameraPreset")?)?,
            depth_tolerance: o.double("depthTolerance", "CameraPreset")?,
            buffer,
        })
    }
    fn json(&self) -> JsonValue {
        object([
            ("name", string(&self.name)),
            ("projection", self.projection.json()),
            ("depthTolerance", JsonValue::Double(self.depth_tolerance)),
            ("buffer", JsonValue::Int(self.buffer)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    pub function: String,
    /// The recipe that produced this: "function" for a native landscape,
    /// whose request `params` records, so it can be sampled again; "fixture"
    /// for a golden fixture. `function` names the mathematics.
    pub example: String,
    pub params: JsonObject,
    /// The contourpy chunk count the bundle was written with. Anything but 1
    /// means the contours were stitched in thread-completion order and the
    /// bundle is not reproducible -- worth knowing before blaming a diff on
    /// anything else.
    pub cpu_count: i64,
    pub git_sha: String,
}

impl Provenance {
    fn from_json(json: &JsonValue) -> Result<Provenance, SchemaError> {
        let o = json.object("Provenance")?;
        Ok(Provenance {
            function: o.string("function", "Provenance")?,
            example: o.string("example", "Provenance").unwrap_or_default(),
            params: o.value("params", "Provenance")?.object("Provenance.params")?.clone(),
            cpu_count: o.int("cpuCount", "Provenance")?,
            git_sha: o.string("gitSha", "Provenance")?,
        })
    }
    fn json(&self) -> JsonValue {
        object([
            ("function", string(&self.function)),
            ("example", string(&self.example)),
            ("params", JsonValue::Object(self.params.clone())),
            ("cpuCount", JsonValue::Int(self.cpu_count)),
            ("gitSha", string(&self.git_sha)),
        ])
    }

    pub fn is_reproducible(&self) -> bool {
        self.cpu_count == 1
    }
}

fn find_preset<'a>(presets: &'a [CameraPreset], name: &str) -> Result<&'a CameraPreset, ManifestError> {
    presets.iter().find(|p| p.name == name).ok_or_else(|| ManifestError::NoSuchPreset {
        name: name.to_string(),
        have: presets.iter().map(|p| p.name.clone()).collect(),
    })
}

fn layers_from(list: &[JsonValue], geometry: Geometry) -> Result<Vec<LayerSpec>, SchemaError> {
    list.iter().map(|l| LayerSpec::from_json_on(l, geometry)).collect()
}

fn presets_from(list: &[JsonValue]) -> Result<Vec<CameraPreset>, SchemaError> {
    list.iter().map(CameraPreset::from_json).collect()
}

// MARK: - the manifest

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub schema: i64,
    pub axes: Vec<String>,
    pub domain: Domain,
    pub height: GridRef,
    pub phase: Option<GridRef>,
    pub truncation: Truncation,
    pub occluder: Occluder,
    pub layers: Vec<LayerSpec>,
    pub presets: Vec<CameraPreset>,
    pub provenance: Provenance,
}

impl Manifest {
    #[allow(clippy::too_many_arguments)] // Swift's initializer, field for field
    pub fn new(
        domain: Domain,
        height: GridRef,
        phase: Option<GridRef>,
        truncation: Truncation,
        occluder: Occluder,
        layers: Vec<LayerSpec>,
        presets: Vec<CameraPreset>,
        provenance: Provenance,
    ) -> Manifest {
        Manifest {
            schema: KURVEN_SCHEMA,
            axes: kurven_axes(),
            domain,
            height,
            phase,
            truncation,
            occluder,
            layers,
            presets,
            provenance,
        }
    }

    pub fn from_json(json: &JsonValue) -> Result<Manifest, SchemaError> {
        let o = json.object("Manifest")?;
        let schema = o.int("schema", "Manifest")?;
        if schema != KURVEN_SCHEMA {
            return Err(ManifestError::UnsupportedSchema { schema, supported: KURVEN_SCHEMA }.into());
        }
        if o.contains_key("surface") {
            return Err(ManifestError::Unsupported(
                "a parametric surface, not a landscape: its bundle is built from its request (`SurfaceManifest`, \
                 `SurfaceBundle`)"
                    .to_string(),
            )
            .into());
        }
        if o.contains_key("polyhedron") {
            return Err(ManifestError::Unsupported(
                "a polyhedron, not a landscape: its bundle is its vertices and loops (`PolyhedronManifest`, \
                 `PolyhedronBundle`)"
                    .to_string(),
            )
            .into());
        }
        let axes = o.strings("axes", "Manifest")?;
        if axes != kurven_axes() {
            return Err(ManifestError::UnexpectedAxes { axes, expected: kurven_axes() }.into());
        }
        let domain = Domain::from_json(o.value("domain", "Manifest")?)?;
        let height = GridRef::from_json(o.value("height", "Manifest")?)?;
        let phase = match o.get("phase") {
            Some(v @ JsonValue::Object(_)) => Some(GridRef::from_json(v)?),
            _ => None,
        };
        let truncation = Truncation::from_json(o.value("truncation", "Manifest")?)?;
        let occluder = Occluder::from_json(o.value("occluder", "Manifest")?)?;
        let layers = layers_from(o.array("layers", "Manifest")?, Geometry::Landscape)?;
        let presets = presets_from(o.array("presets", "Manifest")?)?;
        let provenance = Provenance::from_json(o.value("provenance", "Manifest")?)?;
        for layer in &layers {
            if let Some(why) = Manifest::unsupported(&layer.source, &domain, height.shape) {
                return Err(ManifestError::Unsupported(format!("layer '{}': {why}", layer.name)).into());
            }
        }
        Ok(Manifest { schema, axes, domain, height, phase, truncation, occluder, layers, presets, provenance })
    }

    /// Why a layer cannot be drawn on a landscape over `domain` sampled on a
    /// grid of `shape` `(ny, nx)`, beyond what the layer says of itself:
    /// integral curves seeded closer than a cell apart along their level set,
    /// which asks for more curves than the grid resolves.
    pub fn unsupported(source: &LayerSource, domain: &Domain, shape: (i64, i64)) -> Option<String> {
        let LayerSource::IntegralCurves { field, seeds, faces } = source else { return None };
        let cell = (domain.real.length().abs() / (shape.1 - 1).max(1) as f64)
            .max(domain.imag.length().abs() / (shape.0 - 1).max(1) as f64);
        LayerSource::unsupported_integral_curves(field, seeds, faces, Some(cell))
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Manifest, SchemaError> {
        Manifest::from_json(&read_json(path.as_ref())?)
    }

    pub fn json(&self) -> JsonValue {
        object([
            ("schema", JsonValue::Int(self.schema)),
            ("axes", JsonValue::Array(self.axes.iter().map(|a| string(a)).collect())),
            ("domain", self.domain.json()),
            ("height", self.height.json()),
            ("phase", self.phase.as_ref().map_or(JsonValue::Null, GridRef::json)),
            ("truncation", self.truncation.json()),
            ("occluder", self.occluder.json()),
            ("layers", JsonValue::Array(self.layers.iter().map(LayerSpec::json).collect())),
            ("presets", JsonValue::Array(self.presets.iter().map(CameraPreset::json).collect())),
            ("provenance", self.provenance.json()),
        ])
    }

    /// Canonical JSON with the trailing newline `Manifest.to_json` writes, so
    /// the contract test compares whole files.
    pub fn canonical_json(&self) -> String {
        self.json().canonical() + "\n"
    }

    pub fn preset(&self, name: &str) -> Result<&CameraPreset, ManifestError> {
        find_preset(&self.presets, name)
    }

    pub fn layer(&self, name: &str) -> Result<&LayerSpec, ManifestError> {
        self.layers.iter().find(|l| l.name == name).ok_or_else(|| ManifestError::NoSuchLayer {
            name: name.to_string(),
            have: self.layers.iter().map(|l| l.name.clone()).collect(),
        })
    }
}

// MARK: - a parametric surface's manifest

const OTHER_GEOMETRY_KEYS: [&str; 6] = ["axes", "domain", "height", "phase", "truncation", "occluder"];

/// A parametric surface's manifest: the request that builds the surface
/// (`SurfaceRequest`), the curves drawn on it, its camera presets and where it
/// came from. The map is code, so the bundle stores the request rather than
/// samples of it, and is this file alone: no arrays, and no axes to state.
///
/// Its layers lie on the surface's one face: contours of its coordinates and
/// its height, and of |f| and arg f where the request paints a function
/// (kept below the truncation where it truncates one, as sn's relief does);
/// coordinate curves, linear curves, the fold curves, a forced torus's
/// trajectory, and its lines of curvature (`DirectionField::PrincipalDirection`)
/// but in relief. They are the drawing as it was kept, restyled or not, and
/// a reader draws each on the surface the request builds.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceManifest {
    pub schema: i64,
    pub surface: SurfaceRequest,
    pub layers: Vec<LayerSpec>,
    pub presets: Vec<CameraPreset>,
    pub provenance: Provenance,
}

impl SurfaceManifest {
    pub fn new(
        surface: SurfaceRequest,
        layers: Vec<LayerSpec>,
        presets: Vec<CameraPreset>,
        provenance: Provenance,
    ) -> SurfaceManifest {
        SurfaceManifest { schema: KURVEN_SCHEMA, surface, layers, presets, provenance }
    }

    pub fn from_json(json: &JsonValue) -> Result<SurfaceManifest, SchemaError> {
        let o = json.object("SurfaceManifest")?;
        let schema = o.int("schema", "SurfaceManifest")?;
        if schema != KURVEN_SCHEMA {
            return Err(ManifestError::UnsupportedSchema { schema, supported: KURVEN_SCHEMA }.into());
        }
        if let Some(key) = OTHER_GEOMETRY_KEYS.iter().find(|k| o.contains_key(**k)) {
            return Err(ManifestError::Unsupported(format!(
                "a parametric surface with a landscape's {key}: a manifest holds one geometry, a landscape, a \
                 parametric surface or a polyhedron"
            ))
            .into());
        }
        if o.contains_key("polyhedron") {
            return Err(ManifestError::Unsupported(
                "a parametric surface with a polyhedron: a manifest holds one geometry, a landscape, a parametric \
                 surface or a polyhedron"
                    .to_string(),
            )
            .into());
        }
        let surface = SurfaceRequest::from_json(o.value("surface", "SurfaceManifest")?)?;
        let layers = layers_from(o.array("layers", "SurfaceManifest")?, Geometry::Surface)?;
        let presets = presets_from(o.array("presets", "SurfaceManifest")?)?;
        let provenance = Provenance::from_json(o.value("provenance", "SurfaceManifest")?)?;
        let mut names = HashSet::new();
        for layer in &layers {
            if let Some(why) = SurfaceManifest::unsupported(&layer.source, &surface) {
                return Err(ManifestError::Unsupported(format!("layer '{}': {why}", layer.name)).into());
            }
            // A layer is kept across a request edit, and its slider found,
            // by its name.
            if !names.insert(layer.name.clone()) {
                return Err(ManifestError::Unsupported(format!(
                    "two layers named '{}': a parametric surface's layers are told apart by name",
                    layer.name
                ))
                .into());
            }
        }
        Ok(SurfaceManifest { schema, surface, layers, presets, provenance })
    }

    /// Why a layer cannot lie on the surface `request` builds, or `None` when
    /// it can: |f| and arg f where no function is painted, a contour kept
    /// below a truncation the function does not have, a trajectory on a
    /// torus no flow winds round, lines of curvature on sn in relief, whose
    /// map has no second derivatives in closed form.
    pub fn unsupported(source: &LayerSource, request: &SurfaceRequest) -> Option<String> {
        use crate::surface_request::{LinearCurves, Shape};
        // n · l needs the surface's outward side, which a surface bounding
        // no solid has not.
        if source.light().is_some() && request.bounds_solid() == Some(false) {
            return Some(format!("{} on a surface that bounds no solid, which has no outward normal", source.kind()));
        }
        match source {
            LayerSource::Contour { field: ContourField::Isophote, light: None, .. } => {
                Some("an isophote with no light to read n · l by".to_string())
            }
            LayerSource::Contour { field, restriction, .. } => {
                if field.is_field_of_f() && !request.paints_function() {
                    return Some(format!("a contour of {} on a surface that paints no function", field.raw()));
                }
                if *restriction == Restriction::BelowTruncation
                    && !(field.is_field_of_f() && request.truncates_function())
                {
                    return Some(
                        "a contour kept below a truncation: only |f| and arg f are truncated, and only in relief"
                            .to_string(),
                    );
                }
                None
            }
            LayerSource::Trajectory => {
                if let Shape::Forced { .. } = request.shape {
                    return None;
                }
                Some("a trajectory on a surface no flow winds round".to_string())
            }
            LayerSource::CoordinateCurves { u, v } => {
                if *u < 0 || *v < 0 {
                    return Some(format!("coordinate curves must be counts, not {u} and {v}"));
                }
                SurfaceRequest::vertices_of_coordinate_curves((*u, *v), request.line_resolution())
                    .map(|why| format!("a layer that {why}"))
            }
            LayerSource::LinearCurves { slope, turns, count } => request
                .vertices_of_linear_curves(&LinearCurves { slope: *slope, turns: *turns, count: *count })
                .map(|why| format!("a linear curve that {why}")),
            LayerSource::IntegralCurves { field, seeds, faces } => {
                if let Some(why) = LayerSource::unsupported_integral_curves(field, seeds, faces, None) {
                    return Some(why);
                }
                if let Shape::Relief { .. } = request.shape {
                    return Some(
                        "lines of curvature on sn in relief: its map is displaced by central differences, with no \
                         second derivatives in closed form, and creased where it is truncated"
                            .to_string(),
                    );
                }
                None
            }
            _ => None,
        }
    }

    pub fn json(&self) -> JsonValue {
        object([
            ("schema", JsonValue::Int(self.schema)),
            ("surface", self.surface.json()),
            ("layers", JsonValue::Array(self.layers.iter().map(LayerSpec::json).collect())),
            ("presets", JsonValue::Array(self.presets.iter().map(CameraPreset::json).collect())),
            ("provenance", self.provenance.json()),
        ])
    }

    /// Canonical JSON with a trailing newline, as a landscape's manifest writes.
    pub fn canonical_json(&self) -> String {
        self.json().canonical() + "\n"
    }

    pub fn preset(&self, name: &str) -> Result<&CameraPreset, ManifestError> {
        find_preset(&self.presets, name)
    }
}

// MARK: - a polyhedron's manifest

/// A polyhedron's manifest: its vertices and its faces' loops, as
/// `Solid::new` takes them, the curves drawn on it, its camera
/// presets and where it came from. A polyhedron is stored as its geometry,
/// not as the request that built it: the vertices and loops are its exact
/// boundary representation, they are small, and they hold solids no named
/// kind builds. Like a parametric surface's, the bundle is this file alone.
///
/// Its layers are its edges and hatching on its faces (`FaceSelector::Solid`).
/// The geometry is checked as the solid is built (`Solid::checking`),
/// and everything that drives work is bounded: the vertices, the corners of a
/// face and of all of them, the layers, and what the hatchings lay
/// (`unsupported`), so no manifest traps or costs without bound.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyhedronManifest {
    pub schema: i64,
    pub solid: Solid,
    pub layers: Vec<LayerSpec>,
    pub presets: Vec<CameraPreset>,
    pub provenance: Provenance,
}

impl PolyhedronManifest {
    /// The most vertices a polyhedron's manifest holds.
    pub const MAX_VERTICES: usize = 1 << 12;
    /// The most corners one face's loops have together, and all the faces'.
    /// A face is cut into convex pieces in time quadratic in its corners,
    /// and the edges are tested against every piece.
    pub const MAX_FACE_CORNERS: usize = 1 << 10;
    pub const MAX_CORNERS: usize = 1 << 14;
    /// The most layers: each is drawn for every camera.
    pub const MAX_LAYERS: usize = 64;
    /// The most hatch lines the hatchings lay together, the most times they
    /// read their tones, and the most times a hatch line changes width,
    /// reckoned from each face's extent. A change is found by bisecting the
    /// tone between two readings, some sixty readings more, and starts a
    /// stroke of its own.
    pub const MAX_HATCH_LINES: i64 = 1 << 16;
    pub const MAX_READINGS: i64 = 1 << 20;
    pub const MAX_WIDTH_CHANGES: i64 = 1 << 20;
    /// The most widths a hatch line has: a width step is a layer of the
    /// bake, and a change of width between two readings is bisected.
    pub const MAX_WIDTH_STEPS: i64 = 16;

    pub fn new(
        solid: Solid,
        layers: Vec<LayerSpec>,
        presets: Vec<CameraPreset>,
        provenance: Provenance,
    ) -> PolyhedronManifest {
        PolyhedronManifest { schema: KURVEN_SCHEMA, solid, layers, presets, provenance }
    }

    pub fn from_json(json: &JsonValue) -> Result<PolyhedronManifest, SchemaError> {
        let o = json.object("PolyhedronManifest")?;
        let schema = o.int("schema", "PolyhedronManifest")?;
        if schema != KURVEN_SCHEMA {
            return Err(ManifestError::UnsupportedSchema { schema, supported: KURVEN_SCHEMA }.into());
        }
        if let Some(key) = OTHER_GEOMETRY_KEYS.iter().find(|k| o.contains_key(**k)) {
            return Err(ManifestError::Unsupported(format!(
                "a polyhedron with a landscape's {key}: a manifest holds one geometry, a landscape, a parametric \
                 surface or a polyhedron"
            ))
            .into());
        }
        if o.contains_key("surface") {
            return Err(ManifestError::Unsupported(
                "a polyhedron with a parametric surface: a manifest holds one geometry, a landscape, a parametric \
                 surface or a polyhedron"
                    .to_string(),
            )
            .into());
        }
        let solid = PolyhedronManifest::solid_from_json(o.value("polyhedron", "PolyhedronManifest")?)?;
        let layer_list = o.array("layers", "PolyhedronManifest")?;
        if layer_list.len() > PolyhedronManifest::MAX_LAYERS {
            return Err(ManifestError::Invalid(format!(
                "a polyhedron with {} layers, more than the {} it holds",
                layer_list.len(),
                PolyhedronManifest::MAX_LAYERS
            ))
            .into());
        }
        let layers = layers_from(layer_list, Geometry::Polyhedron)?;
        let presets = presets_from(o.array("presets", "PolyhedronManifest")?)?;
        let provenance = Provenance::from_json(o.value("provenance", "PolyhedronManifest")?)?;
        if let Some(why) = PolyhedronManifest::unsupported(&layers, &solid) {
            return Err(ManifestError::Invalid(why).into());
        }
        Ok(PolyhedronManifest { schema, solid, layers, presets, provenance })
    }

    /// The solid a manifest's `polyhedron` spells, counted before it is
    /// built and checked as it is (`Solid::checking`).
    pub fn solid_from_json(json: &JsonValue) -> Result<Solid, SchemaError> {
        let what = "PolyhedronManifest.polyhedron";
        let o = json.object(what)?;
        let points = o.array("vertices", what)?;
        if points.len() > PolyhedronManifest::MAX_VERTICES {
            return Err(ManifestError::Invalid(format!(
                "polyhedron.vertices: {} vertices, more than the {} a polyhedron holds",
                points.len(),
                PolyhedronManifest::MAX_VERTICES
            ))
            .into());
        }
        let mut vertices = Vec::with_capacity(points.len());
        for (k, point) in points.iter().enumerate() {
            let c = point.doubles(&format!("vertices[{k}]"), what)?;
            if c.len() != 3 {
                return Err(ManifestError::bad_shape(&format!("polyhedron.vertices[{k}]"), &[c.len() as i64]).into());
            }
            // A coordinate the depth pass's single precision holds.
            if !c.iter().all(|x| x.is_finite() && x.abs() <= f64::from(f32::MAX)) {
                return Err(ManifestError::Invalid(format!(
                    "polyhedron.vertices[{k}] is not a finite point: {}",
                    text::doubles(&c)
                ))
                .into());
            }
            vertices.push(P3::<WorldSpace>::new(c[0], c[1], c[2]));
        }
        let mut corners = 0usize;
        let mut faces = Vec::new();
        for (i, face) in o.array("faces", what)?.iter().enumerate() {
            let JsonValue::Array(loops) = face else {
                return Err(JsonError::wrong_type(&format!("faces[{i}]"), "an array of loops", what).into());
            };
            let face: Vec<Vec<i64>> =
                loops.iter().map(|l| l.ints(&format!("faces[{i}]"), what)).collect::<Result<_, _>>()?;
            let count: usize = face.iter().map(Vec::len).sum();
            corners += count;
            if count > PolyhedronManifest::MAX_FACE_CORNERS || corners > PolyhedronManifest::MAX_CORNERS {
                return Err(ManifestError::Invalid(format!(
                    "polyhedron.faces[{i}]: more corners than a polyhedron holds, {} a face and {} in all",
                    PolyhedronManifest::MAX_FACE_CORNERS,
                    PolyhedronManifest::MAX_CORNERS
                ))
                .into());
            }
            faces.push(face);
        }
        Solid::checking(vertices, faces).map_err(|e| ManifestError::Invalid(format!("polyhedron: {e}")).into())
    }

    /// Why layers cannot be drawn on `solid`, or `None` when they can: a layer
    /// other than its edges or hatching on its faces, a width that is not a
    /// number of at least 0, two layers of one name, more widths than
    /// `MAX_WIDTH_STEPS`, or hatchings that together would lay more lines,
    /// read their tones more often or change width more often than
    /// `MAX_HATCH_LINES`, `MAX_READINGS` and `MAX_WIDTH_CHANGES` allow. Each is
    /// reckoned per face from its extent: at most `extent / spacing + 1`
    /// lines, each at most `extent / step + 1` readings, and, under Turing's
    /// pattern at `scale`, at most `Turing.slope × widthSteps × extent /
    /// scale + 1` changes of width, since a change is a crossing of one of
    /// the tone's `widthSteps` levels and the tone moves at most
    /// `Turing.slope / scale` per unit along a line. Uniform and half-Lambert
    /// tones are constant on a face, and change nothing. The CLI's flags and
    /// the app's restyles are held to the same rule.
    pub fn unsupported(layers: &[LayerSpec], solid: &Solid) -> Option<String> {
        let mut names = HashSet::new();
        let (mut lines, mut readings, mut changes) = (0.0f64, 0.0f64, 0.0f64);
        for layer in layers {
            let refuse = |why: String| format!("layer '{}': {why}", layer.name);
            // A layer is kept across a restyle, and its slider found, by
            // its name.
            if !names.insert(layer.name.clone()) {
                return Some(format!(
                    "two layers named '{}': a polyhedron's layers are told apart by name",
                    layer.name
                ));
            }
            if !(layer.width.is_finite() && layer.width >= 0.0) {
                return Some(refuse(format!(
                    "a width must be a number not below 0, not {}",
                    text::double(layer.width)
                )));
            }
            match &layer.source {
                LayerSource::Edges => continue,
                LayerSource::Hatching { hatching, faces: FaceSelector::Solid } => {
                    if let Some(why) = FaceSelector::Solid.unsupported_hatching(hatching) {
                        return Some(refuse(why));
                    }
                    if hatching.width_steps > PolyhedronManifest::MAX_WIDTH_STEPS {
                        return Some(refuse(format!(
                            "hatching in {} widths, more than the {} a hatch line has",
                            hatching.width_steps,
                            PolyhedronManifest::MAX_WIDTH_STEPS
                        )));
                    }
                    // How fast the tone moves along a line, per unit of length.
                    let slope = match hatching.tone {
                        Tone::Uniform | Tone::HalfLambert { .. } => 0.0,
                        Tone::Turing { scale, .. } => Turing::SLOPE / scale,
                    };
                    for frame in &solid.frames {
                        let across = frame.extent / hatching.spacing + 1.0;
                        lines += across;
                        readings += across * (frame.extent / hatching.step + 1.0);
                        if slope > 0.0 {
                            changes += across * (slope * hatching.width_steps as f64 * frame.extent + 1.0);
                        }
                    }
                }
                _ => return Some(refuse("a polyhedron draws its edges and hatching on its faces".to_string())),
            }
        }
        if !(lines <= PolyhedronManifest::MAX_HATCH_LINES as f64) {
            return Some(format!(
                "the hatchings would lay {} lines, more than the {} a polyhedron's hatchings lay",
                text::g3(lines),
                PolyhedronManifest::MAX_HATCH_LINES
            ));
        }
        if !(readings <= PolyhedronManifest::MAX_READINGS as f64) {
            return Some(format!(
                "the hatchings would read their tones {} times, more than the {} a polyhedron's hatchings read",
                text::g3(readings),
                PolyhedronManifest::MAX_READINGS
            ));
        }
        if !(changes <= PolyhedronManifest::MAX_WIDTH_CHANGES as f64) {
            return Some(format!(
                "the hatchings would change width {} times, more than the {} a polyhedron's hatchings change: \
                 Turing's pattern is too fine for its lines",
                text::g3(changes),
                PolyhedronManifest::MAX_WIDTH_CHANGES
            ));
        }
        None
    }

    pub fn json(&self) -> JsonValue {
        let vertices = self.solid.vertices.iter().map(|v| json::doubles(&v.v.to_array())).collect();
        let faces = self
            .solid
            .loops
            .iter()
            .map(|face| {
                JsonValue::Array(
                    face.iter().map(|l| json::ints(&l.iter().map(|&k| k as i64).collect::<Vec<_>>())).collect(),
                )
            })
            .collect();
        object([
            ("schema", JsonValue::Int(self.schema)),
            ("polyhedron", object([("vertices", JsonValue::Array(vertices)), ("faces", JsonValue::Array(faces))])),
            ("layers", JsonValue::Array(self.layers.iter().map(LayerSpec::json).collect())),
            ("presets", JsonValue::Array(self.presets.iter().map(CameraPreset::json).collect())),
            ("provenance", self.provenance.json()),
        ])
    }

    /// Canonical JSON with a trailing newline, as a landscape's manifest writes.
    pub fn canonical_json(&self) -> String {
        self.json().canonical() + "\n"
    }

    pub fn preset(&self, name: &str) -> Result<&CameraPreset, ManifestError> {
        find_preset(&self.presets, name)
    }
}

/// A bundle's manifest, by what it holds: a landscape (`Manifest`), a
/// parametric surface by the request that builds it (`SurfaceManifest`), or a
/// polyhedron by its vertices and loops (`PolyhedronManifest`). One schema,
/// told apart by key: a manifest with a `surface` is a parametric surface's,
/// one with a `polyhedron` a polyhedron's, any other a landscape's.
#[derive(Debug, Clone, PartialEq)]
pub enum BundleManifest {
    Landscape(Manifest),
    Surface(SurfaceManifest),
    Polyhedron(PolyhedronManifest),
}

impl BundleManifest {
    pub fn from_json(json: &JsonValue) -> Result<BundleManifest, SchemaError> {
        let o = json.object("Manifest")?;
        Ok(if o.contains_key("polyhedron") {
            BundleManifest::Polyhedron(PolyhedronManifest::from_json(json)?)
        } else if o.contains_key("surface") {
            BundleManifest::Surface(SurfaceManifest::from_json(json)?)
        } else {
            BundleManifest::Landscape(Manifest::from_json(json)?)
        })
    }

    pub fn canonical_json(&self) -> String {
        match self {
            BundleManifest::Landscape(m) => m.canonical_json(),
            BundleManifest::Surface(m) => m.canonical_json(),
            BundleManifest::Polyhedron(m) => m.canonical_json(),
        }
    }
}

#[cfg(test)]
mod tests;
