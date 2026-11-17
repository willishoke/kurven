//! Several geometries in one scene, under one camera: the port of
//! `Sources/KurvenCore/Composite.swift`.
//!
//! Each part is a `Scene` of its own -- its faces, its curves, its depth
//! tolerance -- standing at its placement. Every clipped curve is tested
//! against every geometry's occlusion test (`OcclusionTest`): its own,
//! exactly as when it stands alone (`Scene::visible`), and then each
//! other's, which forgives it nothing (`OcclusionTest::is_clear_of_another_solid`).
//! Curves of different geometries that cross on the page are cut only where
//! one hides the other, never joined.
//!
//! Another geometry cuts a curve's segments where their lines of sight meet
//! it (`Occluder`): a polyhedron and a landscape on its grid in closed form,
//! the round torus by the clearances of its sphere tracing.
//!
//! Only a geometry that cuts another's segments exactly stands in a
//! composite: a landscape on its grid, a polyhedron, or the round torus. The
//! coordinate image says only whether a point is the front-most of its own
//! surface, nothing about another solid, so a surface tested by it -- the
//! forced torus -- is refused; so is a displaced torus, whose sphere tracing
//! steps by a Lipschitz constant read off its grid rather than proved, and a
//! landscape on its function, whose slope nothing bounds. No depth buffer
//! stands in between solids: the drawing is a function of the geometry and
//! the camera.

use std::fmt;

use kurven_math::swift;

use crate::camera::Camera;
use crate::json;
use crate::occlusion::{SolidOcclusion, SphereTracing, sort_by_start};
use crate::parametric::PortRefusal;
use crate::polyline::{Layer, PolylineSet};
use crate::runs::Runs;
use crate::scene::{Scene, SceneGeometry};
use crate::simd::Vec3;
use crate::slab::Slab;
use crate::spaces::{P3, Transform, ViewSpace, WorldSpace};

/// Where a geometry stands in a composite: scaled uniformly about the origin,
/// then translated. A similarity, so a line of sight in world space is a line
/// of sight in the geometry's own coordinates, and each occlusion test is
/// asked there, exactly as when the geometry stands alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub scale: f64,
    pub offset: Vec3,
}

impl Default for Placement {
    fn default() -> Self {
        Placement::IDENTITY
    }
}

impl Placement {
    pub const IDENTITY: Placement = Placement { scale: 1.0, offset: Vec3::ZERO };

    /// The scales a placement takes, and how far from the origin its offset
    /// reaches on each axis. Within them a geometry's arithmetic in world
    /// space stays far from overflow and from the rounding of its distance
    /// to the origin, and the steps along a segment stay countable.
    pub const SCALES: (f64, f64) = (1e-3, 1e3);
    pub const REACH: f64 = 1e4;

    pub fn new(scale: f64, offset: Vec3) -> Placement {
        Placement { scale, offset }
    }

    /// Why this placement is out of range; `None` when it is in it.
    pub fn refusal(&self) -> Option<String> {
        let (lo, hi) = Placement::SCALES;
        if !(self.scale.is_finite() && self.scale >= lo && self.scale <= hi) {
            return Some(format!(
                "a placement's scale is between {} and {}, not {}",
                json::number(lo),
                json::number(hi),
                json::number(self.scale)
            ));
        }
        let o = self.offset;
        if !(0..3).all(|k| o[k].is_finite() && o[k].abs() <= Placement::REACH) {
            return Some(format!(
                "a placement's offset is within {} of the origin on each axis, not ({}, {}, {})",
                json::number(Placement::REACH),
                json::number(o.x),
                json::number(o.y),
                json::number(o.z)
            ));
        }
        None
    }

    /// From the geometry's own coordinates into world space.
    pub fn transform(&self) -> Transform<WorldSpace, WorldSpace> {
        let s = self.scale;
        Transform::from_rows((Vec3::new(s, 0.0, 0.0), Vec3::new(0.0, s, 0.0), Vec3::new(0.0, 0.0, s)), self.offset)
    }

    /// A world point in the geometry's own coordinates.
    pub fn back(&self, p: P3<WorldSpace>) -> P3<WorldSpace> {
        P3::from_vec((p.v - self.offset) / self.scale)
    }
}

/// Why geometries cannot stand in one composite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositeError(pub String);

impl fmt::Display for CompositeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompositeError {}

/// A part's own drawing reaching what the port has not brought across is
/// refused through the composite, by its text.
impl From<PortRefusal> for CompositeError {
    fn from(refusal: PortRefusal) -> Self {
        CompositeError(refusal.to_string())
    }
}

/// One part: what it is called in a refusal (its bundle's name), the
/// geometry and its curves looking through the composite's camera from where
/// it stands, and its placement.
#[derive(Debug, Clone)]
pub struct Part {
    pub name: String,
    pub scene: Scene,
    pub placement: Placement,
}

/// How a part hides the segments of another's curves: the spans of a
/// segment whose lines of sight meet the part, as fractions of the way
/// along it. Each works only where those lines can meet the part at all,
/// through the box or the ball that holds it.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Occluder {
    /// A polyhedron's pieces, in closed form, asked of the segment's ends
    /// in view space, where its occlusion is built.
    Pieces(SolidOcclusion),
    /// A landscape on its grid, in closed form (`Slab`), in its own
    /// coordinates: world space less its offset.
    Slab { slab: Slab, offset: Vec3 },
    /// The round torus: sphere tracing's clearances
    /// (`SphereTracing::clearance_of_another_solid`), each a stretch of the
    /// segment with one verdict.
    Tube { tracing: SphereTracing, placement: Placement },
}

impl Occluder {
    /// The most points a segment is asked at against one part: past it,
    /// the drawing is refused rather than cut.
    pub const STEPS: i64 = 1_000_000;

    pub fn new(part: &Part) -> Result<Occluder, CompositeError> {
        let view = part.scene.camera.view;
        let placement = part.placement;
        Ok(match part.scene.geometry() {
            SceneGeometry::Solid(s) => Occluder::Pieces(s.occlusion(&view)),
            // On its grid: one on its function is refused at the door.
            SceneGeometry::Heightfield(h) => {
                Occluder::Slab { slab: Slab::new(h, &part.name)?, offset: placement.offset }
            }
            // Refused at the composite's door without one.
            SceneGeometry::Parametric(s) => Occluder::Tube {
                tracing: SphereTracing::new(s, view.projection_direction()).expect("a tubular neighbourhood"),
                placement,
            },
        })
    }

    /// The spans of the segment `a -> b` (view space; `world` takes it
    /// back to world space) whose lines of sight, along `toward`, meet
    /// this part.
    pub fn hidden(
        &self,
        a: P3<ViewSpace>,
        b: P3<ViewSpace>,
        world: &Transform<ViewSpace, WorldSpace>,
        toward: Vec3,
    ) -> Result<Vec<(f64, f64)>, CompositeError> {
        match self {
            Occluder::Pieces(occlusion) => Ok(occlusion.hidden(a, b, &|_| false)),
            Occluder::Slab { slab, offset } => {
                Ok(slab.hidden(world.apply(a).v - *offset, world.apply(b).v - *offset, toward))
            }
            Occluder::Tube { tracing, placement } => {
                let wa = placement.back(world.apply(a)).v;
                let wb = placement.back(world.apply(b)).v;
                let e = &tracing.neighbourhood;
                let Some((s0, s1)) = Composite::sweep_of_ball(e.centre, 1.01 * e.radius, wa, wb, toward) else {
                    return Ok(Vec::new());
                };
                let length = (wb - wa).length();
                let least = 1e-5 * e.radius;
                let clearance = |s: f64| tracing.clearance_of_another_solid(P3::from_vec(wa + (wb - wa) * s));
                if !(length > 0.0) {
                    return Ok(if clearance(0.0) < 0.0 { vec![(0.0, 1.0)] } else { Vec::new() });
                }
                let mut hidden = Vec::new();
                let (mut s, mut m, mut steps) = (s0, clearance(s0), 0i64);
                while s < s1 {
                    // Every point within the clearance keeps the verdict, so
                    // the next point to ask is the clearance along; a clearance
                    // under the least step is a verdict to that resolution.
                    let next = swift::min(s + swift::max(m.abs(), least) / length, s1);
                    let mn = clearance(next);
                    if (m < 0.0) != (mn < 0.0) {
                        let (mut lo, mut hi) = (s, next);
                        while hi - lo > 1e-13 {
                            let mid = 0.5 * (lo + hi);
                            if (clearance(mid) < 0.0) == (m < 0.0) {
                                lo = mid;
                            } else {
                                hi = mid;
                            }
                        }
                        hidden.push(if m < 0.0 { (s, lo) } else { (hi, next) });
                    } else if m < 0.0 {
                        hidden.push((s, next));
                    }
                    s = next;
                    m = mn;
                    steps += 1;
                    if steps > Occluder::STEPS {
                        return Err(Composite::too_fine(steps as f64));
                    }
                }
                Ok(hidden)
            }
        }
    }
}

/// Several geometries in one scene, under one camera (`Composite`).
#[derive(Debug, Clone)]
pub struct Composite {
    pub parts: Vec<Part>,
    pub camera: Camera,
    pub occluders: Vec<Occluder>,
}

impl Composite {
    /// The parts, each `scene` taken with its geometry, layers and depth
    /// tolerance (its camera is the composite's), standing at its placement.
    pub fn new(parts: Vec<(String, Scene, Placement)>, camera: Camera) -> Result<Composite, CompositeError> {
        if camera.is_perspective() {
            return Err(CompositeError(
                "a composite is drawn under a parallel projection, and this camera is perspective".to_string(),
            ));
        }
        let mut placed = Vec::new();
        for (name, scene, placement) in parts {
            if let Some(why) = placement.refusal() {
                return Err(CompositeError(format!("{name}: {why}")));
            }
            match scene.geometry() {
                SceneGeometry::Parametric(s) if s.neighbourhood.is_none() => {
                    return Err(CompositeError(format!(
                        "{name} is a surface with no exact occlusion test (it has no tubular neighbourhood): its \
                         coordinate image says only what of the surface itself is in front, nothing about another \
                         solid, so it cannot stand in a composite. It bakes alone."
                    )));
                }
                SceneGeometry::Parametric(s)
                    if s.neighbourhood.as_ref().is_some_and(|e| e.above != 0.0 || e.below != 0.0) =>
                {
                    return Err(CompositeError(format!(
                        "{name} is a displaced torus: its sphere tracing steps by a Lipschitz constant read off its \
                         grid, not proved, so it cannot hide another solid's curves exactly, and it cannot stand in \
                         a composite. It bakes alone."
                    )));
                }
                SceneGeometry::Heightfield(h) if h.refine.as_ref().is_some_and(|r| r.magnitude.is_some()) => {
                    return Err(CompositeError(format!(
                        "{name} is a landscape on its function: nothing bounds f's slope, so it cannot cut another \
                         solid's curves exactly, and it cannot stand in a composite. It stands on its grid \
                         (kurven-cli scene --no-refine), or bakes alone."
                    )));
                }
                SceneGeometry::Heightfield(_) if placement.scale != 1.0 => {
                    return Err(CompositeError(format!(
                        "{name}: a landscape stands at scale 1 in a composite, since its depth tolerance and its \
                         function's tolerance are lengths in its own domain; scale the other geometries instead"
                    )));
                }
                _ => {}
            }
            let view = placement.transform().then(&camera.view);
            let looking = scene.looking(Camera { view, projection: camera.projection });
            placed.push(Part { name, scene: looking, placement });
        }
        let occluders = placed.iter().map(Occluder::new).collect::<Result<Vec<_>, _>>()?;
        Ok(Composite { parts: placed, camera, occluders })
    }

    /// Every part's layers in view space, in order, tested for the camera:
    /// each by its own geometry as alone, then cut wherever another part
    /// hides it. After a part's own layers comes its contact with the
    /// landscapes, where it has one (`visible_contact`), cut the same
    /// way: a landscape hides its own contact curve only where the line of
    /// sight goes into its slab, as it hides any point on its boundary
    /// (`Slab`). An unclipped layer is drawn whole. Refused where a segment
    /// would be asked at more points against one part than
    /// `Occluder::STEPS`.
    pub fn visible_layers(&self) -> Result<Vec<(Layer, PolylineSet<ViewSpace>)>, CompositeError> {
        let world = self.camera.view.inverse();
        let mut out = Vec::new();
        for i in 0..self.parts.len() {
            let others: Vec<&Occluder> =
                (0..self.occluders.len()).filter(|&j| j != i).map(|j| &self.occluders[j]).collect();
            let mut own = self.parts[i].scene.visible_layers(None, None)?;
            own.extend(self.visible_contact(i)?);
            for (layer, seen) in own {
                if !layer.spec.clipped || others.is_empty() {
                    out.push((layer, seen));
                    continue;
                }
                let cut = Composite::cut(&seen, &others, &world)?;
                out.push((layer, cut));
            }
        }
        Ok(out)
    }

    pub fn too_fine(count: f64) -> CompositeError {
        CompositeError(format!(
            "a segment would be asked at {} points against one geometry, past the {} a composite allows",
            swift::min(count, 1e18) as i64,
            Occluder::STEPS
        ))
    }

    /// The fractions of the segment `a -> b` whose lines of sight, along
    /// the unit `d`, pass within `radius` of `centre`: where the distance
    /// from the centre to the line through the point is at most the radius,
    /// a quadratic in the fraction. Those that meet the ball ahead of the
    /// point are among them.
    pub fn sweep_of_ball(centre: Vec3, radius: f64, a: Vec3, b: Vec3, d: Vec3) -> Option<(f64, f64)> {
        // The part of (a + s e - centre) across d.
        let w0 = a - centre;
        let e = b - a;
        let p0 = w0 - w0.dot(d) * d;
        let pe = e - e.dot(d) * d;
        let qa = pe.dot(pe);
        let qb = 2.0 * p0.dot(pe);
        let qc = p0.dot(p0) - radius * radius;
        if qa <= 1e-24 * radius * radius {
            return if qc <= 0.0 { Some((0.0, 1.0)) } else { None };
        }
        let disc = qb * qb - 4.0 * qa * qc;
        if !(disc >= 0.0) {
            return None;
        }
        let root = disc.sqrt();
        let lo = swift::max((-qb - root) / (2.0 * qa), 0.0);
        let hi = swift::min((-qb + root) / (2.0 * qa), 1.0);
        if lo <= hi { Some((lo, hi)) } else { None }
    }

    /// The runs of `paths` (view space) that no occluder hides: each
    /// segment cut to the spans every occluder leaves, a run kept across a
    /// vertex where the spans either side meet it, and a run of fewer than
    /// two vertices dropped. `world` takes a view point back to world space.
    pub fn cut(
        paths: &PolylineSet<ViewSpace>,
        occluders: &[&Occluder],
        world: &Transform<ViewSpace, WorldSpace>,
    ) -> Result<PolylineSet<ViewSpace>, CompositeError> {
        if paths.vertices.is_empty() {
            return Ok(paths.clone());
        }
        // Toward the eye: view depth grows toward it.
        let toward = (world.apply(P3::new(0.0, 0.0, 1.0)).v - world.apply(P3::new(0.0, 0.0, 0.0)).v).normalized();
        let vertices = &paths.vertices;
        let mut segments: Vec<usize> = Vec::new();
        for path in 0..paths.count() {
            segments.extend(paths.offsets[path] + 1..paths.offsets[path + 1]);
        }
        // Each segment alone, in order: Swift spreads them over threads.
        let mut spans: Vec<Vec<(f64, f64)>> = Vec::with_capacity(segments.len());
        for &k in &segments {
            let (a, b) = (vertices[k - 1], vertices[k]);
            let mut hidden = Vec::new();
            for o in occluders {
                hidden.extend(o.hidden(a, b, world, toward)?);
            }
            spans.push(Composite::seen(hidden));
        }
        let mut runs = Runs::new();
        let mut s = 0;
        for path in 0..paths.count() {
            for k in paths.offsets[path] + 1..paths.offsets[path + 1] {
                let (a, b) = (vertices[k - 1], vertices[k]);
                runs.segment(&spans[s], |t| Composite::along(a, b, t));
                s += 1;
            }
            runs.end_path();
        }
        let out = runs.finish();
        Ok(PolylineSet::from_paths(&out))
    }

    /// The spans of [0, 1] left by `hidden`, in order.
    pub fn seen(mut hidden: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        sort_by_start(&mut hidden);
        let mut t = 0.0;
        let mut spans = Vec::new();
        for (a, b) in hidden {
            if a > t {
                spans.push((t, a));
            }
            t = swift::max(t, b);
        }
        if t < 1.0 {
            spans.push((t, 1.0));
        }
        spans
    }

    /// The point `t` of the way from `a` to `b`, the ends to the bit.
    pub fn along(a: P3<ViewSpace>, b: P3<ViewSpace>, t: f64) -> P3<ViewSpace> {
        if t == 0.0 {
            a
        } else if t == 1.0 {
            b
        } else {
            P3::from_vec(a.v + t * (b.v - a.v))
        }
    }
}
