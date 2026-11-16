//! Faces in the boundary-representation sense: the port of
//! `Sources/KurvenCore/Face.swift`.
//!
//! Swift's `Face.Copy` is [`FaceCopy`] here, since `Copy` is a Rust trait;
//! its `Computed` is [`Computed`], made once behind a `OnceLock`. The two
//! initializers are [`Face::single`] (one copy, on the identity) and
//! [`Face::new`] with a [`FaceInit`], whose `Option` fields are Swift's
//! defaulted arguments.

use std::sync::{Arc, OnceLock};

use kurven_math::swift;

use crate::hatching::FaceKind;
use crate::manifest::{Affine2, Interval, LayerRole, LayerSource, LayerSpec, Region, Walls};
use crate::parametric::{
    JetMap, ParamAxis, ParamGradient, ParamLift, ParamPredicate, ParamRectangle, ParamScalar, ParametricSurface,
    SecondOrderMap, SurfaceJet,
};
use crate::scalar_field::{self, Contoured, ParamPaths, ScalarField, Trim, TrimIndex};
use crate::scene::{Heightfield, SceneGeometry};
use crate::simd::{Vec2, Vec3};
use crate::solid::Solid;
use crate::spaces::{DomainSpace, P2, P3, ParamSpace};
use crate::surface::{ContourRefine, Surface};

/// Which part of a solid's boundary a face is, in the order a landscape
/// lists its faces (`Surface::faces`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceID {
    /// A landscape's graph, where z = |f|.
    Graph,
    /// The plateau of one band of the truncation, numbered as
    /// `Truncation::band_at` numbers them; a uniform truncation's one
    /// plateau is band 0.
    Plateau { band: usize },
    /// The riser where a banded truncation steps, at the end of band `band`
    /// toward greater Re z.
    Riser { band: usize },
    /// The riser at the tile boundary between the copies on two tiles, by
    /// their place in the tiles: where the copy on `tile` stands over the
    /// copy on `facing`, and faces it (`Surface::tile_boundary_risers`).
    TileBoundaryRiser { tile: usize, facing: usize },
    /// The wall along one edge of the walls' perimeter.
    Wall { edge: usize },
    /// A parametric surface's one face.
    Surface,
    /// A polyhedron's face, as `Solid.loops` lists them.
    Solid { face: usize },
}

impl FaceID {
    pub fn kind(self) -> FaceKind {
        match self {
            FaceID::Graph => FaceKind::Graph,
            FaceID::Plateau { .. } => FaceKind::Plateau,
            FaceID::Riser { .. } | FaceID::TileBoundaryRiser { .. } => FaceKind::Riser,
            FaceID::Wall { .. } => FaceKind::Wall,
            FaceID::Surface => FaceKind::Surface,
            FaceID::Solid { .. } => FaceKind::Solid,
        }
    }
}

/// A value made the first time it is asked, and kept: shared by whatever
/// holds it, as a face's copies share its trim loops. Made once, under
/// `OnceLock`'s lock, as Swift's `Mutex` makes it once.
pub struct Computed<T> {
    make: Box<dyn Fn() -> T + Send + Sync>,
    made: OnceLock<T>,
}

impl<T> Computed<T> {
    pub fn new(make: impl Fn() -> T + Send + Sync + 'static) -> Computed<T> {
        Computed { make: Box::new(make), made: OnceLock::new() }
    }

    pub fn value(&self) -> &T {
        self.made.get_or_init(|| (self.make)())
    }
}

/// The point of the parameter plane a point of space stands at.
pub type FootMap = Arc<dyn Fn(Vec3) -> P2<ParamSpace> + Send + Sync>;
/// A copy's boundary, given its tile.
pub type Bounding = Arc<dyn Fn(Affine2) -> ParamPaths + Send + Sync>;
/// Membership asked with a copy's tile.
pub type TileContains = Arc<dyn Fn(P2<ParamSpace>, Affine2) -> bool + Send + Sync>;
/// The footprint where a copy stands: its perimeter carried back into the
/// parameter plane, and whether it holds a point there.
pub type Footprint = (ParamPaths, ParamPredicate);
/// The footprint of a copy, given its tile.
pub type FootprintOf = Arc<dyn Fn(Affine2) -> Option<Footprint> + Send + Sync>;
/// A face's trim loops, made when first asked.
pub type Loops = Arc<Computed<ParamPaths>>;

/// A face in the boundary-representation sense (Mäntylä 1988): a region of
/// one smooth surface, flat or curved, bounded by loops. A solid's faces
/// together are its manifold, the surface it is bounded by, in boundary
/// representation its shell. That is closed for a polyhedron, a torus, or a
/// landscape standing on walls given as a perimeter; a landscape's faces
/// leave its sides open where it has no walls, or walls given as a mesh,
/// which describe no face.
///
/// A face is a map from a parameter rectangle into space, trimmed: the part
/// of the parameter plane it covers is its *trimmed domain*, bounded by its
/// trim loops (`trim_loops`), and each copy's `contains` is the domain's point
/// membership classification (Requicha 1980). A landscape's faces are its
/// graph, a plateau per band of the truncation, a riser wherever the drawn
/// top drops inside the solid, where the truncation steps between two bands
/// or two copies meet at a tile boundary with different tops, and a wall
/// along each edge of the walls' perimeter (`Surface::faces`); a parametric
/// surface is one face (`ParametricSurface::face`); a polyhedron's faces are
/// its loops, flat (`Solid::face`). `SceneGeometry::faces` lists a scene's.
///
/// A face is a view onto the grids and the solids it is read from, and
/// costs nothing to make: it holds closures, not samples.
#[derive(Clone)]
pub struct Face {
    pub id: FaceID,
    /// The parameter rectangle the map is defined on, and where a field on
    /// the face is sampled to be contoured. It covers the trimmed domain,
    /// except up a wall or a riser with no finite truncation over it: there
    /// it stops at the grid's highest sample, and f can rise over its
    /// samples.
    pub domain: ParamRectangle,
    /// The point of space at a point of the parameter plane, with its two
    /// partial derivatives: the face where it is first, before a copy moves
    /// it (`FaceCopy::place`).
    pub map: JetMap,
    /// Where the face is in space. One copy, on the identity, but for a tiled
    /// landscape's graph, plateaus and band boundaries' risers, which have
    /// one per tile, as the curves on them do (`Occluder.tiles`): contoured
    /// once, then replicated.
    pub copies: Vec<FaceCopy>,
    loops: Option<Loops>,
    /// Where a point of space stands on the face, before a copy moves it: a
    /// point of the parameter plane. A flat face's is the point's orthogonal
    /// projection onto its plane; the graph's and a plateau's the point of the
    /// domain under it; a parametric surface's the nearest point, by Newton's
    /// method (`ParametricSurface::nearest_parameter`). What a pick reads
    /// a face's distance off (`Face::pick`).
    pub foot: FootMap,
    /// A copy's boundary, given its tile: the curves of the parameter plane
    /// its trimmed domain ends on (`Face::boundary`).
    pub(crate) bounding: Bounding,
    /// On the graph of f under a truncation, |f| less the truncation
    /// height: negative on the graph, zero on the truncation edge, whose
    /// chords the trim loops are. A contour on the face is cut where it
    /// reaches 0, on the edge itself rather than on a chord of it
    /// (`contours`). `None` on every other face, on the graph where nothing
    /// is truncated, and on a grid's graph, whose truncation edge is the
    /// loops' chords themselves, solved linearly on the grid.
    pub truncation_excess: Option<ParamScalar>,
    /// The face's point in space, where its parameter plane is the world's
    /// xy, as the graph's is: a chord's sag off the face is then a length in
    /// the cell's own units, and the plain routine refines it (`contours`).
    /// `None` on every other face.
    pub(crate) lift: Option<ParamLift>,
    /// The derivatives of the height, the map's z, in the parameter plane,
    /// where the face is the graph of a height over the world's xy, as the
    /// graph's is: its slope read without its point (`heightSlope`). `None`
    /// on every other face.
    pub(crate) height_gradient: Option<ParamGradient>,
    /// The map's jet of order two, where the face's surface has it in
    /// closed form (`ParametricSurface::second_order_jet`): its second
    /// fundamental form, and so its principal directions. `None` on every
    /// other face.
    pub second_order_jet: Option<SecondOrderMap>,
}

/// One copy of a face: where it is, which way it faces, and how much of
/// it is there (Swift's `Face.Copy`).
#[derive(Clone)]
pub struct FaceCopy {
    /// The map from the face's own place to this copy's.
    pub tile: Affine2,
    /// `+1` when `du × dv`, carried into place (`place`), points out of
    /// the solid the face bounds, `-1` when it points in; `None` when the
    /// face bounds no solid. The face's own orientation turned by the
    /// tile: a reflected copy's is the other way.
    pub outward: Option<f64>,
    /// Whether a point of the parameter plane lies in this copy's
    /// trimmed domain. Answered anywhere in the plane, not only on the
    /// rectangle: the face's own trim is part of it (a landscape face
    /// ends at the domain's edge, a wall at its edge's ends), and so is
    /// the footprint, tested at the copy's point in space, as the mesh
    /// and visibility test it. On a periodic axis it answers the same a
    /// period along.
    pub contains: ParamPredicate,
    loops: Option<Loops>,
    /// The footprint where this copy stands: its perimeter carried back
    /// into the parameter plane, and whether it holds a point there.
    pub(crate) footprint: Option<Footprint>,
}

impl FaceCopy {
    /// The closed curves this copy's trimmed domain ends on: the face's
    /// trim loops and, where a footprint cuts the copy, the footprint's
    /// perimeter where the copy stands, carried back into the parameter
    /// plane. Not intersected: with a footprint they bound nothing
    /// even-odd, and which side of the perimeter is in the copy is the
    /// footprint's to say (`Face::contours`). `None` where the face's are.
    pub fn trim_loops(&self) -> Option<ParamPaths> {
        self.loops.as_ref().map(|l| {
            let mut all = l.value().clone();
            if let Some((perimeter, _)) = &self.footprint {
                all.extend(perimeter.iter().cloned());
            }
            all
        })
    }

    /// A jet of the face's map carried into place: its point moved by the
    /// tile, its derivatives turned with it, heights kept.
    pub fn place(&self, jet: SurfaceJet) -> SurfaceJet {
        let tile = self.tile;
        let turned = |d: Vec3| Vec3::new(tile.a * d.x + tile.b * d.y, tile.c * d.x + tile.d * d.y, d.z);
        SurfaceJet::new(tile.apply(P3::from_vec(jet.position)).v, turned(jet.du), turned(jet.dv))
    }
}

/// Swift's tiled initializer, argument for argument: the `Option` fields are
/// its defaulted ones.
pub struct FaceInit {
    pub id: FaceID,
    pub domain: ParamRectangle,
    pub map: JetMap,
    pub outward: Option<f64>,
    pub tiles: Vec<Affine2>,
    pub trim_loops: Option<Arc<dyn Fn() -> ParamPaths + Send + Sync>>,
    pub footprint: Option<FootprintOf>,
    pub truncation_excess: Option<ParamScalar>,
    pub lift: Option<ParamLift>,
    pub height_gradient: Option<ParamGradient>,
    pub second_order_jet: Option<SecondOrderMap>,
    pub foot: FootMap,
    pub bounding: Bounding,
    pub contains: TileContains,
}

impl Face {
    /// A face copied by `tiles`: each copy faces as `outward` says, turned by
    /// its tile, and holds what `contains` says asked with its tile. Its
    /// trim loops are what `trim_loops` makes, once, when they are first
    /// asked, and its footprint `footprint`'s asked with its tile, `None`
    /// where none cuts it.
    pub fn new(init: FaceInit) -> Face {
        let loops: Option<Loops> = init.trim_loops.map(|make| Arc::new(Computed::new(move || make())));
        let copies = init
            .tiles
            .iter()
            .map(|&tile| {
                let turn = if tile.a * tile.d - tile.b * tile.c < 0.0 { -1.0 } else { 1.0 };
                let contains = init.contains.clone();
                FaceCopy {
                    tile,
                    outward: init.outward.map(|o| o * turn),
                    contains: Arc::new(move |p| contains(p, tile)),
                    loops: loops.clone(),
                    footprint: init.footprint.as_ref().and_then(|f| f(tile)),
                }
            })
            .collect();
        Face {
            id: init.id,
            domain: init.domain,
            map: init.map,
            copies,
            loops,
            foot: init.foot,
            bounding: init.bounding,
            truncation_excess: init.truncation_excess,
            lift: init.lift,
            height_gradient: init.height_gradient,
            second_order_jet: init.second_order_jet,
        }
    }

    /// A face in one place, on the identity.
    #[allow(clippy::too_many_arguments)] // Swift's initializer, argument for argument
    pub fn single(
        id: FaceID,
        domain: ParamRectangle,
        contains: ParamPredicate,
        trim_loops: Option<Arc<dyn Fn() -> ParamPaths + Send + Sync>>,
        map: JetMap,
        outward: Option<f64>,
        second_order_jet: Option<SecondOrderMap>,
        foot: FootMap,
        bounding: Bounding,
    ) -> Face {
        Face::new(FaceInit {
            id,
            domain,
            map,
            outward,
            tiles: vec![Affine2::IDENTITY],
            trim_loops,
            footprint: None,
            truncation_excess: None,
            lift: None,
            height_gradient: None,
            second_order_jet,
            foot,
            bounding,
            contains: Arc::new(move |p, _| contains(p)),
        })
    }

    /// The closed curves in the parameter plane that bound the face's trimmed
    /// domain even-odd, before a copy is cut to the footprint, each ending
    /// where it begins: where a contour on the face is cut, and which side of
    /// each is kept (`contours`). A polyhedron face's are its loops. The
    /// graph's are the domain's edge and the plateaus' loops, the truncation
    /// edge among them (`Surface::graph_trim_loops`), placed on f with a
    /// refiner, the first time they are asked, and shared by every copy. A
    /// copy ends at a tile boundary, which is the domain's edge, and the
    /// loops are closed along it as a band's are across a riser: the top of
    /// a riser at a tile boundary lies on them, carried back by the higher
    /// copy's tile.
    ///
    /// The graph's bound it only even-odd: a stretch along the domain's edge
    /// or a band boundary can be passed twice, bounding nothing, and the
    /// crease between a riser and the higher plateau lies on them though the
    /// graph does not reach it. They are not one oriented loop per component
    /// of the boundary, as a B-rep's are; and a copy's set
    /// (`FaceCopy::trim_loops`) bounds nothing on its own, for it also needs
    /// the footprint, which `contains` tests.
    ///
    /// `None` where the face has none to give: a parametric surface, whose
    /// domain is its whole rectangle, and, not built yet, a plateau, a riser
    /// and a wall.
    pub fn trim_loops(&self) -> Option<&ParamPaths> {
        self.loops.as_ref().map(|l| l.value())
    }

    /// The contours of a field on one copy of the face, clipped to its
    /// trimmed domain: cut where they cross the face's trim loops or the
    /// footprint's perimeter where the copy stands, kept where the loops
    /// hold them even-odd and the footprint holds them
    /// (`scalar_field::levels`). On a face with no trim loops they are kept
    /// where the copy's `contains` holds, a segment at a time. On the graph
    /// they are cut on the truncation edge exactly (`truncation_excess`), and
    /// a chord is refined where it sags off the graph as well as where it
    /// strays from the level set: the graph's parameter plane is the world's
    /// xy, so its sag is a length in the cell's own units.
    pub fn contours(&self, field: &ScalarField, levels: &[f64], copy: &FaceCopy) -> Contoured {
        let Some(loops) = self.trim_loops() else {
            return scalar_field::levels(
                field,
                levels,
                self.domain,
                &Trim { contains: Some(copy.contains.clone()), ..Trim::default() },
            );
        };
        scalar_field::levels(
            field,
            levels,
            self.domain,
            &Trim {
                trim_loops: Some(loops.clone()),
                contains: copy.footprint.as_ref().map(|f| f.1.clone()),
                cuts: copy.footprint.as_ref().map(|f| f.0.clone()).unwrap_or_default(),
                excess: self.truncation_excess.clone(),
                lift: self.lift.clone(),
            },
        )
    }

    /// The derivatives of the height in the parameter plane, where the face
    /// is the graph of a height (`heightGradient`).
    pub fn height_gradient(&self) -> Option<&ParamGradient> {
        self.height_gradient.as_ref()
    }
}

impl SceneGeometry {
    /// The faces of the solid the scene draws.
    pub fn faces(&self) -> Vec<Face> {
        match self {
            SceneGeometry::Heightfield(h) => h.faces(),
            SceneGeometry::Parametric(s) => vec![s.face()],
            SceneGeometry::Solid(s) => (0..s.faces.len()).map(|i| s.face(i)).collect(),
        }
    }

    /// The kind of face a layer's curves lie on, which is what their
    /// visibility is read from: the facing of that face under each point,
    /// then the scene's occlusion test. `None` for curves on no one face,
    /// tested by occlusion alone: a fold, edge-on by construction; a crease
    /// between two faces, as an outline, a truncation edge and a polyhedron's
    /// edges are; anything else stored with nothing said of where it lies.
    ///
    /// On a landscape a contour or a hatching lies on the faces it selects
    /// (`FaceSelector`): the contours of |f| and arg f on the graph, and
    /// over the truncation, lifted onto it, on a plateau, whose facing the
    /// graph's test reads. A wall's edges lie on the walls, and stored curves
    /// given the roles of |f| and arg f on the graph. The truncation edge,
    /// with the risers' tops and bottoms, is a crease. On a parametric
    /// surface every curve but an outline lies on its one face. On a
    /// polyhedron its hatching lies on its faces.
    pub fn face_of(&self, spec: &LayerSpec) -> Option<FaceKind> {
        if matches!(spec.source, LayerSource::FoldCurves) {
            return None;
        }
        match self {
            SceneGeometry::Heightfield(_) => match &spec.source {
                LayerSource::Contour { faces, .. }
                | LayerSource::Hatching { faces, .. }
                | LayerSource::IntegralCurves { faces, .. } => Some(faces.kind()),
                LayerSource::WallEdges { .. } => Some(FaceKind::Wall),
                _ => (spec.role == LayerRole::Magnitude || spec.role == LayerRole::Phase).then_some(FaceKind::Graph),
            },
            SceneGeometry::Parametric(_) => (spec.role != LayerRole::Outline).then_some(FaceKind::Surface),
            SceneGeometry::Solid(_) => match &spec.source {
                LayerSource::Hatching { faces: crate::hatching::FaceSelector::Solid, .. } => Some(FaceKind::Solid),
                _ => None,
            },
        }
    }
}

// MARK: - a parametric surface

impl ParametricSurface {
    /// The surface as one face: its whole parameter rectangle, glued where
    /// an axis is periodic.
    pub fn face(&self) -> Face {
        let (u, v) = (self.u, self.v);
        let surface = self.clone();
        let seams = self.clone();
        Face::single(
            FaceID::Surface,
            ParamRectangle::new(u, v),
            Arc::new(move |p| u.covers(p.x()) && v.covers(p.y())),
            None,
            self.map.clone(),
            self.outward,
            self.second_order_jet.clone(),
            Arc::new(move |q| surface.nearest_parameter(q)),
            Arc::new(move |_| seams.seams()),
        )
    }
}

// MARK: - a polyhedron

impl Solid {
    /// Face `i` as a `Face`: flat, its parameter rectangle the bounding box
    /// of its loops in its frame (`FaceFrame`), the map the frame's, and its
    /// trimmed domain what the loops bound, even-odd, so a hole is out. Its
    /// trim loops are its loops.
    ///
    /// Membership is read off the loops, not the convex pieces: the pieces
    /// are how visibility and hatching cut the face, and the face is what
    /// they cut. A flat face's fields are linear in its frame, so the
    /// rectangle is sampled at its corners.
    pub fn face(&self, i: usize) -> Face {
        let frame = self.frames[i];
        let loops: Vec<Vec<Vec2>> =
            self.loops[i].iter().map(|l| l.iter().map(|&k| frame.coordinates(self.vertices[k].v)).collect()).collect();
        let corners: Vec<Vec2> = loops.iter().flatten().copied().collect();
        let lo = corners.iter().fold(corners[0], |a, &b| a.min(b));
        let hi = corners.iter().fold(corners[0], |a, &b| a.max(b));
        let (u, v) = (frame.u, frame.v);
        let closed: ParamPaths =
            loops.iter().map(|l| l.iter().chain(std::iter::once(&l[0])).map(|&p| P2::from_vec(p)).collect()).collect();
        let membership = loops.clone();
        let trim = closed.clone();
        let bound = closed;
        let map_frame = frame;
        Face::single(
            FaceID::Solid { face: i },
            ParamRectangle::new(
                ParamAxis::new(Interval::new(lo.x, hi.x), false, 2),
                ParamAxis::new(Interval::new(lo.y, hi.y), false, 2),
            ),
            Arc::new(move |p| {
                // A ray from p along +u crosses the loops an odd number of
                // times exactly when p is inside; a side is half open in v,
                // so a ray through a corner counts it once.
                let mut inside = false;
                for l in &membership {
                    for k in 0..l.len() {
                        let (a, b) = (l[k], l[(k + 1) % l.len()]);
                        if (a.y > p.y()) == (b.y > p.y()) {
                            continue;
                        }
                        if p.x() < a.x + (p.y() - a.y) * (b.x - a.x) / (b.y - a.y) {
                            inside = !inside;
                        }
                    }
                }
                inside
            }),
            Some(Arc::new(move || trim.clone())),
            Arc::new(move |p| SurfaceJet::new(map_frame.point(p.v), u, v)),
            Some(1.0),
            None,
            Arc::new(move |q| P2::from_vec(frame.coordinates(q))),
            Arc::new(move |_| bound.clone()),
        )
    }
}

// MARK: - a landscape

impl Heightfield {
    /// The landscape's faces: `Surface::faces` over its tiles, footprint and
    /// walls, placed on f where it has the function.
    pub fn faces(&self) -> Vec<Face> {
        self.surface.faces(&self.tiles, &self.region, &self.walls, self.refine.as_ref())
    }
}

impl Surface {
    /// The faces of the landscape: the graph, a plateau per band of the
    /// truncation, a riser wherever the truncation steps between two bands, a
    /// riser either way up at each tile boundary where two copies meet, and a
    /// wall along each edge of the walls' perimeter, in that order. Nothing
    /// for a grid too small to span its domain.
    ///
    /// The graph, the plateaus and the risers lie over the domain and are
    /// copied by the tiles: contoured once, then replicated, as the curves on
    /// them are. Each copy is cut to the footprint where it stands in space,
    /// as the mesh and visibility cut it. The walls stand on the perimeter in
    /// space, their top edge read through whichever tile is under it. Walls
    /// given as a mesh describe no face.
    ///
    /// The faces bound the solid the drawing tests curves against. With a
    /// refiner they are f's: the graph is |f| and every membership that
    /// compares |f| with the truncation reads f. Without one the grid is the
    /// surface. Where |f| is not a number it counts as over any truncation,
    /// as the facing reads it.
    pub fn faces(
        &self,
        tiles: &[Affine2],
        region: &Region,
        walls: &Walls,
        refine: Option<&ContourRefine>,
    ) -> Vec<Face> {
        let Some(graph) = self.graph(tiles, region, refine) else { return Vec::new() };
        let mut out = vec![graph];
        out.extend(self.plateaus(tiles, region, refine));
        out.extend(self.risers(tiles, region, refine));
        out.extend(self.tile_boundary_risers(tiles, region, refine));
        out.extend(self.walls(walls, tiles, refine));
        out
    }

    /// The parameter rectangle of the graph and the plateaus: the domain,
    /// sampled as the height grid is. `None` when the grid has fewer than two
    /// samples along an axis, too few to span it.
    pub fn parameter_rectangle(&self) -> Option<ParamRectangle> {
        if self.height.width < 2 || self.height.height < 2 {
            return None;
        }
        let d = self.domain();
        Some(ParamRectangle::new(
            ParamAxis::new(d.real, false, self.height.width),
            ParamAxis::new(d.imag, false, self.height.height),
        ))
    }

    /// The graph, where z = |f| under the truncation: over the parameter
    /// rectangle, its trimmed domain where |f| is under the truncation
    /// height, in the footprint. Its trim loops are the domain's edge and the
    /// plateaus' loops (`graph_trim_loops`), each copy's with the footprint's
    /// perimeter carried back through its tile. `None` when the grid is too
    /// small to span its domain. Its derivatives are the slope the graph's
    /// normal is read from (`graph_normal`), on the face's own domain, before
    /// a copy moves it, so `du × dv` is the normal the facing reads: f's own,
    /// untruncated, with a refiner; without one the grid's chords, truncated.
    pub fn graph(&self, tiles: &[Affine2], region: &Region, refine: Option<&ContourRefine>) -> Option<Face> {
        let rectangle = self.parameter_rectangle()?;
        let magnitude = self.face_magnitude(refine);
        let loops = self.graph_loops(refine);
        let mut excess: Option<ParamScalar> = None;
        if !matches!(self.truncation, crate::manifest::Truncation::None)
            && refine.is_some_and(|r| r.magnitude.is_some())
        {
            let (surface, magnitude) = (self.clone(), magnitude.clone());
            excess = Some(Arc::new(move |p: P2<ParamSpace>| {
                let h = surface.truncation_height_at(p.x());
                if h.is_finite() { magnitude(P2::new(p.x(), p.y())) - h } else { f64::NEG_INFINITY }
            }));
        }
        let refine = refine.cloned();
        let (s1, r1, m1) = (self.clone(), refine.clone(), magnitude.clone());
        let (s2, r2) = (self.clone(), refine.clone());
        let m3 = magnitude.clone();
        let (s4, l4, region4) = (self.clone(), loops.clone(), region.clone());
        let (s5, region5, m5) = (self.clone(), region.clone(), magnitude);
        let l6 = loops;
        let region7 = region.clone();
        Some(Face::new(FaceInit {
            id: FaceID::Graph,
            domain: rectangle,
            map: Arc::new(move |p| {
                let n = s1.graph_normal(P2::new(p.x(), p.y()), &[Affine2::IDENTITY], r1.as_ref());
                SurfaceJet::new(
                    Vec3::new(p.x(), p.y(), m1(P2::new(p.x(), p.y()))),
                    Vec3::new(1.0, 0.0, -n.x),
                    Vec3::new(0.0, 1.0, -n.y),
                )
            }),
            outward: Some(1.0),
            tiles: tiles.to_vec(),
            trim_loops: Some(Arc::new(move || l6.value().clone())),
            footprint: Some(Arc::new(move |tile| Surface::footprint(&region7, tile))),
            truncation_excess: excess,
            lift: Some(Arc::new(move |p| Vec3::new(p.x(), p.y(), m3(P2::new(p.x(), p.y()))))),
            height_gradient: Some(Arc::new(move |p| {
                let n = s2.graph_normal(P2::new(p.x(), p.y()), &[Affine2::IDENTITY], r2.as_ref());
                Vec2::new(-n.x, -n.y)
            })),
            second_order_jet: None,
            foot: Arc::new(|q| P2::new(q.x, q.y)),
            bounding: Arc::new(move |tile| s4.top_boundary(FaceID::Graph, l4.value(), &region4, tile)),
            contains: Arc::new(move |p, tile| {
                rectangle.u.covers(p.x())
                    && rectangle.v.covers(p.y())
                    && region5.contains(tile.apply(P3::new(p.x(), p.y(), 0.0)).xy())
                    && m5(P2::new(p.x(), p.y())) < s5.truncation_height_at(p.x())
            }),
        }))
    }

    /// A plateau per band of the truncation that meets the domain and has a
    /// finite height: over the domain, where |f| reaches the band's height
    /// in the band, the domain's right edge in the band before it
    /// (`band_at_from`).
    pub fn plateaus(&self, tiles: &[Affine2], region: &Region, refine: Option<&ContourRefine>) -> Vec<Face> {
        let Some(rectangle) = self.parameter_rectangle() else { return Vec::new() };
        let magnitude = self.face_magnitude(refine);
        let real = self.domain().real;
        let loops = self.graph_loops(refine);
        self.truncation
            .bands(swift::min(real.lo, real.hi), swift::max(real.lo, real.hi))
            .into_iter()
            .filter_map(|span| {
                let (h, k) = (span.height, span.band);
                if !h.is_finite() {
                    return None;
                }
                let (s1, l1, region1) = (self.clone(), loops.clone(), region.clone());
                let (s2, region2, m2) = (self.clone(), region.clone(), magnitude.clone());
                Some(Face::new(FaceInit {
                    id: FaceID::Plateau { band: k },
                    domain: rectangle,
                    map: Arc::new(move |p| {
                        SurfaceJet::new(Vec3::new(p.x(), p.y(), h), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
                    }),
                    outward: Some(1.0),
                    tiles: tiles.to_vec(),
                    trim_loops: None,
                    footprint: None,
                    truncation_excess: None,
                    lift: None,
                    height_gradient: None,
                    second_order_jet: None,
                    foot: Arc::new(|q| P2::new(q.x, q.y)),
                    bounding: Arc::new(move |tile| {
                        s1.top_boundary(FaceID::Plateau { band: k }, l1.value(), &region1, tile)
                    }),
                    contains: Arc::new(move |p, tile| {
                        rectangle.u.covers(p.x())
                            && rectangle.v.covers(p.y())
                            && region2.contains(tile.apply(P3::new(p.x(), p.y(), 0.0)).xy())
                            && s2.band_at(p.x()) == Some(k)
                            && !(m2(P2::new(p.x(), p.y())) < h)
                    }),
                }))
            })
            .collect()
    }

    /// A riser at each boundary x = b inside the domain between two bands of
    /// different heights: the vertical face there, its parameters (y, z),
    /// from the lesser height up to |f| or the greater, whichever is lower,
    /// facing the lower band. It stands where the footprint holds both
    /// sides of the boundary, read a hair either side of it. Where it holds
    /// one, a wall on the boundary reads the band on the footprint's side
    /// (`walls`) and rises to the solid's top there, and there is no drop to
    /// face. A boundary on the domain's edge has none: the band beyond the
    /// far edge starts outside the domain (`truncation_height_at_from`).
    pub fn risers(&self, tiles: &[Affine2], region: &Region, refine: Option<&ContourRefine>) -> Vec<Face> {
        let Some(rectangle) = self.parameter_rectangle() else { return Vec::new() };
        let magnitude = self.face_magnitude(refine);
        let real = self.domain().real;
        let spans = self.truncation.bands(swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
        let highest = self.highest_sample();
        let side = 1e-9 * self.cell();
        let y = rectangle.v;
        let tolerance = self.profile_tolerance(refine);
        spans
            .windows(2)
            .filter_map(|pair| {
                let (left, right) = (pair[0], pair[1]);
                let b = left.hi;
                let lesser = swift::min(left.height, right.height);
                let greater = swift::max(left.height, right.height);
                if !(lesser.is_finite() && lesser < greater) {
                    return None;
                }
                let z = ParamAxis::new(
                    Interval::new(lesser, if greater.is_finite() { greater } else { swift::max(highest, lesser) }),
                    false,
                    2,
                );
                let (m, region_in) = (magnitude.clone(), region.clone());
                let in_riser: TileContains = Arc::new(move |p: P2<ParamSpace>, tile: Affine2| {
                    if !(y.covers(p.x()) && p.y() >= lesser && p.y() <= swift::min(m(P2::new(b, p.x())), greater)) {
                        return false;
                    }
                    region_in.contains(tile.apply(P3::new(b - side, p.x(), 0.0)).xy())
                        && region_in.contains(tile.apply(P3::new(b + side, p.x(), 0.0)).xy())
                });
                let (in_riser_b, region_b, m_b) = (in_riser.clone(), region.clone(), magnitude.clone());
                Some(Face::new(FaceInit {
                    id: FaceID::Riser { band: left.band },
                    domain: ParamRectangle::new(y, z),
                    map: Arc::new(move |p| {
                        SurfaceJet::new(Vec3::new(b, p.x(), p.y()), Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0))
                    }),
                    outward: Some(if left.height > right.height { 1.0 } else { -1.0 }),
                    tiles: tiles.to_vec(),
                    trim_loops: None,
                    footprint: None,
                    truncation_excess: None,
                    lift: None,
                    height_gradient: None,
                    second_order_jet: None,
                    foot: Arc::new(|q| P2::new(q.y, q.z)),
                    // Where it stands along its boundary, the rows and where the
                    // footprint's perimeter crosses the boundary seeding the search,
                    // from the lesser height up to its top.
                    bounding: Arc::new(move |tile| {
                        let stops: Vec<f64> = Surface::footprint(&region_b, tile)
                            .map(|f| f.0)
                            .unwrap_or_default()
                            .iter()
                            .flat_map(|l| Surface::crossings(l, b))
                            .collect();
                        let mut stations: Vec<f64> = (0..y.samples)
                            .map(|i| y.coordinate(i as i64))
                            .chain(stops)
                            .filter(|&s| y.covers(s))
                            .collect();
                        sort_doubles(&mut stations);
                        Surface::column_loops(
                            &stations,
                            side,
                            tolerance,
                            &|s| in_riser_b(P2::new(s, lesser), tile),
                            &|_| lesser,
                            &|s| swift::min(swift::min(m_b(P2::new(b, s)), greater), z.range.hi),
                        )
                    }),
                    contains: in_riser,
                }))
            })
            .collect()
    }

    /// A riser at each tile boundary where two copies meet
    /// (`tile_boundaries`), one each way up: the vertical face there, its
    /// parameters the distance along the tile boundary and the height, from
    /// the lower copy's top up to the higher's, facing the lower copy. Each
    /// copy's top is read on the tile boundary through its own tile, at the
    /// point of the domain's edge the tile boundary point is the copy of
    /// (`tile_boundary_top`), and the riser stands where one stands over the
    /// other (`stands`): a turned, translated or glide-reflected neighbour's
    /// top can, one reflected across the tile boundary never does, reading
    /// the same points of the domain. As at a band boundary, it stands where
    /// the footprint holds both sides of the tile boundary, read a hair
    /// either side of it; where it holds one, a wall on the tile boundary
    /// bounds the copy its footprint is in (`wall_tile`) and rises to that
    /// copy's top. The parameter rectangle runs from 0, under any |f|, to
    /// the highest truncation, or the grid's highest sample without one.
    pub fn tile_boundary_risers(
        &self,
        tiles: &[Affine2],
        region: &Region,
        refine: Option<&ContourRefine>,
    ) -> Vec<Face> {
        let magnitude = self.face_magnitude(refine);
        let side = 1e-9 * self.cell();
        let top = if self.truncation.highest().is_finite() { self.truncation.highest() } else { self.highest_sample() };
        let mut out: Vec<Face> = Vec::new();
        for boundary in self.tile_boundaries(tiles) {
            let (Some(first_back), Some(second_back)) =
                (tiles[boundary.first].inverse(), tiles[boundary.second].inverse())
            else {
                continue;
            };
            let length = boundary.length();
            let n = boundary.normal();
            let samples = swift::int((length / self.cell()).ceil()).map_or(2, |k| (k + 1).max(2) as usize);
            let s = ParamAxis::new(Interval::new(0.0, length), false, samples);
            let along = (boundary.end.v - boundary.start.v) / length;
            // Which side of the tile boundary the second copy is on: its tile's
            // image of the domain's centre.
            let d = self.domain();
            let centre = tiles[boundary.second]
                .apply(P3::new(0.5 * (d.real.lo + d.real.hi), 0.5 * (d.imag.lo + d.imag.hi), 0.0))
                .xy()
                .v;
            let second_left = (centre - boundary.start.v).dot(n) > 0.0;
            let tolerance = self.profile_tolerance(refine);
            // The stations a riser's boundary is found from along the tile
            // boundary: the grid's spacing, and where the footprint's
            // perimeter crosses it.
            let start = boundary.start.v;
            let stops: Vec<f64> = Surface::footprint(region, Affine2::IDENTITY)
                .map(|f| f.0)
                .unwrap_or_default()
                .iter()
                .flat_map(|l| {
                    l.windows(2)
                        .filter_map(|w| {
                            let (a, c) = (w[0], w[1]);
                            let e = c.v - a.v;
                            let r = a.v - start;
                            let det = along.x * e.y - along.y * e.x;
                            if det == 0.0 {
                                return None;
                            }
                            let t = (r.x * e.y - r.y * e.x) / det;
                            let u = (r.x * along.y - r.y * along.x) / det;
                            ((0.0..=1.0).contains(&u) && t > 0.0 && t < length).then_some(t)
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();
            let mut stations: Vec<f64> = (0..s.samples).map(|i| s.coordinate(i as i64)).chain(stops).collect();
            sort_doubles(&mut stations);
            let pairs = [
                (boundary.first, first_back, boundary.second, second_back, !second_left),
                (boundary.second, second_back, boundary.first, first_back, second_left),
            ];
            for (higher, up, lower, down, lower_right) in pairs {
                let surface = self.clone();
                let m = magnitude.clone();
                let high = Arc::new(move |s: f64| surface.tile_boundary_top(boundary.point(s), up, &*m));
                let surface = self.clone();
                let m = magnitude.clone();
                let low = Arc::new(move |s: f64| surface.tile_boundary_top(boundary.point(s), down, &*m));
                let (h1, l1, region1) = (high.clone(), low.clone(), region.clone());
                let in_riser: ParamPredicate = Arc::new(move |p: P2<ParamSpace>| {
                    if !s.covers(p.x()) {
                        return false;
                    }
                    let w = boundary.point(p.x());
                    if !(region1.contains(P2::from_vec(w.v - side * n))
                        && region1.contains(P2::from_vec(w.v + side * n)))
                    {
                        return false;
                    }
                    let (high, low) = (h1(p.x()), l1(p.x()));
                    Surface::stands(high, low) && p.y() >= low && p.y() <= high
                });
                let (in_b, l2, h2, st) = (in_riser.clone(), low.clone(), high.clone(), stations.clone());
                out.push(Face::single(
                    FaceID::TileBoundaryRiser { tile: higher, facing: lower },
                    ParamRectangle::new(s, ParamAxis::new(Interval::new(0.0, swift::max(top, 0.0)), false, 2)),
                    in_riser,
                    None,
                    Arc::new(move |p| {
                        let w = boundary.point(p.x());
                        SurfaceJet::new(
                            Vec3::new(w.x(), w.y(), p.y()),
                            Vec3::new(along.x, along.y, 0.0),
                            Vec3::new(0.0, 0.0, 1.0),
                        )
                    }),
                    // du × dv is the right of the tile boundary's direction.
                    Some(if lower_right { 1.0 } else { -1.0 }),
                    None,
                    Arc::new(move |q| P2::new((Vec2::new(q.x, q.y) - boundary.start.v).dot(along), q.z)),
                    // Where it stands along the tile boundary, from the lower
                    // copy's top up to the higher's.
                    Arc::new(move |_| {
                        Surface::column_loops(&st, side, tolerance, &|x| in_b(P2::new(x, l2(x))), &|x| l2(x), &|x| {
                            h2(x)
                        })
                    }),
                ));
            }
        }
        out.sort_by_key(|f| f.id);
        out
    }

    /// A wall along each edge of the perimeter: its parameters the distance
    /// along the edge and the height, from the base up to the top edge, the
    /// truncated surface over the edge read through the tile under it, as
    /// the wall's curtain reads it (`Mesh::wall_curtain`), here exact rather
    /// than sampled: the truncation read from the footprint's side
    /// (`Surface::footprint_side`). The point on the edge is `a + (b - a) t`,
    /// which keeps a vertical edge's x, so a wall standing on a band's
    /// boundary reads the band its footprint is in, all along it. The sign
    /// of the perimeter's signed area says which side is out:
    /// counter-clockwise from above, `du × dv` is.
    pub fn walls(&self, walls: &Walls, tiles: &[Affine2], refine: Option<&ContourRefine>) -> Vec<Face> {
        let Walls::Perimeter { perimeter, base } = walls else { return Vec::new() };
        let base = *base;
        let magnitude = self.face_magnitude(refine);
        let area = perimeter.signed_area();
        let top = if self.truncation.highest().is_finite() { self.truncation.highest() } else { self.highest_sample() };
        let refine = refine.cloned();
        perimeter
            .edges()
            .iter()
            .enumerate()
            .filter_map(|(k, edge)| {
                let edge = *edge;
                let along = edge.end.v - edge.start.v;
                let length = along.length();
                if !(length > 0.0) {
                    return None;
                }
                let s = ParamAxis::new(Interval::new(0.0, length), false, edge.density.max(2) as usize);
                let inward = perimeter.inward(k);
                let foot = move |distance: f64| -> P2<DomainSpace> {
                    P2::from_vec(edge.start.v + along * (distance / length))
                };
                let unit = along / length;
                let (grid, m, tiles_c) = (self.clone(), magnitude.clone(), tiles.to_vec());
                let (grid_b, tiles_b, refine_b) = (self.clone(), tiles.to_vec(), refine.clone());
                Some(Face::single(
                    FaceID::Wall { edge: k },
                    ParamRectangle::new(s, ParamAxis::new(Interval::new(base, swift::max(top, base)), false, 2)),
                    Arc::new(move |p| {
                        if !(s.covers(p.x()) && p.y() >= base) {
                            return false;
                        }
                        let (point, truncation_height) =
                            grid.wall_foot(foot(p.x()), p.x() / length, along, inward, &tiles_c, None);
                        p.y() <= swift::min(m(point), truncation_height)
                    }),
                    None,
                    Arc::new(move |p| {
                        let q = foot(p.x());
                        SurfaceJet::new(
                            Vec3::new(q.x(), q.y(), p.y()),
                            Vec3::new(along.x / length, along.y / length, 0.0),
                            Vec3::new(0.0, 0.0, 1.0),
                        )
                    }),
                    Some(if area < 0.0 { -1.0 } else { 1.0 }),
                    None,
                    Arc::new(move |q| P2::new((Vec2::new(q.x, q.y) - edge.start.v).dot(unit), q.z)),
                    // Its bottom, its far end, its top edge as its layer draws it,
                    // back along, and its near end.
                    Arc::new(move |_| {
                        let top_edge = grid_b.top_edge(&edge, inward, &tiles_b, refine_b.as_ref());
                        let mut lp = vec![P2::new(0.0, base), P2::new(length, base)];
                        lp.extend(top_edge.iter().rev().map(|v| P2::new((v.xy().v - edge.start.v).dot(unit), v.z())));
                        lp.push(P2::new(0.0, base));
                        vec![lp]
                    }),
                ))
            })
            .collect()
    }

    /// The footprint where a copy on `tile` stands, in the parameter plane:
    /// its perimeter carried back through the tile, closed, and whether it
    /// holds the point of space a point of the plane lands on. `None` where
    /// there is no footprint. A tile with no inverse carries no perimeter
    /// back, and the footprint is asked a point at a time.
    pub fn footprint(region: &Region, tile: Affine2) -> Option<Footprint> {
        let Region::Inside(perimeter) = region else { return None };
        let corners: Vec<P2<ParamSpace>> = tile
            .inverse()
            .map(|back| {
                perimeter
                    .edges()
                    .iter()
                    .map(|e| {
                        let q = back.apply(P3::new(e.start.x(), e.start.y(), 0.0));
                        P2::new(q.x(), q.y())
                    })
                    .collect()
            })
            .unwrap_or_default();
        let loops = if corners.is_empty() {
            Vec::new()
        } else {
            let mut c = corners.clone();
            c.push(corners[0]);
            vec![c]
        };
        let region = region.clone();
        Some((loops, Arc::new(move |p| region.contains(tile.apply(P3::new(p.x(), p.y(), 0.0)).xy()))))
    }

    /// |f| at a point of the domain, untruncated, as the faces read it: f's
    /// own with a refiner, the grid's without, and over any truncation
    /// where it is not a number.
    pub fn face_magnitude(&self, refine: Option<&ContourRefine>) -> Arc<dyn Fn(P2<DomainSpace>) -> f64 + Send + Sync> {
        let grid = self.clone();
        let f = refine.and_then(|r| r.magnitude.clone());
        Arc::new(move |p| {
            let u = match &f {
                Some(f) => f(p),
                None => grid.magnitude_at(p),
            };
            if u.is_nan() { f64::INFINITY } else { u }
        })
    }

    /// The greatest finite sample of the height grid, or 0. (Swift's `max()`
    /// over the finite samples, which keeps the first of equals.)
    pub fn highest_sample(&self) -> f64 {
        let mut best: Option<f32> = None;
        for &v in self.height.values.iter().filter(|v| v.is_finite()) {
            if best.is_none_or(|b| b < v) {
                best = Some(v);
            }
        }
        f64::from(best.unwrap_or(0.0))
    }

    /// The graph's trim loops (`graph_trim_loops`), in the parameter
    /// plane, made the first time they are asked.
    pub fn graph_loops(&self, refine: Option<&ContourRefine>) -> Loops {
        let surface = self.clone();
        let refine = refine.cloned();
        Arc::new(Computed::new(move || {
            surface
                .graph_trim_loops(refine.as_ref())
                .loops
                .iter()
                .map(|l| l.iter().map(|p| P2::from_vec(p.v)).collect())
                .collect()
        }))
    }

    /// The boundary of a copy of the graph or of a plateau on `tile`, in the
    /// parameter plane, as the graph's trim loops `loops` bound it: the
    /// trimmed domain read even-odd off them, as a contour on the face is
    /// clipped (`Face::contours`). The graph's is the region the loops bound,
    /// a plateau's the region its band's plateau loops bound in its band,
    /// each cut to the footprint where the copy stands. Its pieces are those
    /// of the loops, of each band boundary inside the domain, across it, and
    /// of the footprint's perimeter, each cut where another crosses it
    /// (`Face::pieces`), and kept where the face lies a ten-millionth of a
    /// cell on one side and not the other: a stretch two loops pass,
    /// bounding nothing, is left out, and so is a band boundary or a stretch
    /// of the perimeter the face does not reach. Ends within a millionth of a
    /// cell are one (`Face::snapped`).
    pub fn top_boundary(&self, id: FaceID, loops: &ParamPaths, region: &Region, tile: Affine2) -> ParamPaths {
        let index =
            TrimIndex::new(if id == FaceID::Graph { loops } else { &loops[loops.len().min(1)..] }, (None, None));
        let d = self.domain();
        let (real, imag) = (d.real, d.imag);
        let (y0, y1) = (swift::min(imag.lo, imag.hi), swift::max(imag.lo, imag.hi));
        let spans = self.truncation.bands(swift::min(real.lo, real.hi), swift::max(real.lo, real.hi));
        let boundaries: Vec<Vec<Vec2>> =
            spans.windows(2).map(|w| vec![Vec2::new(w[0].hi, y0), Vec2::new(w[0].hi, y1)]).collect();
        let perimeter: Vec<Vec<Vec2>> = Surface::footprint(region, tile)
            .map(|f| f.0)
            .unwrap_or_default()
            .iter()
            .map(|l| l.iter().map(|p| p.v).collect())
            .collect();
        let holds = |p: Vec2| -> bool {
            if !(index.encloses(p) && region.contains(tile.apply(P3::new(p.x, p.y, 0.0)).xy())) {
                return false;
            }
            if let FaceID::Plateau { band } = id {
                return self.band_at(p.x) == Some(band);
            }
            true
        };
        let h = 1e-7 * self.cell();
        let mut curves: Vec<Vec<Vec2>> = loops.iter().map(|l| l.iter().map(|p| p.v).collect()).collect();
        curves.extend(boundaries);
        curves.extend(perimeter);
        let kept: Vec<(Vec2, Vec2)> = Face::pieces(&curves, 1e-9 * self.cell())
            .into_iter()
            .filter(|&(a, b)| {
                let d = b - a;
                let length = d.length();
                if !(length > 0.0) {
                    return false;
                }
                let m = 0.5 * (a + b);
                let n = Vec2::new(-d.y, d.x) / length;
                holds(m + h * n) != holds(m - h * n)
            })
            .collect();
        Face::chained(&Face::snapped(&kept, 1e-6 * self.cell()))
            .into_iter()
            .map(|c| c.into_iter().map(P2::from_vec).collect())
            .collect()
    }

    /// The boundary of a vertical face whose trimmed domain is, at each
    /// point along it, an interval from `bottom` up to `top`: a closed loop
    /// round each stretch where it `stands`. The stretches are found at
    /// `stations`, each end where the verdict turns between two placed by
    /// bisection to within `eps`; a loop runs along the bottom, up the far
    /// end, back along the top and down the near end, the bottom and the
    /// top through the stations between and subdivided to `tolerance`
    /// (`profile`).
    pub fn column_loops(
        stations: &[f64],
        eps: f64,
        tolerance: f64,
        stands: &dyn Fn(f64) -> bool,
        bottom: &dyn Fn(f64) -> f64,
        top: &dyn Fn(f64) -> f64,
    ) -> ParamPaths {
        let Some(&first) = stations.first() else { return Vec::new() };
        let mut stretches: Vec<(f64, f64)> = Vec::new();
        let mut previous = first;
        let mut standing = stands(first);
        let mut start: Option<f64> = standing.then_some(first);
        for &s in &stations[1..] {
            if !(s > previous) {
                continue;
            }
            let verdict = stands(s);
            if verdict != standing {
                let (mut a, mut c) = (previous, s);
                while c - a > eps {
                    let m = 0.5 * (a + c);
                    if !(m > a && m < c) {
                        break;
                    }
                    if stands(m) == standing {
                        a = m;
                    } else {
                        c = m;
                    }
                }
                if let (Some(lo), true) = (start, standing) {
                    stretches.push((lo, a));
                    start = None;
                } else {
                    start = Some(c);
                }
            }
            previous = s;
            standing = verdict;
        }
        if let (Some(lo), true) = (start, standing) {
            stretches.push((lo, previous));
        }
        stretches
            .into_iter()
            .filter_map(|(lo, hi)| {
                if !(hi > lo) {
                    return None;
                }
                let mut along = vec![lo];
                along.extend(stations.iter().copied().filter(|&s| s > lo && s < hi));
                along.push(hi);
                let under = Surface::profile(&along, tolerance, bottom);
                let over = Surface::profile(&along, tolerance, top);
                let mut lp = under.clone();
                lp.extend(over.into_iter().rev());
                lp.push(under[0]);
                Some(lp)
            })
            .collect()
    }

    /// How far a profile along a riser's line may depart from its chord: the
    /// refiner's tolerance on f, and on the grid, whose heights are linear
    /// between its rows, a millionth of a cell.
    pub fn profile_tolerance(&self, refine: Option<&ContourRefine>) -> f64 {
        let t = match refine {
            Some(r) if r.magnitude.is_some() => r.tolerance,
            _ => 1e-6,
        };
        t * self.cell()
    }

    /// The heights where a closed polyline of the domain crosses the line
    /// Re z = `x`.
    pub fn crossings(lp: &[P2<ParamSpace>], x: f64) -> Vec<f64> {
        lp.windows(2)
            .filter_map(|w| {
                let (a, b) = (w[0], w[1]);
                if a.x() == b.x() || !((a.x() - x) * (b.x() - x) <= 0.0) {
                    return None;
                }
                Some(a.y() + (x - a.x()) / (b.x() - a.x()) * (b.y() - a.y()))
            })
            .collect()
    }

    /// The curve `(s, top(s))` through `stations`, each chord halved until
    /// its midpoint lies within `tolerance` of the curve, ten deep.
    pub fn profile(stations: &[f64], tolerance: f64, top: &dyn Fn(f64) -> f64) -> Vec<P2<ParamSpace>> {
        #[allow(clippy::too_many_arguments)] // a recursion carrying its two ends
        fn subdivide(
            out: &mut Vec<P2<ParamSpace>>,
            top: &dyn Fn(f64) -> f64,
            tolerance: f64,
            s0: f64,
            z0: f64,
            s1: f64,
            z1: f64,
            depth: usize,
        ) {
            let sm = 0.5 * (s0 + s1);
            let zm = top(sm);
            if !(depth < 10 && sm > s0 && sm < s1 && (zm - 0.5 * (z0 + z1)).abs() > tolerance) {
                return;
            }
            subdivide(out, top, tolerance, s0, z0, sm, zm, depth + 1);
            out.push(P2::new(sm, zm));
            subdivide(out, top, tolerance, sm, zm, s1, z1, depth + 1);
        }
        let mut out: Vec<P2<ParamSpace>> = Vec::new();
        for (k, &s) in stations.iter().enumerate() {
            let z = top(s);
            if k > 0
                && let Some(&last) = out.last()
            {
                subdivide(&mut out, top, tolerance, last.x(), last.y(), s, z, 0);
            }
            out.push(P2::new(s, z));
        }
        out
    }
}

/// Swift's `sorted()` on doubles: ascending by `<`, stable. A NaN compares
/// false both ways and so stays where the sort leaves it, as there (Swift's
/// sort and Rust's differ in where that is; no station is a NaN).
fn sort_doubles(v: &mut [f64]) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}
