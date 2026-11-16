//! Polylines in compressed sparse row form: the port of
//! `Sources/KurvenCore/Polyline.swift`.

use crate::manifest::LayerSpec;
use crate::simd::Vec2;
use crate::spaces::{AABB, P2, P3, ParamSpace, Transform, WorldSpace};

/// A set of polylines in compressed sparse row form: one flat vertex array plus
/// path offsets.
///
/// Path identity is structural. The Python pipeline tags each vertex with a
/// running integer and recovers paths with `groupby`, which forces ad-hoc offset
/// arithmetic (`BUMP = 1_000_000`, `100 * (i + 2)`) at every place two families
/// of curves are concatenated, purely to stop unrelated runs from welding.
/// Offsets cannot weld.
pub struct PolylineSet<S> {
    /// All vertices of all paths, in order.
    pub vertices: Vec<P3<S>>,
    /// `count + 1` offsets; path `i` is `vertices[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<usize>,
    /// Where on a surface each vertex lies, parallel to `vertices`, for curves
    /// drawn *on* a parametric surface. `None` for curves that only have a position.
    ///
    /// A position says where the curves are; a surface coordinate says which piece
    /// of the surface it belongs to, and that is the question hidden-line
    /// removal actually asks. Two sheets a hair apart in depth are far apart
    /// in coordinates. Kept unwrapped along periodic axes, so a path that
    /// crosses a seam does not jump a whole period between two vertices.
    pub coords: Option<Vec<P2<ParamSpace>>>,
}

impl<S> Clone for PolylineSet<S> {
    fn clone(&self) -> Self {
        PolylineSet { vertices: self.vertices.clone(), offsets: self.offsets.clone(), coords: self.coords.clone() }
    }
}
impl<S> PartialEq for PolylineSet<S> {
    fn eq(&self, o: &Self) -> bool {
        self.vertices == o.vertices && self.offsets == o.offsets && self.coords == o.coords
    }
}
impl<S> std::fmt::Debug for PolylineSet<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolylineSet")
            .field("vertices", &self.vertices)
            .field("offsets", &self.offsets)
            .field("coords", &self.coords)
            .finish()
    }
}

impl<S> PolylineSet<S> {
    pub fn new(vertices: Vec<P3<S>>, offsets: Vec<usize>, coords: Option<Vec<P2<ParamSpace>>>) -> PolylineSet<S> {
        assert!(offsets.first() == Some(&0) || vertices.is_empty(), "CSR offsets must start at 0");
        let last = offsets.last().copied().unwrap_or(0);
        assert!(last == vertices.len(), "CSR offsets end at {last}, {} vertices", vertices.len());
        if let Some(c) = &coords {
            assert!(c.len() == vertices.len(), "{} surface coordinates for {} vertices", c.len(), vertices.len());
        }
        PolylineSet { vertices, offsets, coords }
    }

    /// Paths of at least two vertices each; a shorter one draws nothing
    /// and is dropped.
    pub fn from_paths<P: AsRef<[P3<S>]>>(paths: &[P]) -> PolylineSet<S> {
        let mut vertices = Vec::new();
        let mut offsets = vec![0];
        for p in paths.iter().map(AsRef::as_ref).filter(|p| p.len() >= 2) {
            vertices.extend_from_slice(p);
            offsets.push(vertices.len());
        }
        PolylineSet { vertices, offsets, coords: None }
    }

    /// Paths with a surface coordinate for every vertex.
    pub fn from_paths_with_coords(paths: &[Vec<P3<S>>], coords: &[Vec<P2<ParamSpace>>]) -> PolylineSet<S> {
        assert!(paths.len() == coords.len(), "one coordinate path per path");
        let mut vertices = Vec::new();
        let mut cs = Vec::new();
        let mut offsets = vec![0];
        for (p, c) in paths.iter().zip(coords).filter(|(p, _)| p.len() >= 2) {
            assert!(p.len() == c.len(), "one coordinate per vertex");
            vertices.extend_from_slice(p);
            cs.extend_from_slice(c);
            offsets.push(vertices.len());
        }
        PolylineSet { vertices, offsets, coords: Some(cs) }
    }

    pub fn empty() -> PolylineSet<S> {
        PolylineSet { vertices: Vec::new(), offsets: vec![0], coords: None }
    }

    pub fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Path `i`: Swift's `self[path: i]`.
    pub fn path(&self, i: usize) -> &[P3<S>] {
        &self.vertices[self.offsets[i]..self.offsets[i + 1]]
    }

    pub fn paths(&self) -> impl Iterator<Item = &[P3<S>]> {
        (0..self.count()).map(|i| self.path(i))
    }

    /// Map every vertex through a transform, keeping the path structure.
    pub fn mapped<T>(&self, f: &Transform<S, T>) -> PolylineSet<T> {
        PolylineSet {
            vertices: self.vertices.iter().map(|&p| f.apply(p)).collect(),
            offsets: self.offsets.clone(),
            coords: self.coords.clone(),
        }
    }

    /// The surface coordinates of path `i`, when the set has them.
    pub fn coords_of_path(&self, i: usize) -> Option<&[P2<ParamSpace>]> {
        self.coords.as_ref().map(|c| &c[self.offsets[i]..self.offsets[i + 1]])
    }

    pub fn bounds(&self) -> Option<AABB<S>> {
        AABB::of(&self.vertices)
    }

    /// Total 2D length, ignoring z -- the "length" an end-to-end
    /// equivalence check compares between two renderings.
    pub fn total_length(&self) -> f64 {
        let mut total = 0.0;
        for p in self.paths() {
            for w in p.windows(2) {
                total += Vec2::new(w[1].x() - w[0].x(), w[1].y() - w[0].y()).length();
            }
        }
        total
    }
}

/// A drawn layer: geometry plus how it is drawn.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub spec: LayerSpec,
    pub paths: PolylineSet<WorldSpace>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "core: CSR path identity cannot weld unrelated runs"
    #[test]
    fn csr_path_identity_cannot_weld_unrelated_runs() {
        let a: Vec<P3<WorldSpace>> = (0..5).map(|i| P3::new(i as f64, 0.0, 0.0)).collect();
        let b: Vec<P3<WorldSpace>> = (0..3).map(|i| P3::new(i as f64, 1.0, 0.0)).collect();
        let set = PolylineSet::from_paths(&[a, b, vec![P3::new(9.0, 9.0, 9.0)]]); // the singleton draws nothing
        assert!(
            set.count() == 2 && set.offsets == [0, 5, 8],
            "two paths, offsets [0, 5, 8], the one-vertex path dropped"
        );
        assert_eq!(set.total_length(), 6.0);
    }
}
