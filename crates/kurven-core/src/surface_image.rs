//! The coordinate image: the depth pass with a parametric surface's
//! coordinate beside each pixel's depth. The type of
//! `Sources/KurvenCore/SurfaceImage.swift`, which the scene's visibility
//! takes; the CPU rasterizer is Phase 4's.

use crate::depth::{DepthFrame, DepthImage};
use crate::parametric::ParametricSurface;
use crate::parametric::PortRefusal;
use crate::spaces::{P2, ParamSpace, Transform, ViewSpace, WorldSpace};

/// What the coordinate pass draws: the depth image, and the surface
/// coordinates of the front-most point at each pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceImage {
    pub depth: DepthImage,
    /// The surface coordinates of the front-most point at each pixel, row
    /// major like `depth.values`; NaN where nothing was drawn. Unwrapped as
    /// rasterized, so compare them with `ParamAxis::difference`.
    pub coords: Vec<[f32; 2]>,
}

/// A coordinate image whose coordinates do not match its pixels: Swift's
/// precondition, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoordinateCount {
    pub coordinates: usize,
    pub pixels: usize,
}

impl SurfaceImage {
    /// One coordinate per pixel, or the counts that differ.
    pub fn new(depth: DepthImage, coords: Vec<[f32; 2]>) -> Result<SurfaceImage, CoordinateCount> {
        if coords.len() != depth.values.len() {
            return Err(CoordinateCount { coordinates: coords.len(), pixels: depth.values.len() });
        }
        Ok(SurfaceImage { depth, coords })
    }

    pub fn frame(&self) -> DepthFrame {
        self.depth.frame
    }

    /// The front-most surface coordinate at a pixel, or None where nothing was
    /// drawn.
    pub fn coordinate(&self, row: i64, col: i64) -> Option<P2<ParamSpace>> {
        let c = self.coords[(row * self.frame().cols + col) as usize];
        if c[0].is_nan() { None } else { Some(P2::new(f64::from(c[0]), f64::from(c[1]))) }
    }

    /// Rasterize a parametric surface's grid on the CPU.
    pub fn rasterize(
        _surface: &ParametricSurface,
        _view: &Transform<WorldSpace, ViewSpace>,
        _frame: DepthFrame,
    ) -> Result<SurfaceImage, PortRefusal> {
        // PORT: phase 4 -- two triangles per grid cell, sampled at the frame's grid points, keeping the greatest view z and carrying the interpolated, unwrapped surface coordinate of whichever triangle won.
        Err(PortRefusal { phase: 4, what: "SurfaceImage.rasterize" })
    }
}
