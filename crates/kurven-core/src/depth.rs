//! The mapping between view coordinates and depth-buffer pixels: the port
//! of `Sources/KurvenCore/Depth.swift`.
//!
//! This is `kurven.zbuffer.ZBuffer`'s mapping, verbatim, including the `0.01`
//! nudge: samples are at grid *vertices*, so the coordinate `axis0.lo` lands
//! at pixel 0 and `axis0.hi` at pixel `rows - 1`, and the nudge keeps a
//! coordinate exactly on `lo` from flooring to -1 under float jitter. Anything
//! else here is a half-pixel shift between what the renderer drew and what the
//! clipper reads, which shows up as curves leaking through the surface along
//! outlines and nowhere else.
//!
//! `axis0` is the first view component and indexes *rows*; `axis1` is the second
//! and indexes columns. That is the Python convention and the reason `ZBuffer`
//! documents that it "doesn't care about x/y semantics" -- here the phantom
//! space says which is which.

use crate::manifest::Interval;
use crate::simd::{Mat4, Vec2, Vec4};
use crate::spaces::{AABB, P2, ViewSpace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterOrder {
    /// Rows index view x and columns index view y -- `kurven.zbuffer.ZBuffer`'s
    /// convention. The bake must use this: it is the grid the Python oracle
    /// describes, and a buffer is a lookup table, never something anyone looks
    /// at, so its orientation is free to be inconvenient.
    Buffer,
    /// Rows index view y downward and columns index view x -- the picture the
    /// right way up, with view x running across the screen as it does in the
    /// SVG. What a preview wants, and the same rectangle either way.
    Screen,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthFrame {
    /// The value at row 0 and at row `rows - 1`. Directed: for `.screen` order
    /// `lo` is the *top* of the picture, so `hi < lo`. Every formula below is
    /// sign-agnostic, which is why the direction can live in the data instead of
    /// in a branch.
    pub axis0: Interval,
    /// The value at column 0 and at column `cols - 1`.
    pub axis1: Interval,
    pub rows: i64,
    pub cols: i64,
    pub nudge: f64,
    pub order: RasterOrder,
}

impl DepthFrame {
    /// A frame in buffer order with the 0.01 nudge, Swift's defaults.
    pub fn new(axis0: Interval, axis1: Interval, rows: i64, cols: i64) -> DepthFrame {
        DepthFrame { axis0, axis1, rows, cols, nudge: 0.01, order: RasterOrder::Buffer }
    }

    pub fn with_order(self, order: RasterOrder) -> DepthFrame {
        DepthFrame { order, ..self }
    }

    pub fn with_nudge(self, nudge: f64) -> DepthFrame {
        DepthFrame { nudge, ..self }
    }

    /// The frame that covers `bounds` at `resolution` squared, in buffer order.
    pub fn covering(bounds: &AABB<ViewSpace>, resolution: i64) -> DepthFrame {
        DepthFrame::new(
            Interval::new(bounds.lo.x, bounds.hi.x),
            Interval::new(bounds.lo.y, bounds.hi.y),
            resolution,
            resolution,
        )
    }

    /// The view components that drive rows and columns, in this order.
    pub fn components(&self, p: P2<ViewSpace>) -> (f64, f64) {
        match self.order {
            RasterOrder::Buffer => (p.x(), p.y()),
            RasterOrder::Screen => (p.y(), p.x()),
        }
    }

    /// The pixel `(row, column)` a view point falls in; outside the frame
    /// either can be negative or past the end.
    pub fn index(&self, p: P2<ViewSpace>) -> [i64; 2] {
        let (row, col) = self.components(p);
        [
            (self.nudge + (self.rows - 1) as f64 * (row - self.axis0.lo) / self.axis0.length()).floor() as i64,
            (self.nudge + (self.cols - 1) as f64 * (col - self.axis1.lo) / self.axis1.length()).floor() as i64,
        ]
    }

    /// Pixel size in view units along each screen axis. Equal for a frame a
    /// `Framing` built, which is what keeps preview pixels square.
    pub fn units_per_pixel(&self) -> Vec2 {
        Vec2::new(
            self.axis1.length().abs() / (self.cols - 1).max(1) as f64,
            self.axis0.length().abs() / (self.rows - 1).max(1) as f64,
        )
    }

    /// Whether this frame's grid contains a view point.
    pub fn contains(&self, p: P2<ViewSpace>) -> bool {
        let i = self.index(p);
        i[0] >= 0 && i[0] < self.rows && i[1] >= 0 && i[1] < self.cols
    }

    /// The sub-frame covering tile `(i, j)` of an `n x n` split.
    ///
    /// Split on *pixel* boundaries, not coordinate boundaries: the tile's
    /// grid is a contiguous slice of the whole drawing's grid, so a
    /// coordinate lands on the same pixel whether it was rendered whole or in
    /// pieces. Splitting the coordinate range instead would give each tile its
    /// own grid and leave gaps wherever two disagreed by a fraction of a
    /// pixel.
    pub fn tile(&self, i: i64, j: i64, n: i64) -> DepthFrame {
        if n <= 1 {
            return *self;
        }
        let (r0, r1) = (self.rows * i / n, self.rows * (i + 1) / n);
        let (c0, c1) = (self.cols * j / n, self.cols * (j + 1) / n);
        let a0 = self.axis0.length() / (self.rows - 1) as f64;
        let a1 = self.axis1.length() / (self.cols - 1) as f64;
        // Buffer order, as Swift's `tile` leaves the order at its default.
        DepthFrame::new(
            Interval::new(self.axis0.lo + a0 * r0 as f64, self.axis0.lo + a0 * (r1 - 1) as f64),
            Interval::new(self.axis1.lo + a1 * c0 as f64, self.axis1.lo + a1 * (c1 - 1) as f64),
            r1 - r0,
            c1 - c0,
        )
        .with_nudge(self.nudge)
    }

    pub fn coordinate(&self, row: i64, col: i64) -> P2<ViewSpace> {
        let a = row as f64 / (self.rows - 1) as f64 * self.axis0.length() + self.axis0.lo;
        let b = col as f64 / (self.cols - 1) as f64 * self.axis1.length() + self.axis1.lo;
        match self.order {
            RasterOrder::Buffer => P2::new(a, b),
            RasterOrder::Screen => P2::new(b, a),
        }
    }

    /// The view-to-clip matrix for this frame, under an orthographic camera.
    ///
    /// Places each pixel's *sample point* exactly on the grid vertex
    /// `coordinate(row:col:)` names. Metal samples pixel `i` at its centre,
    /// `(2i + 1) / N - 1` in NDC; solving for the coordinate that lands there
    /// gives the scale and offset below. The Python GPU path does not do this --
    /// it maps the coordinate range straight onto [-1, 1], half a pixel off the
    /// grid its own `index_to_coord` describes -- so this is half a pixel
    /// closer to the CPU oracle than Python's own GPU path is.
    ///
    /// The y row is negated because Metal puts NDC +1 at the *top* of the render
    /// target, which is row 0, while the OpenGL path the Python side uses puts
    /// it at the bottom. Which view component drives which screen axis is the
    /// frame's business too, and folded in here: the bake rasterizes in
    /// ZBuffer's order so it can be compared against Python, the preview in
    /// screen order so it looks like the drawing.
    ///
    /// A 4x4, so it can stand where a perspective matrix stands; its `w` row is
    /// constant, because an orthographic map has no divide.
    pub fn metal_clip(&self) -> Mat4 {
        let (rows, cols) = (self.rows as f64, self.cols as f64);
        let sx = 2.0 * (cols - 1.0) / (cols * self.axis1.length());
        let sy = 2.0 * (rows - 1.0) / (rows * self.axis0.length());
        let ox = 1.0 / cols - 1.0 - sx * self.axis1.lo;
        let oy = 1.0 / rows - 1.0 - sy * self.axis0.lo;
        let (a, b, c, d) = match self.order {
            RasterOrder::Buffer => (0.0, sx, -sy, 0.0), // ndc.x from view.y, ndc.y from view.x
            RasterOrder::Screen => (sx, 0.0, 0.0, -sy), // ndc.x from view.x, ndc.y from view.y
        };
        // Columns, which is simd's own layout.
        Mat4::from_columns(
            Vec4::new(a, c, 0.0, 0.0),
            Vec4::new(b, d, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(ox, -oy, 0.5, 1.0),
        )
    }
}

/// A rendered depth buffer as a value.
///
/// `empty` is the value stored where nothing was drawn -- `-.infinity` for a
/// buffer built the way Python builds one, and a large negative sentinel for one
/// that came off the GPU, because Metal cannot clear a float attachment to
/// infinity. The two are interchangeable for the visibility test (`z + depthTolerance`
/// beats either for any real depth), so carrying the sentinel rather than
/// rewriting it costs nothing and saves a pass over the whole buffer: on a
/// 16000-square bake that pass was 400 ms of the 1600 the bake took.
///
/// Coverage is the one question that does distinguish them, and it asks
/// `is_covered` rather than testing for a magic value.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthImage {
    pub frame: DepthFrame,
    pub values: Vec<f32>,
    /// The value meaning "nothing was drawn here".
    pub empty: f32,
}

impl DepthImage {
    pub fn new(frame: DepthFrame, values: Vec<f32>, empty: f32) -> DepthImage {
        assert!(
            values.len() as i64 == frame.rows * frame.cols,
            "depth image is {}x{} but has {} values",
            frame.rows,
            frame.cols,
            values.len()
        );
        DepthImage { frame, values, empty }
    }

    pub fn at(&self, row: i64, col: i64) -> f64 {
        f64::from(self.values[(row * self.frame.cols + col) as usize])
    }

    /// Whether anything was drawn at a pixel. With a MAX blend a written value
    /// is `max(empty, z)`, so "covered" is exactly "greater than empty".
    pub fn is_covered(&self, row: i64, col: i64) -> bool {
        self.values[(row * self.frame.cols + col) as usize] > self.empty
    }

    /// The stored depth under a view point, or `+.infinity` outside the frame --
    /// which reads as "occluded by the edge of the world", the same convention
    /// `clip_hidden_lines` uses for out-of-buffer lookups.
    pub fn depth(&self, p: P2<ViewSpace>) -> f64 {
        let i = self.frame.index(p);
        if i[0] < 0 || i[0] >= self.frame.rows || i[1] < 0 || i[1] >= self.frame.cols {
            return f64::INFINITY;
        }
        self.at(i[0], i[1])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// "core: the depth frame's pixel mapping is ZBuffer's"
    #[test]
    fn the_depth_frames_pixel_mapping_is_zbuffers() {
        let frame = DepthFrame::new(Interval::new(-2.0, 3.0), Interval::new(10.0, 14.0), 6, 5);
        for r in 0..frame.rows {
            for c in 0..frame.cols {
                assert_eq!(
                    frame.index(frame.coordinate(r, c)),
                    [r, c],
                    "every grid vertex round trips to its own pixel"
                );
            }
        }
        assert_eq!(
            frame.index(P2::new(-2.0, 10.0)),
            [0, 0],
            "the low corner lands on pixel 0, not -1 (the 0.01 nudge)"
        );
    }

    /// "core: the NDC transform puts each sample point on its grid vertex".
    /// Both raster orders, and the screen order's descending row axis, have
    /// to land on the same grid the clipper indexes -- a half-pixel shift
    /// here shows up as curves leaking through the surface along outlines
    /// and nowhere else.
    #[test]
    fn the_ndc_transform_puts_each_sample_point_on_its_grid_vertex() {
        for (name, frame) in [
            ("buffer order", DepthFrame::new(Interval::new(-2.0, 3.0), Interval::new(10.0, 14.0), 6, 5)),
            (
                "screen order",
                DepthFrame::new(Interval::new(3.0, -2.0), Interval::new(10.0, 14.0), 6, 5)
                    .with_order(RasterOrder::Screen),
            ),
        ] {
            let clip = frame.metal_clip();
            let mut worst = 0.0f64;
            for r in 0..frame.rows {
                for c in 0..frame.cols {
                    let p = frame.coordinate(r, c);
                    assert_eq!(frame.index(p), [r, c], "{name}: every grid vertex indexes its own pixel");
                    // Metal samples pixel (col, row) at NDC
                    // ((2c+1)/cols - 1, 1 - (2r+1)/rows): y from the top.
                    let h = clip * Vec4::new(p.x(), p.y(), 0.0, 1.0);
                    worst = worst.max((h.x / h.w - ((2 * c + 1) as f64 / frame.cols as f64 - 1.0)).abs());
                    worst = worst.max((h.y / h.w - (1.0 - (2 * r + 1) as f64 / frame.rows as f64)).abs());
                }
            }
            assert!(worst < 1e-12, "{name}: no half-pixel shift: worst |Δ| = {worst}");
        }
    }

    /// "core: bake tiles partition the drawing's grid exactly"
    #[test]
    fn bake_tiles_partition_the_drawings_grid_exactly() {
        let frame = DepthFrame::new(Interval::new(-3.0, 5.0), Interval::new(0.0, 7.0), 240, 240);
        for n in [1, 2, 3, 7] {
            let mut covered = HashSet::new();
            let mut overlapped = false;
            for i in 0..n {
                for j in 0..n {
                    let sub = frame.tile(i, j, n);
                    for r in 0..sub.rows {
                        for c in 0..sub.cols {
                            if !covered.insert(frame.index(sub.coordinate(r, c))) {
                                overlapped = true;
                            }
                        }
                    }
                }
            }
            assert!(
                !overlapped && covered.len() as i64 == frame.rows * frame.cols,
                "n = {n}: tiles cover every pixel exactly once: {} of {}",
                covered.len(),
                frame.rows * frame.cols
            );
        }
    }
}
