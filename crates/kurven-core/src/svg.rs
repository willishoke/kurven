//! Stroke output: the port of `Sources/KurvenCore/SVG.swift`.

use crate::manifest::LayerSpec;
use crate::polyline::PolylineSet;
use crate::spaces::{AABB, PageSpace};

#[derive(Debug, Clone, PartialEq)]
pub struct Style {
    pub color: String,
    /// Line width in points, as matplotlib means it.
    pub width: f64,
}

impl Style {
    pub fn of(spec: &LayerSpec) -> Style {
        Style { color: spec.color.clone(), width: spec.width }
    }
}

/// The finished drawing: strokes in page coordinates, in draw order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Strokes {
    pub layers: Vec<(Style, PolylineSet<PageSpace>)>,
}

impl Strokes {
    pub fn bounds(&self) -> Option<AABB<PageSpace>> {
        self.layers.iter().filter_map(|(_, p)| p.bounds()).reduce(|a, b| a.union(&b))
    }

    pub fn path_count(&self) -> usize {
        self.layers.iter().map(|(_, p)| p.count()).sum()
    }

    pub fn total_length(&self) -> f64 {
        self.layers.iter().map(|(_, p)| p.total_length()).sum()
    }
}

/// Points per inch, and the CSS pixels per inch an SVG viewer assumes.
const POINTS_PER_INCH: f64 = 72.0;
const PIXELS_PER_INCH: f64 = 96.0;

/// Swift's defaults for `render`: a sixteen-inch page, a margin of 2% of the
/// drawing's longer side, and coordinates to four decimals.
pub const FIGURE_WIDTH: f64 = 16.0;
pub const MARGIN: f64 = 0.02;
pub const DECIMALS: usize = 4;

/// `Strokes` as an SVG document.
///
/// Deliberately not matplotlib's SVG: no figure, no axes, no clip groups, no
/// transform stack -- one `<g>` per layer and one `<polyline>` per path, in a
/// viewBox that is the drawing's own bounds. Equivalence with the Python plate
/// is asserted on strokes, not on pixels, so there is nothing to gain from
/// reproducing matplotlib's chrome and a great deal to lose in trying.
///
/// Render at a given physical width, in inches.
///
/// The width matters because a `LayerSpec.width` is a matplotlib line width
/// in *points*, and points are only a size once you know how big the drawing
/// is. matplotlib resolves this with the figure size; the examples use 16
/// inches, so that is the default here and a stroke comes out the same
/// physical thickness it does on the drawing. Changing `figure_width` scales
/// the paper, not the strokes -- exactly as re-saving a matplotlib figure at a
/// different `figsize` would.
pub fn render(strokes: &Strokes, figure_width: f64, margin: f64, decimals: usize) -> String {
    let Some(b) = strokes.bounds() else {
        return "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\"/>\n".to_string();
    };
    let size = b.size();
    let pad = size.x.max(size.y) * margin;
    let (x0, y0) = (b.lo.x - pad, b.lo.y - pad);
    let (w, h) = (size.x + 2.0 * pad, size.y + 2.0 * pad);
    // Drawing units per point at this figure width.
    let units_per_point = w / (figure_width * POINTS_PER_INCH);

    let mut out = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{} {} {} {}\" \
         width=\"{}\" height=\"{}\">\n<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\" \
         transform=\"translate(0,{}) scale(1,-1)\">\n",
        fmt(x0, decimals),
        fmt(y0, decimals),
        fmt(w, decimals),
        fmt(h, decimals),
        fmt(figure_width * PIXELS_PER_INCH, 3),
        fmt(figure_width * PIXELS_PER_INCH * h / w, 3),
        fmt(2.0 * y0 + h, decimals),
    );
    // The y flip: SVG's y grows downward, the drawing's grows upward. Doing it
    // once on the outer group means every coordinate below is a drawing
    // coordinate, unnegated, so a stroke list can be read against the
    // Python arrays directly.
    for (style, paths) in strokes.layers.iter().filter(|(_, p)| !p.is_empty()) {
        out.push_str(&format!(
            "<g stroke=\"{}\" stroke-width=\"{}\">\n",
            style.color,
            fmt(style.width * units_per_point, 6)
        ));
        for path in paths.paths() {
            out.push_str("<polyline points=\"");
            let points: Vec<String> =
                path.iter().map(|p| format!("{},{}", fmt(p.x(), decimals), fmt(p.y(), decimals))).collect();
            out.push_str(&points.join(" "));
            out.push_str("\"/>\n");
        }
        out.push_str("</g>\n");
    }
    out.push_str("</g>\n</svg>\n");
    out
}

/// `String(format: "%.Nf")` with its trailing zeros and point dropped, and
/// `-0` written `0`. Rust's fixed formatting rounds the exact binary value
/// half to even, as Darwin's printf does.
pub fn fmt(v: f64, decimals: usize) -> String {
    let mut s = format!("{v:.decimals$}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    if s == "-0" { "0".to_string() } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::P3;

    #[test]
    fn numbers_as_printf_writes_them() {
        assert_eq!(fmt(1.0, 4), "1");
        assert_eq!(fmt(-0.00001, 4), "0");
        assert_eq!(fmt(0.125, 2), "0.12", "half to even on the exact value");
        assert_eq!(fmt(2.5, 4), "2.5");
        assert_eq!(fmt(1536.0, 3), "1536");
    }

    #[test]
    fn an_empty_drawing_and_a_small_one() {
        assert_eq!(
            render(&Strokes::default(), FIGURE_WIDTH, MARGIN, DECIMALS),
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\"/>\n"
        );
        let paths = PolylineSet::<PageSpace>::from_paths(&[vec![P3::new(0.0, 0.0, 0.0), P3::new(1.0, 2.0, 0.0)]]);
        let strokes = Strokes {
            layers: vec![
                (Style { color: "#000000".to_string(), width: 0.5 }, paths),
                (Style { color: "#ff0000".to_string(), width: 1.0 }, PolylineSet::empty()),
            ],
        };
        let svg = render(&strokes, FIGURE_WIDTH, MARGIN, DECIMALS);
        assert_eq!(
            svg,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"-0.04 -0.04 \
             1.08 2.08\" width=\"1536\" height=\"2958.222\">\n<g fill=\"none\" stroke-linecap=\"round\" \
             stroke-linejoin=\"round\" transform=\"translate(0,2) scale(1,-1)\">\n<g stroke=\"#000000\" \
             stroke-width=\"0.000469\">\n<polyline points=\"0,0 1,2\"/>\n</g>\n</g>\n</svg>\n"
        );
    }
}
