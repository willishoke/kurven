//! Reading a drawing back from the SVG kurven writes.
//!
//! `Sources/KurvenCore/SVG.swift` writes one shape of document and this reads
//! exactly that shape: an `<svg>` whose `viewBox` is the drawing's bounds
//! with a margin, one outer `<g>` that flips y, and inside it one
//! `<g stroke=... stroke-width=...>` per non-empty layer, in draw order,
//! holding one `<polyline points="x,y x,y ...">` per stroke. Coordinates are
//! the drawing's own page units, unflipped (the flip is the outer group's), so
//! two drawings of one scene compare coordinate for coordinate. Anything else
//! is refused rather than guessed at.

use std::path::Path;

pub type Point = [f64; 2];

/// One `<g>` of strokes: a layer as the SVG has it.
#[derive(Debug, Clone, PartialEq)]
pub struct StrokeLayer {
    pub color: String,
    /// `stroke-width`, in page units, as written.
    pub stroke_width: f64,
    pub paths: Vec<Vec<Point>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Drawing {
    /// x, y, width, height.
    pub view_box: [f64; 4],
    pub layers: Vec<StrokeLayer>,
}

/// SVG.swift's default page: sixteen inches across, at 72 points an inch.
pub const PAGE_POINTS: f64 = 16.0 * 72.0;

impl Drawing {
    /// A layer's stroke width in points, the unit a `LayerSpec.width` is in:
    /// SVG.swift writes `width * viewBox width / (16 in * 72 pt/in)`, so the
    /// written number depends on the drawing's bounds and only this compares.
    pub fn width_points(&self, layer: &StrokeLayer) -> f64 {
        layer.stroke_width * PAGE_POINTS / self.view_box[2]
    }
}

pub fn read(path: impl AsRef<Path>) -> Result<Drawing, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// The value of `name="..."` in a tag's text.
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!(" {name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')?;
    Some(&tag[start..start + end])
}

fn number(text: &str, what: &str) -> Result<f64, String> {
    text.trim().parse::<f64>().map_err(|_| format!("{what} is not a number: {text:?}"))
}

fn points(text: &str) -> Result<Vec<Point>, String> {
    text.split_ascii_whitespace()
        .map(|pair| {
            let (x, y) = pair.split_once(',').ok_or_else(|| format!("a point without a comma: {pair:?}"))?;
            Ok([number(x, "x")?, number(y, "y")?])
        })
        .collect()
}

#[derive(PartialEq)]
enum Group {
    Outer,
    Layer,
}

pub fn parse(text: &str) -> Result<Drawing, String> {
    let mut view_box = None;
    let mut layers: Vec<StrokeLayer> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut closed = false;
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        let close = rest[open..].find('>').ok_or("a tag that does not close")? + open;
        let tag = &rest[open..=close];
        rest = &rest[close + 1..];
        // The element's name, a closing tag's with its slash.
        let end = tag[2..].find(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/').map_or(tag.len(), |i| i + 2);
        let name = &tag[1..end];
        match name {
            _ if tag.starts_with("<?") => {}
            "svg" => {
                if view_box.is_some() {
                    return Err("a second <svg>".into());
                }
                let vb = attribute(tag, "viewBox").ok_or("<svg> without a viewBox")?;
                let v: Vec<f64> =
                    vb.split_ascii_whitespace().map(|t| number(t, "viewBox")).collect::<Result<_, _>>()?;
                if v.len() != 4 || v[2].is_nan() || v[2] <= 0.0 {
                    return Err(format!("viewBox {vb:?} is not x y width height"));
                }
                view_box = Some([v[0], v[1], v[2], v[3]]);
                if tag.ends_with("/>") {
                    closed = true;
                }
            }
            "g" => {
                if view_box.is_none() || closed {
                    return Err("a <g> outside the <svg>".into());
                }
                match (attribute(tag, "stroke"), groups.last()) {
                    (None, None) => groups.push(Group::Outer),
                    (Some(color), Some(Group::Outer)) => {
                        let width = number(
                            attribute(tag, "stroke-width").ok_or("a layer without a stroke-width")?,
                            "stroke-width",
                        )?;
                        layers.push(StrokeLayer { color: color.to_string(), stroke_width: width, paths: Vec::new() });
                        groups.push(Group::Layer);
                    }
                    _ => return Err("groups nested other than SVG.swift nests them".into()),
                }
            }
            "/g" => {
                groups.pop().ok_or("a </g> with no group open")?;
            }
            "polyline" => {
                if groups.last() != Some(&Group::Layer) {
                    return Err("a <polyline> outside a layer".into());
                }
                let p = points(attribute(tag, "points").ok_or("a <polyline> without points")?)?;
                if p.is_empty() {
                    return Err("a <polyline> with no points".into());
                }
                layers.last_mut().expect("a layer is open").paths.push(p);
            }
            "/svg" => closed = true,
            other => return Err(format!("<{other}> is not in a kurven drawing")),
        }
    }
    if !groups.is_empty() {
        return Err("a group that does not close".into());
    }
    if !closed {
        return Err("the <svg> does not close".into());
    }
    Ok(Drawing { view_box: view_box.ok_or("no <svg>")?, layers })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1 -2 4 3" width="1536" height="1152">
<g fill="none" stroke-linecap="round" stroke-linejoin="round" transform="translate(0,-1) scale(1,-1)">
<g stroke="#000000" stroke-width="0.002">
<polyline points="0,0 1,0 1,1"/>
<polyline points="-0.5,0.25 2.5,-1.5"/>
</g>
<g stroke="#203040" stroke-width="0.001">
<polyline points="2,2 3,3"/>
</g>
</g>
</svg>
"##;

    #[test]
    fn reads_what_svg_swift_writes() {
        let d = parse(SMALL).unwrap();
        assert_eq!(d.view_box, [-1.0, -2.0, 4.0, 3.0]);
        assert_eq!(d.layers.len(), 2);
        assert_eq!(d.layers[0].paths, vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], vec![[-0.5, 0.25], [2.5, -1.5]]]);
        assert_eq!(d.layers[1].color, "#203040");
        assert!((d.width_points(&d.layers[0]) - 0.002 * 1152.0 / 4.0).abs() < 1e-15);
    }

    #[test]
    fn an_empty_drawing() {
        let d = parse("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1 1\"/>\n").unwrap();
        assert!(d.layers.is_empty());
    }

    #[test]
    fn refuses_other_documents() {
        for bad in [
            SMALL.replace("<polyline points=\"2,2 3,3\"/>", "<path d=\"M 0 0\"/>"),
            SMALL.replace("</svg>", ""),
            SMALL.replace("2.5,-1.5", "2.5;-1.5"),
            SMALL.replace("viewBox=\"-1 -2 4 3\"", "viewBox=\"0 0 0 1\""),
            SMALL.replace("</g>\n</g>", "</g>"),
        ] {
            assert!(parse(&bad).is_err(), "read {bad}");
        }
    }

    #[test]
    fn reads_every_reference_drawing() {
        let index = kurven_core::fixtures::port_json("swift/index.json");
        let drawings = index.get("drawings").and_then(|d| d.as_array()).unwrap();
        assert!(!drawings.is_empty());
        for entry in drawings {
            let name = entry.get("name").and_then(|n| n.as_str()).unwrap();
            let d = read(kurven_core::fixtures::port(&format!("swift/{name}.svg"))).unwrap();
            let layers = entry.get("layers").and_then(|l| l.as_array()).unwrap();
            let paths = entry.get("paths").and_then(|l| l.as_array()).unwrap();
            assert_eq!(d.layers.len(), layers.len(), "{name}: the groups are the index's layers");
            for (layer, count) in d.layers.iter().zip(paths) {
                assert_eq!(Some(layer.paths.len() as i64), count.as_i64(), "{name}");
            }
        }
    }
}
