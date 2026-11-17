//! The controls' callbacks, as the window's sliders call them and a headless
//! drag does: `LandscapeSlider` (`LandscapeSection.swift`), `LevelSlider`,
//! `StrokeSlider` and `LightSlider` (`PaintSection.swift`), and
//! `SurfaceFieldSlider` (`SurfaceSection.swift`). Each is a drag taken hold
//! of, moved and let go -- one undo step -- or a value typed, an edit of its
//! own.

use kurven_core::manifest::Truncation;
use kurven_core::simd::Vec3;
use kurven_core::surface_request::{Field, SurfaceRequest, surface_key};
use kurven_landscape::catalog::LandscapeRequest;
use kurven_math::real;

use crate::document::Document;

/// A slider of the document's: which control, by what it edits.
#[derive(Debug, Clone, PartialEq)]
pub enum Slider {
    /// The window's and the resolution's sliders. A drag samples at the
    /// draft resolution while it lasts and at the real one when it ends.
    Landscape { label: String },
    /// A layer's level slider. While a drag lasts the curves are the grid's,
    /// and placed on f when it ends.
    Levels { name: String },
    /// A hatching's strokes slider.
    Strokes { name: String },
    /// A layer's light sliders: the direction toward the light by its
    /// azimuth round z and its elevation over the xy plane, in degrees.
    Light { name: String },
    /// One number of a surface's request.
    SurfaceField { field: Field, label: String },
    /// A uniform truncation's height (`LandscapeSection`'s
    /// `truncationSlider`): a drag is one undo step, each move a truncation.
    Truncation,
}

impl Slider {
    /// Taken hold of (`true`) or let go (`false`).
    pub fn editing_changed(&self, document: &mut Document, editing: bool) {
        match self {
            Slider::Landscape { label } => {
                if editing {
                    document.begin_drag();
                    document.drag_landscape(true);
                } else {
                    // Let go first, so a drag back to where it began records
                    // the bundle it began on (`end_drag`); then sampled in full.
                    document.end_drag(label);
                    document.landscape_edited(false);
                }
            }
            Slider::Levels { name } => {
                if editing {
                    document.begin_drag();
                } else {
                    document.rebuild_curves(false);
                    document.end_drag(&format!("Levels of {name}"));
                }
            }
            Slider::Strokes { name } => {
                if editing {
                    document.begin_drag();
                } else {
                    document.end_drag(&format!("Strokes of {name}"));
                }
            }
            Slider::Light { name } => {
                if editing {
                    document.begin_drag();
                } else {
                    document.end_drag(&format!("Light of {name}"));
                }
            }
            Slider::Truncation => {
                if editing {
                    document.begin_drag();
                } else {
                    document.end_drag("Truncation");
                }
            }
            Slider::SurfaceField { label, .. } => {
                if editing {
                    document.begin_drag();
                } else {
                    // Let go first, so a drag back to where it began records
                    // the bundle it began on; then the full request.
                    document.end_drag(label);
                    document.surface_edited(false);
                }
            }
        }
    }

    /// A value: a count, an amplitude, a number of a request.
    pub fn set(&self, document: &mut Document, value: f64) {
        match self {
            Slider::Levels { name } => {
                let draft = document.edited_before_drag.is_some();
                if let Some(count) = kurven_math::swift::int(value.round()) {
                    document.paint_levels(count, name, draft);
                }
            }
            Slider::Strokes { name } => {
                if let Some(count) = kurven_math::swift::int(value.round()) {
                    document.paint_strokes(count, name);
                }
            }
            Slider::SurfaceField { field, label } => set_surface_field(document, *field, label, value),
            Slider::Truncation => document.edit_truncation(Truncation::Uniform(value), "Truncation"),
            Slider::Landscape { .. } | Slider::Light { .. } => {}
        }
    }

    /// The landscape slider's value: a change to the request.
    pub fn set_landscape(&self, document: &mut Document, change: impl FnOnce(&mut LandscapeRequest)) {
        let Slider::Landscape { label } = self else { return };
        if document.edited_before_drag.is_some() {
            let Some(mut request) = document.landscape.clone() else { return };
            change(&mut request);
            document.landscape = Some(request);
        } else {
            document.edit_landscape(label, change);
        }
    }

    /// The light sliders' value, in degrees.
    pub fn set_light(&self, document: &mut Document, azimuth: f64, elevation: f64) {
        let Slider::Light { name } = self else { return };
        let pi = core::f64::consts::PI;
        let (a, e) = (azimuth * pi / 180.0, elevation * pi / 180.0);
        document.paint_light(Vec3::new(real::cos(e) * real::cos(a), real::cos(e) * real::sin(a), real::sin(e)), name);
    }

    /// Whether a surface field's drag builds drafts. A linear curve, and the
    /// trajectory, are curves alone, placed from the grid, so their drags
    /// build the full drawing each step.
    pub fn drafts(field: Field) -> bool {
        !SurfaceRequest::CURVE_FIELDS.contains(&field)
    }
}

/// A value the request refuses -- not a number, or out of its range -- is
/// not taken, a step of a drag included, and the status says why: the
/// drawing and what a save writes stay at the last value taken.
fn set_surface_field(document: &mut Document, field: Field, label: &str, value: f64) {
    let Some(current) = document.surface.clone() else { return };
    if current.get(field) == Some(value) {
        return;
    }
    let next = match current.setting(field, value, &surface_key) {
        Ok(next) => next,
        Err(error) => {
            document.paint_status = Some(error.to_string());
            return;
        }
    };
    // Nor a value the layers drawn on the surface cannot take: a relief past
    // the tube's reach bounds no solid, and isophotes and half-Lambert tone
    // need its outward side.
    if let Some(mut edited) = document.edited() {
        edited.request = Some(next.clone());
        if let Some(why) = document.refusal(&edited) {
            document.paint_status = Some(why);
            return;
        }
    }
    if document.edited_before_drag.is_some() {
        document.drag_surface(next, Slider::drafts(field));
    } else {
        document.edit_surface(next, label);
    }
}
