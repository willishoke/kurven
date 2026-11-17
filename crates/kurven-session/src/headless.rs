//! The app's command line with no window: the port of the headless half of
//! `Sources/KurvenApp/App.swift` (`headlessPaint`, `headlessBundle`,
//! `headlessSurface`, `headlessLandscape`, `headlessPick`, `resample`).
//!
//! The same edits, in the same order, through the same document, as the
//! Swift app runs them from its arguments: what holds the session to Swift
//! (`crates/fixtures/swift/session`). Nothing here touches a file: a run
//! reads its bundle through `open` and returns what it would write -- the
//! bundles saved, the drawings baked, the layers dumped -- with the lines it
//! printed, for the caller to write (`kurven-cli app`, or the wasm module
//! under Node).
//!
//! What needs a GPU or a window is refused: `--screenshot`, `--inspector`,
//! `--thumbnails`, a window left open. A pick reads the CPU's stand-in for
//! the preview's depth (`picking::preview_depth`).

use std::borrow::Cow;

use kurven_core::bundle::BundleFiles;
use kurven_core::manifest::ManifestError;
use kurven_core::manifest::{DomainAxis, Truncation};
use kurven_core::npy;
use kurven_core::simd::{Vec2, Vec3};
use kurven_core::surface_request::Field;
use kurven_core::svg;
use kurven_landscape::landscape::LandscapeStyle;
use kurven_math::format::description;
use kurven_surfaces::drawing::SurfacePreset;

use crate::document::{Document, State};
use crate::landscape::applying_overrides;
use crate::sliders::Slider;

/// What a run writes: a bundle's files at a directory, or a file.
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    Bundle { path: String, files: BundleFiles },
    File { path: String, bytes: Vec<u8> },
}

/// A run of the app's command line: what it printed, what it wrote, and how
/// it ended.
#[derive(Debug, Clone, Default)]
pub struct Run {
    pub stdout: Vec<String>,
    pub stderr: Vec<String>,
    pub outputs: Vec<Output>,
    pub exit: i32,
}

/// How a run ends early: Swift's `fail` and `exit`.
struct Exit(i32);

struct Headless<'a> {
    document: Document,
    run: Run,
    open: &'a Opener<'a>,
}

/// How a run reads the bundle at a path: its files, read (a directory) or
/// borrowed (files already in memory, which are then not copied).
pub type Opener<'a> = dyn Fn(&str) -> Result<Cow<'a, BundleFiles>, String> + 'a;

/// Run the app's command line headlessly (`args` without the program's
/// name). `open` reads the bundle at a path.
pub fn run<'a>(args: &[String], open: &'a Opener<'a>) -> Run {
    let mut h = Headless { document: Document::new(), run: Run::default(), open };
    let code = match h.dispatch(args) {
        Ok(()) => 0,
        Err(Exit(code)) => code,
    };
    h.run.exit = code;
    h.run
}

fn value_after<'s>(args: &'s [String], flag: &str) -> Option<&'s str> {
    let i = args.iter().position(|a| a == flag)?;
    args.get(i + 1).map(|s| s.as_str())
}

/// Swift's `Double(String)`, near enough: Rust's parser, which reads every
/// decimal and `inf`/`nan` spelling Swift's does, but not C's hexadecimal
/// floats (`crates/FINDINGS.md`).
fn double(text: &str) -> Option<f64> {
    text.trim_start_matches('+').parse::<f64>().ok().filter(|_| !text.is_empty())
}

/// Swift's `Int(String)`.
fn int(text: &str) -> Option<i64> {
    text.parse::<i64>().ok()
}

impl Headless<'_> {
    fn out(&mut self, line: String) {
        self.run.stdout.push(line);
    }

    fn fail(&mut self, why: &str) -> Exit {
        self.run.stderr.push(format!("Kurven: {why}"));
        Exit(1)
    }

    fn dispatch(&mut self, args: &[String]) -> Result<(), Exit> {
        let flagged = value_after(args, "--open").map(str::to_string);
        let bare = args.iter().find(|a| !a.starts_with('-') && a.ends_with(".kurven")).cloned();
        let bundle_path = flagged.clone().or(bare);

        if let Some(what) = value_after(args, "--landscape") {
            let what = what.to_string();
            return self.landscape(&what, args);
        }
        let painting = [
            "--add",
            "--remove",
            "--function",
            "--undo",
            "--redo",
            "--axis",
            "--light",
            "--dump",
            "--inspector",
            "--truncation",
            "--hide",
            "--drag",
            "--relief",
        ];
        let editing = ["--set", "--save", "--bake", "--levels", "--strokes"];
        let has = |set: &[&str]| args.iter().any(|a| set.contains(&a.as_str()));
        if let Some(path) = &bundle_path
            && (has(&painting) || (args.iter().any(|a| a == "--pick") && has(&editing)))
        {
            return self.paint(path, args);
        }
        let picks: Vec<&String> =
            args.iter().enumerate().filter(|(_, a)| *a == "--pick").filter_map(|(k, _)| args.get(k + 1)).collect();
        if !picks.is_empty()
            && let Some(path) = &bundle_path
        {
            let pixels: Vec<Vec<f64>> = picks.iter().map(|p| p.split(',').filter_map(double).collect()).collect();
            if !pixels.iter().all(|p| p.len() == 2) {
                self.run.stderr.push("Kurven: --pick takes a pixel, PX,PY".to_string());
                return Err(Exit(2));
            }
            self.load(path)?;
            for p in pixels {
                self.document.pick_at_pixel(Vec2::new(p[0], p[1]));
                self.document.settle();
                let status = self.document.pick_status.clone().unwrap_or_else(|| "no pick".to_string());
                self.out(format!("Kurven: {status}"));
            }
            if args.iter().any(|a| a == "--screenshot") {
                return Err(self.fail("--screenshot draws the preview, which needs the GPU (Phase 7)"));
            }
            return Ok(());
        }
        if let Some(name) = value_after(args, "--surface") {
            let name = name.to_string();
            return self.surface(&name, args);
        }
        if args.iter().any(|a| a == "--thumbnails") {
            return Err(self.fail("--thumbnails draws the picker's pictures, which needs the GPU (Phase 7)"));
        }
        if args.iter().any(|a| a == "--resample")
            && let Some(path) = &bundle_path
        {
            return self.resample(path, args);
        }
        if args.iter().any(|a| a == "--screenshot") && bundle_path.is_some() {
            return Err(self.fail("--screenshot draws the preview, which needs the GPU (Phase 7)"));
        }
        if (value_after(args, "--bake").is_some() || value_after(args, "--save").is_some())
            && let Some(path) = &bundle_path
        {
            return self.bundle(path, args);
        }
        Err(self.fail("with nothing to write, the app opens a window, which a headless run has none of"))
    }

    fn load(&mut self, path: &str) -> Result<(), Exit> {
        match (self.open)(path) {
            Ok(files) => self.document.load(path, &files),
            Err(why) => {
                let why = format!("could not open {path} — {why}");
                return Err(self.fail(&why));
            }
        }
        self.document.settle();
        if self.document.scene().is_none() {
            let why = match self.document.state() {
                State::Failed(_, message) => message.clone(),
                _ => "it did not open".to_string(),
            };
            return Err(self.fail(&format!("could not open {path} — {why}")));
        }
        Ok(())
    }

    fn summary(&self) -> String {
        let layers = self.document.layers();
        format!("{} layers, {} paths", layers.len(), layers.iter().map(|l| l.paths.count()).sum::<usize>())
    }

    /// Every `FLAG KEY=NUMBER` pair, in order.
    fn pairs(args: &[String], flag: &str) -> Vec<(String, f64)> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == flag)
            .filter_map(|(k, _)| args.get(k + 1))
            .filter_map(|p| {
                let (key, value) = p.split_once('=')?;
                Some((key.to_string(), double(value)?))
            })
            .collect()
    }

    /// Write the bundle as ⌘S does, or exit saying why not.
    fn save(&mut self, path: &str) -> Result<(), Exit> {
        let files = self.document.save_bundle(path);
        let status = if self.document.surface_bundle().is_some() {
            self.document.surface_status.clone()
        } else if self.document.polyhedron_bundle().is_some() {
            self.document.polyhedron_status.clone()
        } else {
            self.document.landscape_status.clone()
        }
        .unwrap_or_default();
        self.out(format!("Kurven: {status}"));
        if let Ok(files) = files {
            self.run.outputs.push(Output::Bundle { path: path.to_string(), files });
        }
        if !status.starts_with("saved") {
            return Err(Exit(1));
        }
        Ok(())
    }

    /// The window's bake at `resolution`, or the bundle's own.
    fn bake(&mut self, path: &str, resolution: Option<i64>) -> Result<(), Exit> {
        if let Some(resolution) = resolution {
            self.document.bake_resolution = resolution;
        }
        match self.document.bake() {
            Ok(strokes) => {
                let text = svg::render(&strokes, svg::FIGURE_WIDTH, svg::MARGIN, svg::DECIMALS);
                let name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned());
                self.out(format!("Kurven: {} paths → {}", strokes.path_count(), name.unwrap_or_default()));
                self.run.outputs.push(Output::File { path: path.to_string(), bytes: text.into_bytes() });
                Ok(())
            }
            Err(why) => {
                self.out(format!("Kurven: {why}"));
                if why.starts_with("bake failed") { Err(Exit(1)) } else { Ok(()) }
            }
        }
    }

    /// Edit the surface's request as its controls do, one field at a time,
    /// each built from the drawing before it (Swift's `headlessEdit`).
    fn edit_fields(&mut self, edits: &[(String, f64)]) -> Result<(), Exit> {
        for (name, value) in edits {
            let field =
                Field::parse(name).filter(|&f| self.document.surface_request().and_then(|s| s.get(f)).is_some());
            let Some(field) = field else {
                let fields = self
                    .document
                    .surface_request()
                    .map(|s| s.fields().iter().map(|f| f.raw()).collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                let title = self.document.title();
                return Err(self.fail(&format!("{title} has no field '{name}'; it has: {fields}")));
            };
            // A value the request refuses ends the run, saying why, before
            // anything is built from it or written.
            if !value.is_finite() {
                return Err(self.fail(&format!("--set {name} wants a finite number, not {}", description(*value))));
            }
            let next = self
                .document
                .surface_request()
                .map(|s| s.setting(field, *value, &kurven_core::surface_request::surface_key));
            match next {
                Some(Ok(next)) => self.document.surface = Some(next),
                Some(Err(error)) => return Err(self.fail(&format!("refused: {error}"))),
                None => {}
            }
            self.document.surface_edited(false);
            self.document.settle();
            let status = self.document.surface_status.clone().unwrap_or_default();
            self.out(format!("Kurven: {name} = {} — {status}", description(*value)));
        }
        Ok(())
    }

    /// Open a bundle, edit it as the window's controls do, and save or bake
    /// it (Swift's `headlessBundle`).
    fn bundle(&mut self, path: &str, args: &[String]) -> Result<(), Exit> {
        self.load(path)?;
        for (name, count) in Headless::pairs(args, "--levels") {
            let index = self.document.layers().iter().position(|l| l.spec.name == name);
            let Some(index) = index.filter(|&i| self.document.is_derived(i)) else {
                return Err(self.fail(&format!("no layer '{name}' with levels to set")));
            };
            let Some(count) = kurven_math::swift::int(count) else {
                return Err(self.fail(&format!("no layer '{name}' with levels to set")));
            };
            self.document.set_level_count(count, index, false);
            self.document.settle();
        }
        for (name, count) in Headless::pairs(args, "--strokes") {
            let index = self.document.layers().iter().position(|l| l.spec.name == name);
            let Some(index) = index.filter(|&i| self.document.hatch_count(i).is_some()) else {
                return Err(self.fail(&format!("no hatching '{name}' with strokes to set")));
            };
            let Some(count) = kurven_math::swift::int(count) else {
                return Err(self.fail(&format!("no hatching '{name}' with strokes to set")));
            };
            self.document.set_hatch_count(count, index);
            self.document.settle();
        }
        self.edit_fields(&Headless::pairs(args, "--set"))?;
        self.document.settle();
        let line = format!("Kurven: {} — {}", self.document.title(), self.summary());
        self.out(line);
        let resolution = value_after(args, "--resolution").and_then(int);
        if let Some(save) = value_after(args, "--save") {
            self.save(save)?;
        }
        if let Some(bake) = value_after(args, "--bake") {
            self.bake(bake, resolution)?;
        }
        Ok(())
    }

    /// Open a surface from the catalog and report it, then bake or save it
    /// (Swift's `headlessSurface`).
    fn surface(&mut self, name: &str, args: &[String]) -> Result<(), Exit> {
        let Some(preset) = SurfacePreset::named(name) else {
            let names = SurfacePreset::catalog().iter().map(|p| p.name.clone()).collect::<Vec<_>>().join(", ");
            return Err(self.fail(&format!("no surface '{name}'; the catalog has: {names}")));
        };
        self.document.create_surface(&preset);
        self.document.settle();
        self.edit_fields(&Headless::pairs(args, "--set"))?;
        if self.document.surface_drawing().is_none() {
            let why = self.document.surface_status.clone().unwrap_or_else(|| "no reason given".to_string());
            return Err(self.fail(&format!("could not build {name} — {why}")));
        }
        let line = format!(
            "Kurven: {} — {}, {}",
            self.document.title(),
            self.document.surface_status.clone().unwrap_or_default(),
            self.summary()
        );
        self.out(line);
        let (save, bake) = (value_after(args, "--save"), value_after(args, "--bake"));
        if args.iter().any(|a| a == "--screenshot") {
            return Err(self.fail("--screenshot draws the preview, which needs the GPU (Phase 7)"));
        }
        if save.is_none() && bake.is_none() {
            return Err(self.fail("with nothing to write, the app opens a window, which a headless run has none of"));
        }
        if let Some(save) = save {
            self.save(save)?;
        }
        if let Some(bake) = bake {
            let resolution = value_after(args, "--resolution").and_then(int);
            self.bake(bake, resolution)?;
        }
        Ok(())
    }

    /// Make a landscape and report it (Swift's `headlessLandscape`).
    fn landscape(&mut self, what: &str, args: &[String]) -> Result<(), Exit> {
        let catalog = kurven_landscape::catalog::Catalog::native();
        if let Some(preset) = catalog.preset(what) {
            self.document.create(preset);
        } else {
            // Not a name in the catalog, so it is an expression: the field
            // and the flag accept the same language.
            self.document.create(&catalog.presets[0]);
            self.document.settle();
            if let Some(landscape) = &mut self.document.landscape {
                landscape.expression = what.to_string();
                landscape.name = String::new();
            }
            self.document.landscape_edited(false);
        }
        if let Some(resolution) = value_after(args, "--resolution").and_then(int) {
            self.document.settle();
            if let Some(landscape) = &mut self.document.landscape {
                landscape.resolution = resolution;
            }
            self.document.landscape_edited(false);
        }
        self.document.settle();
        // Truncating is the other kind of edit: no request, no sampling,
        // just the curves derived again from grids that are already here.
        if let Some(height) = value_after(args, "--truncation").and_then(double) {
            self.document.set_truncation(Truncation::Uniform(height));
        }
        self.document.settle();
        if let Some(i) = args.iter().position(|a| a == "--drag")
            && let (Some(control), Some(steps)) = (args.get(i + 1), args.get(i + 2).and_then(|s| int(s)))
        {
            let control = control.clone();
            self.drag_control(&control, steps)?;
        }
        if self.document.scene().is_none() {
            let why = self.document.landscape_status.clone().unwrap_or_else(|| "no reason given".to_string());
            return Err(self.fail(&format!("could not sample {what} — {why}")));
        }
        let line = format!(
            "Kurven: {} — {}, {}",
            self.document.title(),
            self.document.landscape_status.clone().unwrap_or_default(),
            self.summary()
        );
        self.out(line);
        if let Some(save) = value_after(args, "--save") {
            let save = save.to_string();
            let _ = self.document.save_bundle(&save).map(|files| {
                self.run.outputs.push(Output::Bundle { path: save.clone(), files });
            });
        }
        if args.iter().any(|a| a == "--screenshot") {
            return Err(self.fail("--screenshot draws the preview, which needs the GPU (Phase 7)"));
        }
        Ok(())
    }

    /// Step a control as a drag does (Swift's `headlessDrag`, without its
    /// clock: the stall it measures is the main actor's, which a Worker
    /// does not have).
    fn drag_control(&mut self, control: &str, steps: i64) -> Result<(), Exit> {
        let ceiling = LandscapeStyle::ceiling(&self.document.truncation()).unwrap_or(5.0);
        let layers = self.document.layers();
        let layer = layers
            .iter()
            .position(|l| l.spec.name == "ang_minor")
            .or_else(|| (0..layers.len()).find(|&i| self.document.is_derived(i)));
        let base = layer.and_then(|l| self.document.levels_for_layer(l)).map(|l| l.len() as i64).unwrap_or(0);
        for k in 1..=steps.max(1) {
            match control {
                "truncation" => {
                    self.document.set_truncation(Truncation::Uniform(ceiling * (1.0 - 0.3 * k as f64 / steps as f64)))
                }
                "levels" => {
                    if let Some(layer) = layer {
                        self.document.set_level_count(base + k, layer, true);
                    }
                }
                _ => {
                    self.run.stderr.push(format!("Kurven: no control '{control}' to drag"));
                    return Err(Exit(2));
                }
            }
            self.document.settle();
        }
        if control == "levels" {
            self.document.rebuild_curves(false);
        }
        self.document.settle();
        self.out(format!("Kurven: dragged {control} {steps} steps"));
        Ok(())
    }

    /// Resample a landscape bundle the way an edit in the window does
    /// (Swift's `resample`).
    fn resample(&mut self, path: &str, args: &[String]) -> Result<(), Exit> {
        if let Ok(files) = (self.open)(path) {
            self.document.load(path, &files);
        }
        self.document.settle();
        if self.document.bundle().is_none() {
            let why = match self.document.state() {
                State::Failed(_, message) => message.clone(),
                _ => "it did not open".to_string(),
            };
            return Err(self.fail(&format!("cannot resample — {why}")));
        }
        let mut overrides: Vec<(String, f64)> = Vec::new();
        for (k, a) in args.iter().enumerate() {
            if a != "--resample" {
                continue;
            }
            let Some((key, value)) = args.get(k + 1).and_then(|s| s.split_once('=')) else { continue };
            let Some(number) = double(value) else {
                return Err(self.fail(&format!("{key}={value} is not a number")));
            };
            overrides.push((key.to_string(), number));
        }
        let Some(request) = self.document.landscape.clone() else {
            let example = self.document.bundle().map(|b| b.manifest.provenance.example.clone()).unwrap_or_default();
            return Err(self.fail(&format!(
                "cannot resample — this bundle's provenance is '{example}', not a landscape `kurven-cli landscape` \
                 or the window wrote, so it opens read-only"
            )));
        };
        let o: Vec<(&str, f64)> = overrides.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        self.document.landscape = Some(applying_overrides(&request, &o));
        self.document.landscape_edited(false);
        self.document.settle();
        if !self.document.is_landscape() || self.document.scene().is_none() {
            let why = self.document.landscape_status.clone().unwrap_or_else(|| "unknown".to_string());
            return Err(self.fail(&format!("resample failed — {why}")));
        }
        let line = format!("Kurven: rebuilt {path} natively — {}", self.summary());
        self.out(line);
        if args.iter().any(|a| a == "--screenshot") {
            return Err(self.fail("--screenshot draws the preview, which needs the GPU (Phase 7)"));
        }
        Ok(())
    }

    /// Open a bundle and paint it as the "Curves on" section does, a step at
    /// a time in the order given, each edit one undo step (Swift's
    /// `headlessPaint`).
    fn paint(&mut self, path: &str, args: &[String]) -> Result<(), Exit> {
        self.load(path)?;
        // `--at-once`: an edit does not wait for what it asks for before the
        // next. Here the jobs land at the next settle.
        let at_once = args.iter().any(|a| a == "--at-once");
        let resolution = value_after(args, "--resolution").and_then(int);
        let mut k = 0;
        while k < args.len() {
            let flag = args[k].clone();
            if !flag.starts_with("--") || flag == "--at-once" {
                k += 1;
                continue;
            }
            if flag == "--undo" || flag == "--redo" {
                // At once, ⌘Z pressed again before the last undo is drawn.
                if !at_once {
                    self.document.settle();
                }
                let undo = flag == "--undo";
                let can = if undo { self.document.undo() } else { self.document.redo() };
                if !at_once {
                    self.document.settle();
                }
                let word = &flag[2..];
                let line = if can {
                    format!("Kurven: {word} — {} layers", self.document.layers().len())
                } else {
                    format!("Kurven: {word}: nothing to {word}")
                };
                self.out(line);
                k += 1;
                continue;
            }
            if k + 1 >= args.len() {
                return Err(self.fail(&format!("{flag} wants a value")));
            }
            let value = args[k + 1].clone();
            k += 2;
            match flag.as_str() {
                "--pick" => {
                    self.document.settle();
                    let p: Vec<f64> = value.split(',').filter_map(double).collect();
                    if p.len() != 2 || self.document.framed_scene().is_none() || self.document.navigator.is_none() {
                        return Err(self.fail("--pick takes a pixel, PX,PY, of an open drawing"));
                    }
                    self.document.pick_at_pixel(Vec2::new(p[0], p[1]));
                    self.document.settle();
                    let status = self.document.pick_status.clone().unwrap_or_else(|| "no pick".to_string());
                    self.out(format!("Kurven: {status}"));
                }
                "--function" => {
                    // The periods "Function…" offers, unless given.
                    let mut periods = self.document.function_periods(&value);
                    if k + 1 < args.len() && args[k] == "--periods" {
                        let p: Vec<f64> = args[k + 1].split(',').filter_map(double).collect();
                        k += 2;
                        if p.len() != 2 {
                            return Err(self.fail("--periods wants the real and imaginary periods, as in 6.4,4.1"));
                        }
                        periods = (p[0], p[1]);
                    }
                    self.edit(at_once, |d| d.add_function(&value, periods).map_err(|e| e.to_string()))?;
                    self.out(format!(
                        "Kurven: function {value} over {} × {}",
                        description(periods.0),
                        description(periods.1)
                    ));
                }
                "--add" => {
                    // Chosen as soon as the menu offers it, as a hand would:
                    // once what is on screen is what was asked for.
                    let Some(pick) = self.document.picked().copied() else {
                        return Err(self.fail("nothing is picked to paint on: --pick first"));
                    };
                    self.document.settle();
                    let choices = self.document.paint_choices(&pick);
                    let Some(choice) = choices.iter().find(|c| c.id() == value).cloned() else {
                        let title = self.document.paint_title(&pick);
                        let have = choices.iter().map(|c| c.id().to_string()).collect::<Vec<_>>().join(", ");
                        return Err(
                            self.fail(&format!("nothing called '{value}' to add on {title}; the Add menu has: {have}"))
                        );
                    };
                    self.edit(at_once, |d| {
                        d.add(&choice);
                        Ok(())
                    })?;
                    self.out(format!("Kurven: added {value}"));
                }
                "--remove" => {
                    let Some(pick) = self.document.picked().copied() else {
                        return Err(self.fail("nothing is picked to paint on: --pick first"));
                    };
                    let on = self.document.painted_layers(&pick);
                    let layers = self.document.edited().map(|e| e.layers).unwrap_or_default();
                    let Some(&index) = on.iter().find(|&&i| layers.get(i).is_some_and(|l| l.name == value)) else {
                        let title = self.document.paint_title(&pick);
                        let have = on.iter().filter_map(|&i| layers.get(i).map(|l| l.name.clone())).collect::<Vec<_>>();
                        return Err(self.fail(&format!("no layer '{value}' on {title}; it has: {}", have.join(", "))));
                    };
                    self.edit(at_once, |d| d.remove_layer(index).map_err(|e| e.to_string()))?;
                    self.out(format!("Kurven: removed {value}"));
                }
                "--set" => {
                    // A number typed into a control: a surface's field, or a
                    // landscape's window or resolution, as `--resample`
                    // spells them. A number the request refuses ends the run,
                    // saying why, before anything after it is written.
                    let (key, text) = self.pair(&value)?;
                    let Some(number) = double(&text) else { return Err(self.fail("--set wants KEY=NUMBER")) };
                    if !number.is_finite() {
                        return Err(self.fail(&format!("--set {key} wants a finite number, not {text}")));
                    }
                    if let Some(surface) = self.document.surface_request().cloned() {
                        let Some(field) = Field::parse(&key).filter(|&f| surface.get(f).is_some()) else {
                            return Err(self.fail(&format!("the surface has no field '{key}'")));
                        };
                        if let Err(error) = surface.setting(field, number, &kurven_core::surface_request::surface_key) {
                            return Err(self.fail(&format!("refused: {error}")));
                        }
                        let slider = Slider::SurfaceField { field, label: key.clone() };
                        self.edit(at_once, |d| {
                            slider.set(d, number);
                            Ok(())
                        })?;
                        if self.document.paint_status.is_some() {
                            return Err(
                                self.fail(&format!("--set {key}={text} was refused, and nothing after it is done"))
                            );
                        }
                    } else if self.document.landscape_request().is_some() {
                        let slider = Slider::Landscape { label: key.clone() };
                        let key2 = key.clone();
                        self.edit(at_once, |d| {
                            slider.set_landscape(d, |r| *r = applying_overrides(r, &[(key2.as_str(), number)]));
                            Ok(())
                        })?;
                    } else {
                        return Err(
                            self.fail("--set edits a surface's or a landscape's request; this bundle has none to edit")
                        );
                    }
                    self.out(format!("Kurven: {key} = {}", description(number)));
                }
                "--relief" => {
                    // The relief slider set to A: one step of its own, through
                    // the slider's callback. Only sn in relief has one.
                    let Some(number) = double(&value) else {
                        return Err(self.fail("--relief wants an amplitude, as in 0.4"));
                    };
                    let Some(surface) = self.document.surface_request().cloned() else {
                        return Err(self.fail("--relief sets a surface's relief; this bundle has no surface request"));
                    };
                    if surface.get(Field::Relief).is_none() {
                        return Err(self.fail(&format!(
                            "--relief sets sn in relief's amplitude; this surface's shape is {}, which has none \
                             (`kurven-cli surface sn --relief A` makes one)",
                            surface.shape.kind()
                        )));
                    }
                    // Refused as the CLI refuses it, in the CLI's words.
                    let naming = |key: &str| {
                        if key == "shape.amplitude" { "--relief".to_string() } else { format!("surface.{key}") }
                    };
                    if let Err(error) = surface.setting(Field::Relief, number, &naming) {
                        let why = match error {
                            ManifestError::Invalid(why) => why,
                            other => other.to_string(),
                        };
                        return Err(self.fail(&why));
                    }
                    let slider = Slider::SurfaceField { field: Field::Relief, label: "Relief".to_string() };
                    self.edit(at_once, |d| {
                        slider.set(d, number);
                        Ok(())
                    })?;
                    if self.document.paint_status.is_some() {
                        return Err(self.fail(&format!("--relief {value} was refused, and nothing after it is done")));
                    }
                    self.out(format!("Kurven: relief = {}", description(number)));
                }
                "--drag" => {
                    // A drag of a surface's or a landscape's control through
                    // its slider's callbacks: KEY=V1,V2,... taken hold of,
                    // moved through the values, let go.
                    let (key, text) = self.pair(&value)?;
                    let values: Vec<f64> = text.split(',').filter_map(double).collect();
                    if values.is_empty() {
                        return Err(self.fail("--drag wants KEY=V1,V2,..."));
                    }
                    if !values.iter().all(|v| v.is_finite()) {
                        return Err(self.fail(&format!("--drag {key} wants finite numbers, not {text}")));
                    }
                    self.document.paint_status = None;
                    if let Some(surface) = self.document.surface_request().cloned() {
                        let Some(field) = Field::parse(&key).filter(|&f| surface.get(f).is_some()) else {
                            return Err(self.fail(&format!("the surface has no field '{key}'")));
                        };
                        let slider = Slider::SurfaceField { field, label: key.clone() };
                        slider.editing_changed(&mut self.document, true);
                        for v in &values {
                            slider.set(&mut self.document, *v);
                            if !at_once {
                                self.document.settle();
                            }
                        }
                        slider.editing_changed(&mut self.document, false);
                    } else if self.document.landscape_request().is_some() {
                        let slider = Slider::Landscape { label: key.clone() };
                        slider.editing_changed(&mut self.document, true);
                        for v in &values {
                            let key = key.as_str();
                            slider.set_landscape(&mut self.document, |r| *r = applying_overrides(r, &[(key, *v)]));
                        }
                        slider.editing_changed(&mut self.document, false);
                    } else {
                        return Err(
                            self.fail("--drag moves a surface's or a landscape's control; this bundle has none")
                        );
                    }
                    if !at_once {
                        self.document.settle();
                    }
                    if let Some(status) = self.document.paint_status.clone() {
                        self.out(format!("Kurven: refused — {status}"));
                    }
                    self.out(format!("Kurven: dragged {key} through {} values", values.len()));
                }
                "--draft-resolution" => {
                    // The resolution a landscape's drag samples at while it lasts.
                    match int(&value).filter(|&n| n >= 8) {
                        Some(n) => self.document.draft_resolution = n,
                        None => return Err(self.fail("--draft-resolution wants a resolution")),
                    }
                }
                "--truncation" => {
                    let Some(height) = double(&value) else { return Err(self.fail("--truncation wants a height")) };
                    self.edit(at_once, |d| {
                        d.edit_truncation(Truncation::Uniform(height), "Truncation");
                        Ok(())
                    })?;
                    self.out(format!("Kurven: truncated at {}", description(height)));
                }
                "--hide" => {
                    let Some(index) = self.document.layers().iter().position(|l| l.spec.name == value) else {
                        return Err(self.fail(&format!("no layer '{value}' to hide")));
                    };
                    self.document.toggle(index);
                }
                "--levels" | "--strokes" => {
                    // A drag through the slider's own callbacks: taken hold
                    // of, moved, let go, each an event.
                    let (name, text) = self.pair(&value)?;
                    let Some(count) = int(&text) else {
                        return Err(self.fail(&format!("{flag} wants LAYER=COUNT")));
                    };
                    let levels = flag == "--levels";
                    let from = if levels {
                        self.document
                            .layer_specs()
                            .iter()
                            .position(|s| s.name == name)
                            .and_then(|i| self.document.levels_for_layer(i))
                            .map(|l| l.len() as i64)
                    } else {
                        self.document
                            .edited()
                            .and_then(|e| e.layers.iter().position(|l| l.name == name))
                            .and_then(|i| self.document.hatch_count(i))
                    };
                    let Some(from) = from else {
                        let what = if levels { "levels" } else { "strokes" };
                        return Err(self.fail(&format!("no layer '{name}' with {what} to drag")));
                    };
                    let slider = if levels {
                        Slider::Levels { name: name.clone() }
                    } else {
                        Slider::Strokes { name: name.clone() }
                    };
                    // Every count on the way for a short drag; sixteen even
                    // steps for a long one.
                    let span = count - from;
                    let steps = span.abs().clamp(1, 16);
                    slider.editing_changed(&mut self.document, true);
                    for i in 1..=steps {
                        slider.set(&mut self.document, (from + span * i / steps) as f64);
                        if !at_once {
                            self.document.settle();
                        }
                    }
                    slider.editing_changed(&mut self.document, false);
                    if !at_once {
                        self.document.settle();
                    }
                    let tail = self.document.paint_status.clone().map(|s| format!(" — {s}")).unwrap_or_default();
                    self.out(format!("Kurven: dragged the {} of {name} from {from} to {count}{tail}", &flag[2..]));
                }
                "--axis" => {
                    let axis = match value.as_str() {
                        "real" => DomainAxis::Real,
                        "imag" => DomainAxis::Imag,
                        _ => return Err(self.fail("--axis wants real or imag")),
                    };
                    self.edit(at_once, |d| {
                        d.paint_axis(axis);
                        Ok(())
                    })?;
                }
                "--light" => {
                    // The light a layer is read under, LAYER=X,Y,Z: one undo step.
                    let (name, text) = self.pair(&value)?;
                    let l: Vec<f64> = text.split(',').filter_map(double).collect();
                    if l.len() != 3 || !l.iter().all(|x| x.is_finite()) || l == [0.0, 0.0, 0.0] {
                        return Err(self.fail("--light wants LAYER=X,Y,Z, a direction toward the light"));
                    }
                    let lit = self
                        .document
                        .edited()
                        .and_then(|e| e.layers.iter().find(|l| l.name == name).map(|l| l.source.light().is_some()));
                    if lit != Some(true) {
                        return Err(self.fail(&format!(
                            "no layer '{name}' read under a light: an isophote or a half-Lambert tone"
                        )));
                    }
                    let light = Vec3::new(l[0], l[1], l[2]);
                    self.edit(at_once, |d| {
                        d.paint_light(light, &name);
                        Ok(())
                    })?;
                    self.out(format!(
                        "Kurven: light of {name} toward {},{},{}",
                        description(l[0]),
                        description(l[1]),
                        description(l[2])
                    ));
                }
                "--save" => {
                    self.document.settle();
                    self.save(&value)?;
                }
                "--bake" => {
                    self.document.settle();
                    self.bake(&value, resolution)?;
                }
                "--screenshot" | "--inspector" => {
                    return Err(self.fail(&format!("{flag} draws the window, which needs the GPU (Phase 7)")));
                }
                "--dump" => {
                    self.document.settle();
                    let (name, prefix) = self.pair(&value)?;
                    let Some(layer) = self.document.layers().iter().find(|l| l.spec.name == name).cloned() else {
                        return Err(self.fail(&format!("no layer '{name}' to dump")));
                    };
                    let paths = &layer.paths;
                    let vertices: Vec<f64> = paths.vertices.iter().flat_map(|v| [v.x(), v.y(), v.z()]).collect();
                    let offsets: Vec<i64> = paths.offsets.iter().map(|&o| o as i64).collect();
                    self.run.outputs.push(Output::File {
                        path: format!("{prefix}.npy"),
                        bytes: npy::encode_f64(&vertices, &[paths.vertices.len(), 3]),
                    });
                    self.run.outputs.push(Output::File {
                        path: format!("{prefix}.idx.npy"),
                        bytes: npy::encode_i64(&offsets, &[offsets.len()]),
                    });
                    self.out(format!("Kurven: dumped {name}, {} paths", paths.count()));
                }
                "--resolution" => {}
                _ => return Err(self.fail(&format!("{flag} is not an editing step"))),
            }
        }
        Ok(())
    }

    fn pair(&mut self, text: &str) -> Result<(String, String), Exit> {
        match text.split_once('=') {
            Some((a, b)) => Ok((a.to_string(), b.to_string())),
            None => Err(self.fail(&format!("'{text}' wants NAME=VALUE"))),
        }
    }

    /// One edit, and unless `--at-once`, what it asked for drawn. A thrown
    /// refusal ends the run (`fail("refused: ...")`); a refusal the
    /// document records is printed and the run goes on.
    fn edit(&mut self, at_once: bool, body: impl FnOnce(&mut Document) -> Result<(), String>) -> Result<(), Exit> {
        self.document.paint_status = None;
        if let Err(error) = body(&mut self.document) {
            return Err(self.fail(&format!("refused: {error}")));
        }
        if let Some(status) = self.document.paint_status.clone() {
            self.out(format!("Kurven: refused — {status}"));
        }
        if !at_once {
            self.document.settle();
        }
        Ok(())
    }
}
