//! The command line as `Sources/kurven-cli/main.swift` reads it: `Args` and
//! `CLIError`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use kurven_core::manifest::CameraPreset;

/// A refusal to report, Swift's `CLIError`: its text is the whole message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(pub String);

impl CliError {
    pub fn new(description: impl Into<String>) -> CliError {
        CliError(description.into())
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

/// Every error of the engine a command reaches is reported by its text, as
/// Swift's `catch` prints `\(error)`.
macro_rules! from_error {
    ($($t:ty),*) => {$(
        impl From<$t> for CliError {
            fn from(e: $t) -> Self {
                CliError(e.to_string())
            }
        }
    )*};
}
from_error!(
    kurven_core::bundle::BundleError,
    kurven_core::manifest::ManifestError,
    kurven_core::manifest::SchemaError,
    kurven_core::json::JsonError,
    kurven_core::parametric::PortRefusal,
    kurven_landscape::landscape::BuildError,
    kurven_bake::BakeError
);

/// The words of a command line: positionals, flags with a value, switches
/// without one, and every value of an option given more than once.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    pub positional: Vec<String>,
    pub flags: BTreeMap<String, String>,
    pub switches: BTreeSet<String>,
    /// Options that may be given more than once, keeping every value.
    pub repeated: BTreeMap<String, Vec<String>>,
}

/// Swift's `Double(text)`: a decimal, `inf`, `infinity` or `nan`, signed or
/// not. (Swift's also reads C's hexadecimal floats, `0x1p3`, which no flag
/// of the port's is given.)
pub fn double(text: &str) -> Option<f64> {
    text.parse::<f64>().ok()
}

/// Swift's `Int(text)`: an optional sign and decimal digits, in range.
pub fn int(text: &str) -> Option<i64> {
    text.parse::<i64>().ok()
}

impl Args {
    pub fn new<S: AsRef<str>>(argv: &[S]) -> Args {
        let argv: Vec<&str> = argv.iter().map(AsRef::as_ref).collect();
        let mut args = Args::default();
        let mut i = 0;
        while i < argv.len() {
            let a = argv[i];
            if let Some(name) = a.strip_prefix("--") {
                if i + 1 < argv.len() && Args::is_value(argv[i + 1]) {
                    args.flags.insert(name.to_string(), argv[i + 1].to_string());
                    args.repeated.entry(name.to_string()).or_default().push(argv[i + 1].to_string());
                    i += 2;
                } else {
                    args.switches.insert(name.to_string());
                    i += 1;
                }
            } else if a == "-o" && i + 1 < argv.len() {
                args.flags.insert("output".to_string(), argv[i + 1].to_string());
                i += 2;
            } else {
                args.positional.push(a.to_string());
                i += 1;
            }
        }
        args
    }

    /// Whether a token is a value rather than the next option.
    ///
    /// Anything beginning with `-` is an option, *except* a negative number:
    /// `--shear -0.5` means a shear and `--soft -o out.svg` does not mean
    /// a soft called "-o". A negative *interval* too, since `--re -6,8` is
    /// how a landscape's window is written and its left end is usually
    /// negative, and a list of number pairs, `--bands -3.5:2.5,0.5:5.5`, for
    /// the same reason. (Swift's `split(separator: ",")` drops empty pieces;
    /// its split on ":" keeps them.)
    pub fn is_value(token: &str) -> bool {
        if !token.starts_with('-') {
            return true;
        }
        token.split(',').filter(|s| !s.is_empty()).all(|part| part.split(':').all(|n| double(n).is_some()))
    }

    /// Whether the switch `name` is given. A switch given a value is
    /// refused: `--conformal 1` is the flag `conformal` with the value 1,
    /// which a switch read from `switches` alone took as off.
    pub fn switched_on(&self, name: &str) -> Result<bool, CliError> {
        debug_assert!(Args::SWITCHES.contains(&name), "--{name} is read as a switch but not listed in Args::SWITCHES");
        if let Some(value) = self.flags.get(name) {
            return Err(CliError(format!("--{name} is a switch and takes no value, not '{value}'")));
        }
        Ok(self.switches.contains(name))
    }

    pub fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    pub fn string(&self, name: &str) -> Result<&str, CliError> {
        self.flag(name).ok_or_else(|| CliError(format!("missing --{name}")))
    }

    pub fn int_or(&self, name: &str, fallback: i64) -> Result<i64, CliError> {
        let Some(v) = self.flag(name) else { return Ok(fallback) };
        int(v).ok_or_else(|| CliError(format!("--{name} wants an integer, got '{v}'")))
    }

    pub fn int(&self, name: &str) -> Result<Option<i64>, CliError> {
        if self.flags.contains_key(name) { self.int_or(name, 0).map(Some) } else { Ok(None) }
    }

    pub fn double(&self, name: &str) -> Result<Option<f64>, CliError> {
        let Some(v) = self.flag(name) else { return Ok(None) };
        double(v).map(Some).ok_or_else(|| CliError(format!("--{name} wants a number, got '{v}'")))
    }

    /// A depth buffer's resolution, refused by the rule a preset's buffer is
    /// read by (`CameraPreset::unsupported_buffer`).
    pub fn resolution(&self, fallback: i64) -> Result<i64, CliError> {
        let n = self.int_or("resolution", fallback)?;
        if let Some(why) = CameraPreset::unsupported_buffer(n) {
            return Err(CliError(format!("--resolution {why}")));
        }
        Ok(n)
    }

    /// A number that must be positive and finite, as a length is.
    pub fn positive(&self, name: &str) -> Result<Option<f64>, CliError> {
        let Some(d) = self.double(name)? else { return Ok(None) };
        if !(d > 0.0 && d.is_finite()) {
            return Err(CliError(format!("--{name} wants a positive number, got '{}'", self.flags[name])));
        }
        Ok(Some(d))
    }

    /// Whether an option is given at all, as a flag or a switch.
    pub fn given(&self, name: &str) -> bool {
        self.flags.contains_key(name) || self.switches.contains(name)
    }

    /// Refuses every option `command` does not take. Without this a flag
    /// that was renamed or misspelt was read by nothing, and the command ran
    /// on its default.
    pub fn only(&self, command: &str, accepted: &[&str]) -> Result<(), CliError> {
        let unknown: BTreeSet<&str> = self
            .flags
            .keys()
            .map(String::as_str)
            .chain(self.switches.iter().map(String::as_str))
            .filter(|name| !accepted.contains(name))
            .collect();
        if unknown.is_empty() {
            // The parser cannot tell a switch from an option whose value was
            // left off, so `--preset` given bare lands in `switches`, where
            // no reader of a value looks, and the command ran on its default.
            // Swift's `only` accepts it; the port refuses it, as
            // `switched_on` refuses a switch given a value.
            if let Some(bare) = self.switches.iter().find(|name| !Args::SWITCHES.contains(&name.as_str())) {
                return Err(CliError(format!("--{bare} wants a value")));
            }
            return Ok(());
        }
        let spelled = |name: &str| if name == "output" { "-o".to_string() } else { format!("--{name}") };
        let listed: Vec<String> = unknown.iter().map(|n| spelled(n)).collect();
        let takes = if accepted.is_empty() {
            "; it takes no options".to_string()
        } else {
            format!("; it takes {}", accepted.iter().map(|n| spelled(n)).collect::<Vec<_>>().join(" "))
        };
        Err(CliError(format!("{command} takes no {}{takes}", listed.join(", "))))
    }

    /// Every switch a command reads (`switched_on`): the options that take
    /// no value. Any other option given without one is refused by `only`.
    pub const SWITCHES: [&'static str; 8] =
        ["conformal", "draft", "editing", "no-refine", "no-truncation", "preview", "soft", "visibility"];

    /// The options `load_scene` reads.
    pub const SCENE: [&'static str; 2] = ["preset", "no-refine"];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_numbers_intervals_and_bands_are_values() {
        let args = Args::new(&["landscape", "--re", "-6,8", "--bands", "-3.5:2.5,0.5:5.5", "--shear", "-0.5"]);
        assert_eq!(args.flag("re"), Some("-6,8"));
        assert_eq!(args.flag("bands"), Some("-3.5:2.5,0.5:5.5"));
        assert_eq!(args.flag("shear"), Some("-0.5"));
        let args = Args::new(&["bake", "b.kurven", "--no-refine", "-o", "out.svg"]);
        assert!(args.switches.contains("no-refine"));
        assert_eq!(args.flag("output"), Some("out.svg"));
        assert_eq!(args.positional, ["bake", "b.kurven"]);
        assert!(!Args::is_value("-o") && !Args::is_value("--x") && !Args::is_value("-1:"));
        assert!(Args::is_value("-1,,2") && Args::is_value("x"));
    }

    #[test]
    fn switches_and_unknown_options_are_refused_as_swift_refuses_them() {
        let args = Args::new(&["bake", "--no-refine", "2", "--prest", "plate"]);
        assert_eq!(args.switched_on("no-refine").unwrap_err().0, "--no-refine is a switch and takes no value, not '2'");
        assert_eq!(
            args.only("bake", &["preset", "no-refine", "output"]).unwrap_err().0,
            "bake takes no --prest; it takes --preset --no-refine -o"
        );
        assert_eq!(
            Args::new(&["catalog", "-o", "x"]).only("catalog", &[]).unwrap_err().0,
            "catalog takes no -o; it takes no options"
        );
    }

    #[test]
    fn a_value_option_given_bare_is_refused() {
        for (words, bare) in [
            (vec!["bake", "b.kurven", "--preset"], "preset"),
            (vec!["bake", "b.kurven", "--preset", "--no-refine"], "preset"),
            (vec!["landscape", "z", "--truncation", "-o", "x.kurven"], "truncation"),
            (vec!["landscape", "z", "--slope-lines", "-o", "x.kurven"], "slope-lines"),
            (vec!["references", "a", "b", "--only"], "only"),
        ] {
            let accepted = ["preset", "no-refine", "truncation", "slope-lines", "only", "output"];
            assert_eq!(
                Args::new(&words).only(words[0], &accepted).unwrap_err().0,
                format!("--{bare} wants a value"),
                "{words:?}"
            );
        }
        let args = Args::new(&["bake", "b.kurven", "--no-refine", "--preset", "plate"]);
        assert!(args.only("bake", &["preset", "no-refine"]).is_ok(), "a switch bare and a value given are read");
    }
}
