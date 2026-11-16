//! Where the fixture sets are on disk, for the native tests and the CLI.
//!
//! Two roots. `tests/fixtures` is the Swift lane's, shared as it is: the
//! oracle sets (`expr/`, `contour/`, `npy/`), which numpy, scipy, mpmath and
//! contourpy wrote, and Swift's golden sets (`contract/`, `camera/`, `clip/`,
//! `hatch/`, ...). The port reads them with the tolerances the Swift suites
//! use. `crates/fixtures` is the port's: `swift/`, the reference drawings
//! `kurven-cli references` baked in Swift, and `rust/`, the port's own golden
//! sets, which `kurven-cli fixtures` writes and the check compares with the
//! commit.
//!
//! The repository root is `KURVEN_ROOT` when that is set, and otherwise the
//! one this crate was built in.

use std::path::PathBuf;

use crate::json::{self, JsonValue};
use crate::npy::{self, NpyArray};

/// The repository's root directory.
pub fn root() -> PathBuf {
    match std::env::var_os("KURVEN_ROOT") {
        Some(root) => PathBuf::from(root),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
    }
}

/// A file of `tests/fixtures`, the sets shared with the Swift lane.
pub fn shared(relative: &str) -> PathBuf {
    root().join("tests/fixtures").join(relative)
}

/// A file of `crates/fixtures`, the port's own sets.
pub fn port(relative: &str) -> PathBuf {
    root().join("crates/fixtures").join(relative)
}

/// A shared fixture's array; panics, naming the file, if it cannot be read,
/// which in a test is the failure to report.
pub fn shared_npy(relative: &str) -> NpyArray {
    npy::read(shared(relative)).unwrap_or_else(|e| panic!("tests/fixtures/{relative}: {e}"))
}

/// A shared fixture's JSON document; panics, naming the file, if it cannot
/// be read or parsed.
pub fn shared_json(relative: &str) -> JsonValue {
    json_at(shared(relative), "tests/fixtures", relative)
}

/// One of the port's own JSON documents; panics as `shared_json` does.
pub fn port_json(relative: &str) -> JsonValue {
    json_at(port(relative), "crates/fixtures", relative)
}

fn json_at(path: PathBuf, set: &str, relative: &str) -> JsonValue {
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{set}/{relative}: {e}"));
    json::parse(&text).unwrap_or_else(|e| panic!("{set}/{relative}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_both_roots() {
        assert!(shared("expr/language.json").is_file(), "no tests/fixtures under {}", root().display());
        assert!(port("swift/index.json").is_file(), "no crates/fixtures/swift under {}", root().display());
    }

    /// The expression language's contract and the landscape specs parse,
    /// the first documents the later phases read.
    #[test]
    fn loads_the_shared_json() {
        let language = shared_json("expr/language.json");
        assert!(language.as_object().is_some());
        let landscapes = shared_json("expr/landscapes.json");
        let list = landscapes.get("landscapes").and_then(JsonValue::as_array).unwrap();
        assert_eq!(list.len(), 4);
        for entry in list {
            let bundle = entry.get("bundle").and_then(JsonValue::as_str).unwrap();
            let manifest = shared_json(&format!("expr/{bundle}/manifest.json"));
            assert_eq!(manifest.get("schema").and_then(JsonValue::as_i64), Some(3), "{bundle}");
            let height = shared_npy(&format!("expr/{bundle}/height.npy"));
            assert_eq!(height.shape.len(), 2, "{bundle}");
        }
    }
}
