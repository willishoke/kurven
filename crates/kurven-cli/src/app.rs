//! `kurven-cli app`: the app's command line with no window, run through the
//! session (`kurven_session::headless`), as `KurvenApp BUNDLE --set ...
//! --save ...` runs it in Swift. The session reads and writes nothing; this
//! reads the bundle named on the line and writes what the run saved, baked
//! and dumped.

use std::borrow::Cow;
use std::path::Path;

use kurven_core::bundle::{BundleFiles, write_files};
use kurven_session::headless::{self, Output};

/// How deep a bundle's directories go: `layers/` is the deepest a bundle
/// writes, so this is ample, and it ends any walk that would not end.
const DEPTH: usize = 8;

/// Every file under a bundle's directory, by its path in it: its regular
/// files alone. A symbolic link is not followed (one pointing up the tree
/// would walk forever), and a dotfile is not a bundle's (`.DS_Store`, a
/// temporary manifest).
pub fn read_bundle(path: &str) -> Result<BundleFiles, String> {
    fn walk(root: &Path, dir: &Path, depth: usize, files: &mut BundleFiles) -> Result<(), String> {
        if depth > DEPTH {
            return Err(format!("{}: deeper than a bundle goes", dir.display()));
        }
        let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let p = entry.path();
            // The entry's own type, not its target's: a link is neither.
            let kind = entry.file_type().map_err(|e| format!("{}: {e}", p.display()))?;
            if kind.is_dir() {
                walk(root, &p, depth + 1, files)?;
            } else if kind.is_file() {
                let rel = p.strip_prefix(root).map_err(|e| e.to_string())?.to_string_lossy().replace('\\', "/");
                files.insert(&rel, std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?);
            }
        }
        Ok(())
    }
    let mut files = BundleFiles::new();
    let root = Path::new(path);
    if !root.is_dir() {
        return Err(format!("{path} is not a bundle directory"));
    }
    walk(root, root, 0, &mut files)?;
    Ok(files)
}

/// Write what a run made, each where it was asked to go.
pub fn write_outputs(outputs: &[Output]) -> Result<(), String> {
    for output in outputs {
        match output {
            Output::Bundle { path, files } => write_files(files, path).map_err(|e| e.to_string())?,
            Output::File { path, bytes } => {
                if let Some(parent) = Path::new(path).parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                std::fs::write(path, bytes).map_err(|e| format!("{path}: {e}"))?;
            }
        }
    }
    Ok(())
}

/// Run the app's command line: print what it printed, write what it wrote,
/// and exit as it exits.
pub fn app_command(args: &[String]) -> i32 {
    let run = headless::run(args, &|path: &str| read_bundle(path).map(Cow::Owned));
    for line in &run.stdout {
        println!("{line}");
    }
    for line in &run.stderr {
        eprintln!("{line}");
    }
    if let Err(error) = write_outputs(&run.outputs) {
        eprintln!("Kurven: {error}");
        return 1;
    }
    run.exit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bundle_is_its_regular_files_no_link_followed_no_dotfile() {
        let dir = std::env::temp_dir().join(format!("kurven-read-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("layers")).unwrap();
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();
        std::fs::write(dir.join("layers/a.npy"), b"a").unwrap();
        std::fs::write(dir.join(".DS_Store"), b"x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&dir, dir.join("layers/loop")).unwrap();
        let files = read_bundle(dir.to_str().unwrap()).unwrap();
        assert_eq!(files.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["layers/a.npy", "manifest.json"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
