//! `kurven-cli app`: the app's command line with no window, run through the
//! session (`kurven_session::headless`), as `KurvenApp BUNDLE --set ...
//! --save ...` runs it in Swift. The session reads and writes nothing; this
//! reads the bundle named on the line and writes what the run saved, baked
//! and dumped.

use std::path::Path;

use kurven_core::bundle::{BundleFiles, write_files};
use kurven_session::headless::{self, Output};

/// Every file under a bundle's directory, by its path in it.
pub fn read_bundle(path: &str) -> Result<BundleFiles, String> {
    fn walk(root: &Path, dir: &Path, files: &mut BundleFiles) -> Result<(), String> {
        let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let p = entry.path();
            if p.is_dir() {
                walk(root, &p, files)?;
            } else {
                let rel = p.strip_prefix(root).map_err(|e| e.to_string())?.to_string_lossy().replace('\\', "/");
                files.insert(rel, std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?);
            }
        }
        Ok(())
    }
    let mut files = BundleFiles::new();
    let root = Path::new(path);
    if !root.is_dir() {
        return Err(format!("{path} is not a bundle directory"));
    }
    walk(root, root, &mut files)?;
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
    let run = headless::run(args, &read_bundle);
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
