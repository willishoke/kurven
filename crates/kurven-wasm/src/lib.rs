//! The engine as the page's WebAssembly module.
//!
//! In Phase 0 the module exports one thing: the port's golden sets
//! ([`kurven_golden::all`]), so the check can run them under Node and require
//! the bytes native writes. The binding is the plain C ABI, read with nothing
//! but `WebAssembly.instantiate`: the harness needs no generated glue and no
//! npm package. The Worker protocol of Phase 6 decides whether the page's
//! binding is this or wasm-bindgen.
//!
//! A host calls `goldens()` once, which generates the sets and returns how
//! many files there are, then reads file `i`'s path and bytes from the
//! module's memory at `golden_path_ptr(i)` (`golden_path_len(i)` bytes) and
//! `golden_bytes_ptr(i)` (`golden_bytes_len(i)` bytes). `bake_gallery_plate`
//! bakes a gallery plate, for timing.

use std::sync::OnceLock;

use kurven_golden::Golden;

static GOLDENS: OnceLock<Vec<Golden>> = OnceLock::new();

fn goldens_generated() -> &'static [Golden] {
    GOLDENS.get_or_init(kurven_golden::all)
}

fn golden(index: u32) -> &'static Golden {
    &goldens_generated()[index as usize]
}

/// Generates the golden sets, once, and returns how many files they hold.
#[unsafe(no_mangle)]
pub extern "C" fn goldens() -> u32 {
    goldens_generated().len() as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn golden_path_ptr(index: u32) -> *const u8 {
    golden(index).path.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn golden_path_len(index: u32) -> usize {
    golden(index).path.len()
}

#[unsafe(no_mangle)]
pub extern "C" fn golden_bytes_ptr(index: u32) -> *const u8 {
    golden(index).bytes.as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn golden_bytes_len(index: u32) -> usize {
    golden(index).bytes.len()
}

/// Builds and bakes a gallery plate in memory (`kurven_golden::gallery::bake_plate`):
/// 0 is gallery-gamma, 1 gallery-cn. Returns its stroke count, or 0 for
/// another number or a plate that does not bake. For timing the engine under
/// Node (`crates/kurven-wasm/bench.mjs`), against native.
#[unsafe(no_mangle)]
pub extern "C" fn bake_gallery_plate(which: u32) -> u32 {
    let name = match which {
        0 => "gallery-gamma",
        1 => "gallery-cn",
        _ => return 0,
    };
    kurven_golden::gallery::bake_plate(name).map_or(0, |b| b.strokes.path_count() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_exports_describe_every_golden() {
        let n = goldens();
        assert_eq!(n as usize, kurven_golden::all().len());
        for i in 0..n {
            let path = unsafe { std::slice::from_raw_parts(golden_path_ptr(i), golden_path_len(i)) };
            assert_eq!(path, golden(i).path.as_bytes());
            assert_eq!(golden_bytes_len(i), golden(i).bytes.len());
        }
    }
}
