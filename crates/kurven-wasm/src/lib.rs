//! The engine as the page's WebAssembly module.
//!
//! The binding is the plain C ABI (pointer and length), read with nothing
//! but `WebAssembly.instantiate`: no generated glue, no npm package, no
//! wasm-bindgen CLI to keep in step with a crate (crates/FINDINGS.md, Phase
//! 6, says why this and not wasm-bindgen). Two things are exported.
//!
//! **The session** (`kurven_session::protocol`), one document per module
//! instance, which is one per Worker. The host stages a bundle's files with
//! `kv_stage` (each file's bytes in memory it got from `kv_alloc`), sends a
//! request as JSON to `kv_call`, which answers with the reply's length, and
//! reads the reply at `kv_reply_ptr()` and its typed arrays, the `i`th at
//! `kv_array_ptr(i)` for `kv_array_len(i)` bytes, little-endian. The bytes
//! stay until the next call; a host copies them out (`ArrayBuffer.slice`),
//! and the copy is what a Worker transfers to the page.
//!
//! **The port's golden sets** ([`kurven_golden::all`]), so the check can run
//! them under Node and require the bytes native writes: a host calls
//! `goldens()` once, which generates the sets and returns how many files
//! there are, then reads file `i`'s path and bytes from the module's memory
//! at `golden_path_ptr(i)` (`golden_path_len(i)` bytes) and
//! `golden_bytes_ptr(i)` (`golden_bytes_len(i)` bytes). `bake_gallery_plate`
//! bakes a gallery plate, and `bake_surface_reference` a surface drawing, for
//! timing.

use std::sync::{Mutex, OnceLock};

#[cfg(feature = "goldens")]
use kurven_golden::Golden;
use kurven_session::protocol::{Reply, Session};

// MARK: - the session

/// The last reply's JSON and its arrays' bytes, kept until the next call.
type Kept = Option<(Vec<u8>, Vec<Vec<u8>>)>;

static SESSION: OnceLock<Mutex<(Session, Kept)>> = OnceLock::new();

fn with_session<T>(body: impl FnOnce(&mut Session, &mut Kept) -> T) -> T {
    let lock = SESSION.get_or_init(|| Mutex::new((Session::new(), None)));
    let mut guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (session, reply) = &mut *guard;
    body(session, reply)
}

/// `len` bytes the host may write into, for `kv_stage` and `kv_call`: a
/// boxed slice, whose allocation is exactly its length, so `kv_free` and
/// `kv_stage` can take it back by pointer and length alone.
#[unsafe(no_mangle)]
pub extern "C" fn kv_alloc(len: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; len.max(1)].into_boxed_slice()) as *mut u8
}

/// The allocation `kv_alloc` lent at `ptr` for `len` bytes, taken back.
///
/// # Safety
/// `ptr` and `len` must be a pair `kv_alloc` returned, taken back once.
unsafe fn taken(ptr: *mut u8, len: usize) -> Vec<u8> {
    let boxed = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len.max(1))) };
    let mut bytes = boxed.into_vec();
    bytes.truncate(len);
    bytes
}

/// Give back what `kv_alloc` lent.
///
/// # Safety
/// `ptr` and `len` must be a pair `kv_alloc` returned, freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kv_free(ptr: *mut u8, len: usize) {
    drop(unsafe { taken(ptr, len) });
}

/// The bytes at `ptr`, which the host wrote into memory `kv_alloc` lent.
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, len) } }
}

/// Stage a file of the bundle named `bundle`, at `path` in it (`Session::stage`).
/// The file's bytes are taken as they are, not copied: `data` must be what
/// `kv_alloc` lent, and the session owns it after, so the host does not free
/// it (it frees the name and the path as usual), and a bundle is not held
/// twice while it is staged.
///
/// # Safety
/// Each pointer and length must lie in memory the host wrote; `data` and
/// `data_len` a pair `kv_alloc` returned, not freed after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kv_stage(
    bundle: *const u8,
    bundle_len: usize,
    path: *const u8,
    path_len: usize,
    data: *mut u8,
    data_len: usize,
) -> u32 {
    let data = unsafe { taken(data, data_len) };
    let (Ok(bundle), Ok(path)) = (
        std::str::from_utf8(unsafe { bytes(bundle, bundle_len) }),
        std::str::from_utf8(unsafe { bytes(path, path_len) }),
    ) else {
        return 0;
    };
    with_session(|session, _| session.stage(bundle, path, data));
    1
}

/// Forget every staged bundle.
#[unsafe(no_mangle)]
pub extern "C" fn kv_unstage() {
    with_session(|session, _| session.unstage());
}

/// Answer the JSON request at `ptr` (`Session::call`). Returns the reply's
/// length; the reply and its arrays stay until the next call.
///
/// # Safety
/// `ptr` and `len` must lie in memory the host wrote.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kv_call(ptr: *const u8, len: usize) -> usize {
    let request = String::from_utf8_lossy(unsafe { bytes(ptr, len) }).into_owned();
    with_session(|session, kept| {
        let Reply { json, arrays } = session.call(&request);
        let n = json.len();
        *kept = Some((json.into_bytes(), arrays.into_iter().map(|(_, a)| a.into_bytes()).collect()));
        n
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_reply_ptr() -> *const u8 {
    with_session(|_, kept| kept.as_ref().map_or(std::ptr::null(), |(json, _)| json.as_ptr()))
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_array_count() -> usize {
    with_session(|_, kept| kept.as_ref().map_or(0, |(_, arrays)| arrays.len()))
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_array_ptr(index: usize) -> *const u8 {
    with_session(|_, kept| {
        kept.as_ref().and_then(|(_, arrays)| arrays.get(index)).map_or(std::ptr::null(), |a| a.as_ptr())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn kv_array_len(index: usize) -> usize {
    with_session(|_, kept| kept.as_ref().and_then(|(_, arrays)| arrays.get(index)).map_or(0, |a| a.len()))
}

// MARK: - the golden sets

#[cfg(feature = "goldens")]
static GOLDENS: OnceLock<Vec<Golden>> = OnceLock::new();

#[cfg(feature = "goldens")]
fn goldens_generated() -> &'static [Golden] {
    GOLDENS.get_or_init(kurven_golden::all)
}

#[cfg(feature = "goldens")]
fn golden(index: u32) -> &'static Golden {
    &goldens_generated()[index as usize]
}

/// Generates the golden sets, once, and returns how many files they hold.
#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn goldens() -> u32 {
    goldens_generated().len() as u32
}

#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn golden_path_ptr(index: u32) -> *const u8 {
    golden(index).path.as_ptr()
}

#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn golden_path_len(index: u32) -> usize {
    golden(index).path.len()
}

#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn golden_bytes_ptr(index: u32) -> *const u8 {
    golden(index).bytes.as_ptr()
}

#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn golden_bytes_len(index: u32) -> usize {
    golden(index).bytes.len()
}

/// Builds and bakes a gallery plate in memory (`kurven_golden::gallery::bake_plate`):
/// 0 is gallery-gamma, 1 gallery-cn. Returns its stroke count, or 0 for
/// another number or a plate that does not bake. For timing the engine under
/// Node (`crates/kurven-wasm/bench.mjs`), against native.
#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn bake_gallery_plate(which: u32) -> u32 {
    let name = match which {
        0 => "gallery-gamma",
        1 => "gallery-cn",
        _ => return 0,
    };
    kurven_golden::gallery::bake_plate(name).map_or(0, |b| b.strokes.path_count() as u32)
}

/// Builds and bakes a reference surface drawing in memory
/// (`kurven_golden::surfaces::bake_reference`): 0 is `surface sn`, 1 `surface
/// sn --relief 0.4`, each at the CLI's defaults and a 3000-pixel buffer.
/// Returns its stroke count, or 0 for another number or a drawing that does
/// not bake. For timing under Node (`bench.mjs`), against native.
#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn bake_surface_reference(which: u32) -> u32 {
    kurven_golden::surfaces::bake_reference(which).map_or(0, |b| b.strokes.path_count() as u32)
}

/// Bakes a polyhedron drawing in memory (`kurven_golden::polyhedra::bake_named`):
/// 0 the dodecahedron, 1 the frame under Turing's pattern, 2 the composite
/// of the cubic with a cube and a frame. Returns its stroke count, or 0 for
/// another number. For timing under Node (`bench.mjs`).
#[cfg(feature = "goldens")]
#[unsafe(no_mangle)]
pub extern "C" fn bake_polyhedron(which: u32) -> u32 {
    let name = match which {
        0 => "dodecahedron",
        1 => "frame-turing",
        2 => "composite",
        _ => return 0,
    };
    kurven_golden::polyhedra::bake_named(name).map_or(0, |b| b.strokes.path_count() as u32)
}

#[cfg(all(test, feature = "goldens"))]
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
