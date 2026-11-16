//! A minimal `.npy` reader, the port of `Sources/KurvenCore/NPY.swift`.
//!
//! The bundle's arrays are `.npy` because that is what numpy writes without
//! ceremony and what a reader can parse in eighty lines. This supports exactly
//! what the format is used for here -- versions 1.0 and 2.0, C order, and the
//! four dtypes the bundle declares -- and refuses everything else with a typed
//! error naming what it found. A reader that silently reinterprets a Fortran
//! array as C order returns a transposed landscape; that is precisely the class
//! of failure a typed decode error exists to prevent.
//!
//! The writer comes with the bundle schema (Phase 2).

use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpyError {
    TooShort(String),
    BadMagic,
    UnsupportedVersion { major: u8, minor: u8 },
    MalformedHeader(String),
    FortranOrder,
    UnsupportedDType { found: String, expected: Vec<String> },
    ShapeMismatch { expected: Vec<i64>, found: Vec<usize> },
    Truncated { expected_bytes: usize, found_bytes: usize },
    Io(String),
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpyError::TooShort(what) => write!(f, "npy: too short to contain {what}"),
            NpyError::BadMagic => write!(f, "npy: not a .npy file (bad magic)"),
            NpyError::UnsupportedVersion { major, minor } => {
                write!(f, "npy: unsupported version {major}.{minor}")
            }
            NpyError::MalformedHeader(d) => write!(f, "npy: malformed header ({d})"),
            NpyError::FortranOrder => write!(f, "npy: Fortran-order arrays are not supported"),
            NpyError::UnsupportedDType { found, expected } => {
                write!(f, "npy: dtype {found} is not one of {expected:?}")
            }
            NpyError::ShapeMismatch { expected, found } => {
                write!(f, "npy: expected shape {expected:?}, found {found:?}")
            }
            NpyError::Truncated { expected_bytes, found_bytes } => {
                write!(f, "npy: expected {expected_bytes} bytes of data, found {found_bytes}")
            }
            NpyError::Io(e) => write!(f, "npy: {e}"),
        }
    }
}

impl std::error::Error for NpyError {}

/// The dtypes a bundle may declare. Little-endian only: every consumer of this
/// format is Apple silicon, x86 or wasm, and a big-endian file here would be a
/// bug rather than a portability need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDType {
    Float32,
    Float64,
    Int64,
    Complex128,
}

impl NpyDType {
    pub const ALL: [NpyDType; 4] = [NpyDType::Float32, NpyDType::Float64, NpyDType::Int64, NpyDType::Complex128];

    /// numpy's `descr` for the dtype.
    pub fn descr(self) -> &'static str {
        match self {
            NpyDType::Float32 => "<f4",
            NpyDType::Float64 => "<f8",
            NpyDType::Int64 => "<i8",
            NpyDType::Complex128 => "<c16",
        }
    }

    pub fn item_size(self) -> usize {
        match self {
            NpyDType::Float32 => 4,
            NpyDType::Float64 | NpyDType::Int64 => 8,
            NpyDType::Complex128 => 16,
        }
    }

    fn from_descr(descr: &str) -> Option<NpyDType> {
        NpyDType::ALL.into_iter().find(|d| d.descr() == descr)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NpyArray {
    pub dtype: NpyDType,
    pub shape: Vec<usize>,
    /// Raw little-endian element bytes, C order.
    pub data: Vec<u8>,
}

const MAGIC: [u8; 6] = [0x93, b'N', b'U', b'M', b'P', b'Y'];

pub fn read(path: impl AsRef<Path>) -> Result<NpyArray, NpyError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| NpyError::Io(format!("{}: {e}", path.display())))?;
    parse(&bytes)
}

pub fn parse(bytes: &[u8]) -> Result<NpyArray, NpyError> {
    if bytes.len() < 10 {
        return Err(NpyError::TooShort("a header".into()));
    }
    if bytes[0..6] != MAGIC {
        return Err(NpyError::BadMagic);
    }
    let (major, minor) = (bytes[6], bytes[7]);
    let (header_length, header_start) = match (major, minor) {
        (1, 0) => (usize::from(bytes[8]) | usize::from(bytes[9]) << 8, 10),
        (2, 0) | (3, 0) => {
            if bytes.len() < 12 {
                return Err(NpyError::TooShort("a v2 header".into()));
            }
            let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (n as usize, 12)
        }
        _ => return Err(NpyError::UnsupportedVersion { major, minor }),
    };
    if bytes.len() < header_start + header_length {
        return Err(NpyError::TooShort("its declared header".into()));
    }
    let header = String::from_utf8_lossy(&bytes[header_start..header_start + header_length]);

    let descr = dict_value(&header, "descr")?;
    let fortran = dict_value(&header, "fortran_order")?;
    let shape_text = dict_value(&header, "shape")?;

    if fortran == "True" {
        return Err(NpyError::FortranOrder);
    }
    let dtype = NpyDType::from_descr(descr.trim_matches(|c| c == '\'' || c == '"' || c == ' ')).ok_or_else(|| {
        NpyError::UnsupportedDType {
            found: descr.clone(),
            expected: NpyDType::ALL.iter().map(|d| d.descr().to_string()).collect(),
        }
    })?;
    let shape = parse_shape(&shape_text)?;

    let start = header_start + header_length;
    let expected = shape.iter().product::<usize>() * dtype.item_size();
    let found = bytes.len() - start;
    if found < expected {
        return Err(NpyError::Truncated { expected_bytes: expected, found_bytes: found });
    }
    Ok(NpyArray { dtype, shape, data: bytes[start..start + expected].to_vec() })
}

/// The header is a Python dict literal. Rather than write a Python parser,
/// find the key and take the balanced value that follows it -- the format
/// guarantees the three keys and nothing nested beyond a tuple.
fn dict_value(header: &str, key: &str) -> Result<String, NpyError> {
    let at = header
        .find(&format!("'{key}'"))
        .or_else(|| header.find(&format!("\"{key}\"")))
        .ok_or_else(|| NpyError::MalformedHeader(format!("no key {key}")))?;
    let after = &header[at + key.len() + 2..];
    let colon = after.find(':').ok_or_else(|| NpyError::MalformedHeader(format!("no ':' after {key}")))?;
    let mut depth = 0i32;
    let mut out = String::new();
    for c in after[colon + 1..].chars() {
        if c == '(' || c == '[' {
            depth += 1;
        }
        if c == ')' || c == ']' {
            depth -= 1;
        }
        if (c == ',' || c == '}') && depth == 0 {
            break;
        }
        out.push(c);
    }
    Ok(out.trim().to_string())
}

fn parse_shape(text: &str) -> Result<Vec<usize>, NpyError> {
    let inner = text.trim_matches(|c| c == '(' || c == ')' || c == ' ');
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let t = piece.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<usize>().map_err(|_| NpyError::MalformedHeader(format!("shape component {t}")))?);
    }
    Ok(out)
}

impl NpyArray {
    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }

    fn wrong(&self, expected: &[NpyDType]) -> NpyError {
        NpyError::UnsupportedDType {
            found: self.dtype.descr().to_string(),
            expected: expected.iter().map(|d| d.descr().to_string()).collect(),
        }
    }

    pub fn floats(&self) -> Result<Vec<f32>, NpyError> {
        if self.dtype != NpyDType::Float32 {
            return Err(self.wrong(&[NpyDType::Float32]));
        }
        Ok(self.data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }

    pub fn doubles(&self) -> Result<Vec<f64>, NpyError> {
        match self.dtype {
            NpyDType::Float64 => Ok(self.data.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect()),
            NpyDType::Float32 => Ok(self.floats()?.into_iter().map(f64::from).collect()),
            _ => Err(self.wrong(&[NpyDType::Float64, NpyDType::Float32])),
        }
    }

    /// A complex128 array as interleaved `(re, im)` doubles. Core has no
    /// complex type of its own; the caller pairs them up.
    pub fn complex_pairs(&self) -> Result<Vec<f64>, NpyError> {
        if self.dtype != NpyDType::Complex128 {
            return Err(self.wrong(&[NpyDType::Complex128]));
        }
        Ok(self.data.as_chunks::<8>().0.iter().map(|b| f64::from_le_bytes(*b)).collect())
    }

    pub fn ints(&self) -> Result<Vec<i64>, NpyError> {
        if self.dtype != NpyDType::Int64 {
            return Err(self.wrong(&[NpyDType::Int64]));
        }
        Ok(self.data.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect())
    }

    /// Rows of an `(n, 3)` array, the shape every vertex array in a bundle has.
    pub fn rows3(&self) -> Result<Vec<[f64; 3]>, NpyError> {
        if self.shape.len() != 2 || self.shape[1] != 3 {
            return Err(NpyError::ShapeMismatch { expected: vec![-1, 3], found: self.shape.clone() });
        }
        Ok(self.doubles()?.as_chunks::<3>().0.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;

    /// The oracle set `tests/fixtures/npy`, which numpy wrote, as
    /// `Sources/kurven-test/main.swift` reads it.
    #[test]
    fn every_dtype_the_bundle_declares_reads_back() {
        let f4 = read(fixtures::shared("npy/f4.npy")).unwrap();
        assert!(f4.dtype == NpyDType::Float32 && f4.shape == [3, 5] && f4.floats().unwrap().len() == 15);
        let f8 = read(fixtures::shared("npy/f8.npy")).unwrap();
        assert!(f8.dtype == NpyDType::Float64 && f8.shape == [4, 2]);
        let i8 = read(fixtures::shared("npy/i8.npy")).unwrap();
        assert!(i8.dtype == NpyDType::Int64 && i8.ints().unwrap().len() == 6);
        let c16 = read(fixtures::shared("npy/c16.npy")).unwrap();
        assert!(c16.dtype == NpyDType::Complex128 && c16.shape == [2, 3]);
        let one_d = read(fixtures::shared("npy/f8_1d.npy")).unwrap();
        assert_eq!(one_d.shape, [7], "a 1-d array parses its trailing-comma shape");
        assert_eq!(one_d.doubles().unwrap(), (0..7).map(f64::from).collect::<Vec<_>>());
    }

    #[test]
    fn what_the_reader_cannot_represent_is_an_error_not_a_transposed_landscape() {
        assert_eq!(read(fixtures::shared("npy/reject_fortran.npy")), Err(NpyError::FortranOrder));
        assert!(matches!(read(fixtures::shared("npy/reject_dtype.npy")), Err(NpyError::UnsupportedDType { .. })));
        assert_eq!(parse(b"not an npy file"), Err(NpyError::BadMagic));
        assert!(matches!(parse(b"short"), Err(NpyError::TooShort(_))));
    }

    /// A golden set's arrays, written by Swift: the camera's frozen points.
    #[test]
    fn reads_the_swift_golden_arrays() {
        let points = read(fixtures::shared("camera/recip.points.npy")).unwrap().rows3().unwrap();
        let projected = read(fixtures::shared("camera/recip.projected.npy")).unwrap().rows3().unwrap();
        assert!(!points.is_empty());
        assert_eq!(points.len(), projected.len());
        assert!(points.iter().flatten().all(|v| v.is_finite()));
    }
}
