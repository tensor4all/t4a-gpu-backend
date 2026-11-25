//! Shared helpers of the hardware test targets: device selection, host data in every
//! dtype, and the CPU reference with its error bound.

#![allow(dead_code)]

pub mod rocblas_seam;

use rocblas_seam::DType;
use t4a_hip_runtime::{Error, Runtime};

/// Whether a missing device is a failure (`T4A_REQUIRE_HIP=1`) rather than a skip.
pub fn hip_required() -> bool {
    std::env::var("T4A_REQUIRE_HIP").is_ok_and(|value| value == "1")
}

/// Open device 0, or report why not.
///
/// Without HIP or a device the typed error is printed and `None` returned, unless
/// `T4A_REQUIRE_HIP=1`, in which case the test fails. Any other error always fails: an
/// environment problem must never be reported as a skip.
pub fn runtime() -> Option<Runtime> {
    match Runtime::open(0) {
        Ok(runtime) => Some(runtime),
        Err(error @ (Error::Unavailable { .. } | Error::InvalidDevice { .. })) => {
            assert!(
                !hip_required(),
                "T4A_REQUIRE_HIP=1 but no HIP device is usable: {error}"
            );
            eprintln!("skipped: {error}");
            None
        }
        Err(other) => panic!("opening HIP device 0 failed: {other}"),
    }
}

/// A complex value as `(re, im)`; real dtypes use `im = 0`.
pub type Value = (f64, f64);

/// Deterministic values with full mantissas, so that a product computed at lower
/// precision does not happen to be exact.
pub fn values(dtype: DType, count: usize, seed: u64) -> Vec<Value> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        // xorshift64*, mapped to (-1, 1).
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..count)
        .map(|_| {
            let re = round(dtype, next());
            let im = if dtype.is_complex() {
                round(dtype, next())
            } else {
                0.0
            };
            (re, im)
        })
        .collect()
}

/// Round to the dtype's component precision.
pub fn round(dtype: DType, value: f64) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => f64::from(value as f32),
        DType::F64 | DType::C64 => value,
    }
}

/// Encode values in the dtype's device layout.
pub fn encode(dtype: DType, values: &[Value]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.size());
    for &(re, im) in values {
        match dtype {
            DType::F32 => bytes.extend_from_slice(&(re as f32).to_ne_bytes()),
            DType::F64 => bytes.extend_from_slice(&re.to_ne_bytes()),
            DType::C32 => {
                bytes.extend_from_slice(&(re as f32).to_ne_bytes());
                bytes.extend_from_slice(&(im as f32).to_ne_bytes());
            }
            DType::C64 => {
                bytes.extend_from_slice(&re.to_ne_bytes());
                bytes.extend_from_slice(&im.to_ne_bytes());
            }
        }
    }
    bytes
}

/// Decode the dtype's device layout.
pub fn decode(dtype: DType, bytes: &[u8]) -> Vec<Value> {
    let f32_at = |chunk: &[u8]| {
        let mut word = [0u8; 4];
        word.copy_from_slice(chunk);
        f64::from(f32::from_ne_bytes(word))
    };
    let f64_at = |chunk: &[u8]| {
        let mut word = [0u8; 8];
        word.copy_from_slice(chunk);
        f64::from_ne_bytes(word)
    };
    bytes
        .chunks_exact(dtype.size())
        .map(|chunk| match dtype {
            DType::F32 => (f32_at(chunk), 0.0),
            DType::F64 => (f64_at(chunk), 0.0),
            DType::C32 => (f32_at(&chunk[..4]), f32_at(&chunk[4..])),
            DType::C64 => (f64_at(&chunk[..8]), f64_at(&chunk[8..])),
        })
        .collect()
}

/// A compensated sum (Ogita-Rump-Oishi `Dot2`): the result is as accurate as if it were
/// computed in twice the working precision, so the reference's own rounding is far below
/// the f64 bound it is compared against.
#[derive(Default, Clone, Copy)]
struct Dot2 {
    sum: f64,
    error: f64,
}

impl Dot2 {
    fn add_product(&mut self, x: f64, y: f64) {
        let product = x * y;
        let product_error = x.mul_add(y, -product);
        let sum = self.sum + product;
        let z = sum - self.sum;
        let sum_error = (self.sum - (sum - z)) + (product - z);
        self.sum = sum;
        self.error += product_error + sum_error;
    }

    fn value(self) -> f64 {
        self.sum + self.error
    }
}

/// One column-major operand: its values, leading dimension and transpose flag.
pub struct Operand<'a> {
    pub values: &'a [Value],
    pub ld: usize,
    pub transposed: bool,
}

impl Operand<'_> {
    /// Element `(row, col)` of the operand *as it enters the product*.
    fn at(&self, row: usize, col: usize) -> Value {
        if self.transposed {
            self.values[col + row * self.ld]
        } else {
            self.values[row + col * self.ld]
        }
    }
}

/// The reference product `c = op(a) * op(b)` and, per element, the bound
/// `sum_k |op(a)_ik| |op(b)_kj|` that the error is measured against.
pub fn reference(
    a: &Operand<'_>,
    b: &Operand<'_>,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<Value>, Vec<f64>) {
    let mut c = Vec::with_capacity(m * n);
    let mut magnitude = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            let (mut re, mut im) = (Dot2::default(), Dot2::default());
            let mut bound = 0.0f64;
            for inner in 0..k {
                let (ar, ai) = a.at(row, inner);
                let (br, bi) = b.at(inner, col);
                re.add_product(ar, br);
                re.add_product(-ai, bi);
                im.add_product(ar, bi);
                im.add_product(ai, br);
                bound += (ar.abs() + ai.abs()) * (br.abs() + bi.abs());
            }
            c.push((re.value(), im.value()));
            magnitude.push(bound);
        }
    }
    (c, magnitude)
}

/// Safety factor of the element-wise bound `|c - c_ref| <= FACTOR * k * eps * |a||b|`.
///
/// With machine epsilon (twice the unit roundoff) and a compensated reference, 2 covers
/// a real dot product of length `k` in any summation order; complex elements carry two
/// real products per term, hence the larger factor for them.
pub fn bound_factor(dtype: DType) -> f64 {
    if dtype.is_complex() {
        4.0
    } else {
        2.0
    }
}

/// The largest element of `|observed - expected| / (factor * k * eps * |a||b|)`.
///
/// A value at most 1 passes the bound with the dtype's own epsilon.
pub fn worst_ratio(
    dtype: DType,
    observed: &[Value],
    expected: &[Value],
    magnitude: &[f64],
    k: usize,
) -> f64 {
    assert_eq!(observed.len(), expected.len());
    let scale = bound_factor(dtype) * k as f64 * dtype.epsilon();
    observed
        .iter()
        .zip(expected)
        .zip(magnitude)
        .map(|((&(or, oi), &(er, ei)), &bound)| {
            // Not `f64::max`, which would turn a NaN component into "no error".
            let (re_error, im_error) = ((or - er).abs(), (oi - ei).abs());
            let error = if re_error.is_nan() || im_error.is_nan() {
                f64::INFINITY
            } else {
                re_error.max(im_error)
            };
            if bound == 0.0 {
                if error == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                error / (scale * bound)
            }
        })
        // `f64::max` would drop a NaN and let a NaN result pass; a NaN is a failure.
        .fold(0.0, |worst, ratio| {
            if ratio.is_nan() {
                f64::INFINITY
            } else {
                worst.max(ratio)
            }
        })
}
