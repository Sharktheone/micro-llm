use bytemuck::{AnyBitPattern, Zeroable};
use half::f16;
use ndarray::ScalarOperand;
use ndarray_rand::rand::Rng;
use ndarray_rand::rand_distr::uniform::{SampleBorrow, SampleUniform, UniformSampler};
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use safetensors::Dtype;
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Default, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct F16Wrapper(pub f16);

impl F16Wrapper {
    pub fn new(value: f16) -> Self {
        Self(value)
    }

    pub fn from_f32(value: f32) -> Self {
        Self(f16::from_f32(value))
    }

    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    pub fn inner(self) -> f16 {
        self.0
    }
}

impl From<f16> for F16Wrapper {
    fn from(value: f16) -> Self {
        Self(value)
    }
}

impl From<F16Wrapper> for f16 {
    fn from(wrapper: F16Wrapper) -> Self {
        wrapper.0
    }
}

impl From<f32> for F16Wrapper {
    fn from(value: f32) -> Self {
        Self(f16::from_f32(value))
    }
}

impl Debug for F16Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for F16Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Add for F16Wrapper {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for F16Wrapper {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for F16Wrapper {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for F16Wrapper {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Rem for F16Wrapper {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(f16::from_f32(self.0.to_f32() % rhs.0.to_f32()))
    }
}

impl Neg for F16Wrapper {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for F16Wrapper {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for F16Wrapper {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for F16Wrapper {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for F16Wrapper {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl RemAssign for F16Wrapper {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Zero for F16Wrapper {
    fn zero() -> Self {
        Self(f16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0 == f16::ZERO
    }
}

impl One for F16Wrapper {
    fn one() -> Self {
        Self(f16::ONE)
    }
}

impl Num for F16Wrapper {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(|f| Self(f16::from_f32(f)))
    }
}
// ToPrimitive trait (required for NumCast and Float)
impl ToPrimitive for F16Wrapper {
    fn to_i64(&self) -> Option<i64> {
        self.0.to_f32().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.0.to_f32().to_u64()
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0.to_f32())
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }
}

impl AsPrimitive<f32> for F16Wrapper {
    fn as_(self) -> f32 {
        self.0.to_f32()
    }
}

impl Float for F16Wrapper {
    fn nan() -> Self {
        Self(f16::NAN)
    }

    fn infinity() -> Self {
        Self(f16::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(f16::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(-f16::ZERO)
    }

    fn min_value() -> Self {
        Self(f16::MIN)
    }

    fn min_positive_value() -> Self {
        Self(f16::MIN_POSITIVE)
    }

    fn epsilon() -> Self {
        Self(f16::EPSILON)
    }

    fn max_value() -> Self {
        Self(f16::MAX)
    }

    fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    fn is_infinite(self) -> bool {
        self.0.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    fn is_normal(self) -> bool {
        self.0.is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        self.0.to_f32().classify()
    }

    fn floor(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().floor()))
    }

    fn ceil(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().ceil()))
    }

    fn round(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().round()))
    }

    fn trunc(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().trunc()))
    }

    fn fract(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().fract()))
    }

    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn signum(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().signum()))
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(f16::from_f32(
            self.0.to_f32().mul_add(a.0.to_f32(), b.0.to_f32()),
        ))
    }

    fn recip(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().recip()))
    }

    fn powi(self, n: i32) -> Self {
        Self(f16::from_f32(self.0.to_f32().powi(n)))
    }

    fn powf(self, n: Self) -> Self {
        Self(f16::from_f32(self.0.to_f32().powf(n.0.to_f32())))
    }

    fn sqrt(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().sqrt()))
    }

    fn exp(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().exp()))
    }

    fn exp2(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().exp2()))
    }

    fn ln(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().ln()))
    }

    fn log(self, base: Self) -> Self {
        Self(f16::from_f32(self.0.to_f32().log(base.0.to_f32())))
    }

    fn log2(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().log2()))
    }

    fn log10(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().log10()))
    }

    fn to_degrees(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().to_degrees()))
    }

    fn to_radians(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().to_radians()))
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self(f16::from_f32((self.0.to_f32() - other.0.to_f32()).max(0.0)))
    }

    fn cbrt(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().cbrt()))
    }

    fn hypot(self, other: Self) -> Self {
        Self(f16::from_f32(self.0.to_f32().hypot(other.0.to_f32())))
    }

    fn sin(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().sin()))
    }

    fn cos(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().cos()))
    }

    fn tan(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().tan()))
    }

    fn asin(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().asin()))
    }

    fn acos(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().acos()))
    }

    fn atan(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().atan()))
    }

    fn atan2(self, other: Self) -> Self {
        Self(f16::from_f32(self.0.to_f32().atan2(other.0.to_f32())))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.0.to_f32().sin_cos();
        (Self(f16::from_f32(s)), Self(f16::from_f32(c)))
    }

    fn exp_m1(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().exp_m1()))
    }

    fn ln_1p(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().ln_1p()))
    }

    fn sinh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().sinh()))
    }

    fn cosh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().cosh()))
    }

    fn tanh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().tanh()))
    }

    fn asinh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().asinh()))
    }

    fn acosh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().acosh()))
    }

    fn atanh(self) -> Self {
        Self(f16::from_f32(self.0.to_f32().atanh()))
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.to_f32().integer_decode()
    }
}

impl SampleUniform for F16Wrapper {
    type Sampler = UniformF16Wrapper;
}

#[derive(Clone, Copy, Debug)]
pub struct UniformF16Wrapper {
    low: f32,
    range: f32,
}

impl UniformSampler for UniformF16Wrapper {
    type X = F16Wrapper;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low_f32 = low.borrow().to_f32().unwrap_or(0.0);
        let high_f32 = high.borrow().to_f32().unwrap_or(1.0);
        Self {
            low: low_f32,
            range: high_f32 - low_f32,
        }
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low_f32 = low.borrow().to_f32().unwrap_or(0.0);
        let high_f32 = high.borrow().to_f32().unwrap_or(1.0);
        // For inclusive range, we need to add a small epsilon to include the high value
        let epsilon = f32::EPSILON;
        Self {
            low: low_f32,
            range: high_f32 - low_f32 + epsilon,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        let sample: f32 = rng.r#gen();
        F16Wrapper::from_f32(self.low + sample * self.range)
    }
}

impl AddAssign<&'_ Self> for F16Wrapper {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 += rhs.0;
    }
}

impl FromPrimitive for F16Wrapper {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(f16::from_f32(n as f32)))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(f16::from_f32(n as f32)))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(f16::from_f32(n)))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(f16::from_f64(n)))
    }
}

impl NumCast for F16Wrapper {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(|f| Self(f16::from_f32(f)))
    }
}

impl ScalarOperand for F16Wrapper {}

impl crate::load::DType for F16Wrapper {
    fn dtype() -> Dtype {
        Dtype::F16
    }
}

unsafe impl Zeroable for F16Wrapper {}

unsafe impl AnyBitPattern for F16Wrapper {}
