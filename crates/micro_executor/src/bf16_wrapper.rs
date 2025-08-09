use bytemuck::{AnyBitPattern, Zeroable};
use half::bf16;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use safetensors::Dtype;
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Bf16Wrapper(pub bf16);

impl Bf16Wrapper {
    pub fn new(value: bf16) -> Self {
        Self(value)
    }

    pub fn from_f32(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }

    pub fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    pub fn inner(self) -> bf16 {
        self.0
    }
}

impl From<bf16> for Bf16Wrapper {
    fn from(value: bf16) -> Self {
        Self(value)
    }
}

impl From<Bf16Wrapper> for bf16 {
    fn from(wrapper: Bf16Wrapper) -> Self {
        wrapper.0
    }
}

impl From<f32> for Bf16Wrapper {
    fn from(value: f32) -> Self {
        Self(bf16::from_f32(value))
    }
}

impl Debug for Bf16Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bf16Wrapper({:?})", self.0)
    }
}

impl Display for Bf16Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_f32())
    }
}

impl Add for Bf16Wrapper {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Bf16Wrapper {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul for Bf16Wrapper {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div for Bf16Wrapper {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl Rem for Bf16Wrapper {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Self(bf16::from_f32(self.0.to_f32() % rhs.0.to_f32()))
    }
}

impl Neg for Bf16Wrapper {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl AddAssign for Bf16Wrapper {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Bf16Wrapper {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Bf16Wrapper {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Bf16Wrapper {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl RemAssign for Bf16Wrapper {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Zero for Bf16Wrapper {
    fn zero() -> Self {
        Self(bf16::ZERO)
    }

    fn is_zero(&self) -> bool {
        self.0 == bf16::ZERO
    }
}

impl One for Bf16Wrapper {
    fn one() -> Self {
        Self(bf16::ONE)
    }
}

impl Num for Bf16Wrapper {
    type FromStrRadixErr = <f32 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        f32::from_str_radix(str, radix).map(|f| Self(bf16::from_f32(f)))
    }
}
// ToPrimitive trait (required for NumCast and Float)
impl ToPrimitive for Bf16Wrapper {
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

impl Float for Bf16Wrapper {
    fn nan() -> Self {
        Self(bf16::NAN)
    }

    fn infinity() -> Self {
        Self(bf16::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self(bf16::NEG_INFINITY)
    }

    fn neg_zero() -> Self {
        Self(-bf16::ZERO)
    }

    fn min_value() -> Self {
        Self(bf16::MIN)
    }

    fn min_positive_value() -> Self {
        Self(bf16::MIN_POSITIVE)
    }

    fn epsilon() -> Self {
        Self(bf16::EPSILON)
    }

    fn max_value() -> Self {
        Self(bf16::MAX)
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
        Self(bf16::from_f32(self.0.to_f32().floor()))
    }

    fn ceil(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().ceil()))
    }

    fn round(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().round()))
    }

    fn trunc(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().trunc()))
    }

    fn fract(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().fract()))
    }

    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn signum(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().signum()))
    }

    fn is_sign_positive(self) -> bool {
        self.0.is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.0.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(bf16::from_f32(
            self.0.to_f32().mul_add(a.0.to_f32(), b.0.to_f32()),
        ))
    }

    fn recip(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().recip()))
    }

    fn powi(self, n: i32) -> Self {
        Self(bf16::from_f32(self.0.to_f32().powi(n)))
    }

    fn powf(self, n: Self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().powf(n.0.to_f32())))
    }

    fn sqrt(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().sqrt()))
    }

    fn exp(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().exp()))
    }

    fn exp2(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().exp2()))
    }

    fn ln(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().ln()))
    }

    fn log(self, base: Self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().log(base.0.to_f32())))
    }

    fn log2(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().log2()))
    }

    fn log10(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().log10()))
    }

    fn to_degrees(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().to_degrees()))
    }

    fn to_radians(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().to_radians()))
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self(bf16::from_f32(
            (self.0.to_f32() - other.0.to_f32()).max(0.0),
        ))
    }

    fn cbrt(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().cbrt()))
    }

    fn hypot(self, other: Self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().hypot(other.0.to_f32())))
    }

    fn sin(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().sin()))
    }

    fn cos(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().cos()))
    }

    fn tan(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().tan()))
    }

    fn asin(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().asin()))
    }

    fn acos(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().acos()))
    }

    fn atan(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().atan()))
    }

    fn atan2(self, other: Self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().atan2(other.0.to_f32())))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.0.to_f32().sin_cos();
        (Self(bf16::from_f32(s)), Self(bf16::from_f32(c)))
    }

    fn exp_m1(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().exp_m1()))
    }

    fn ln_1p(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().ln_1p()))
    }

    fn sinh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().sinh()))
    }

    fn cosh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().cosh()))
    }

    fn tanh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().tanh()))
    }

    fn asinh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().asinh()))
    }

    fn acosh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().acosh()))
    }

    fn atanh(self) -> Self {
        Self(bf16::from_f32(self.0.to_f32().atanh()))
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.0.to_f32().integer_decode()
    }
}

impl FromPrimitive for Bf16Wrapper {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(bf16::from_f32(n as f32)))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(bf16::from_f32(n as f32)))
    }

    fn from_f32(n: f32) -> Option<Self> {
        Some(Self(bf16::from_f32(n)))
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self(bf16::from_f64(n)))
    }
}

impl NumCast for Bf16Wrapper {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        n.to_f32().map(|f| Self(bf16::from_f32(f)))
    }
}

impl ScalarOperand for Bf16Wrapper {}

impl crate::load::DType for Bf16Wrapper {
    fn dtype() -> Dtype {
        Dtype::BF16
    }
}

unsafe impl Zeroable for Bf16Wrapper {}

unsafe impl AnyBitPattern for Bf16Wrapper {}
