#![allow(warnings)]
mod tensors;
mod store;
mod dim;

use std::fmt::{Debug, Display};
pub use tensors::*;

use std::ops::{Add, Mul};
use half::f16;
use num_traits::{Float, FromPrimitive, One, Zero};

pub use dim::*;
pub use store::*;

pub trait Backend: Sized {
    type Tensor<'a, T: DType, S: Store, D: Dim>: Tensor<'a, T, Self, S, D> where Self: SupportsDType<T>;
}


pub trait SupportsDType<T: DType>: Backend {
    type _Tensor<'a, S: Store, D: Dim>: Tensor<'a, T, Self, S, D> where Self: Sized;
}

pub trait DType: 'static + Display + Debug + Copy + Float + FromPrimitive + Zero + One + Send + Sync {}

impl DType for f32 {}
impl DType for f64 {}
impl DType for f16 {}



pub trait OwnedTensor<B: Backend, T: DType, D: Dim> {}

impl<B: Backend + SupportsDType<T>, T: DType, D: Dim> OwnedTensor<B, T, D> for B::Tensor<'_, T, OwnedStore, D> {}

pub trait Tensor<'a, T: DType, B: Backend + SupportsDType<T>, S: Store, D: Dim>: Sized + Debug + Clone {
    fn shape(&self) -> &[usize];
    fn dim(&self) -> D;
    fn data(&self) -> &[T];

    fn t(&self) -> B::Tensor<'a, T, RefStore, D>;

    fn to_dtype<'b, U: DType>(&self) -> B::Tensor<'b, U, OwnedStore, D> where B: SupportsDType<U>;
    fn to_owned<'b>(&self) -> B::Tensor<'b, T, OwnedStore, D>;
    fn as_ref(&self) -> B::Tensor<'a, T, RefStore, D>;

    fn add<'b, S2: Store>(&self, other: &B::Tensor<'_, T, S2, D>) -> B::Tensor<'b, T, OwnedStore, D>;
    fn mul<'b, S2: Store>(&self, other: &B::Tensor<'_, T, S2, D>) -> B::Tensor<'b, T, OwnedStore, D>;

    fn mul_inplace<S2: Store>(&mut self, other: &B::Tensor<'_, T, S2, D>) where Self: OwnedTensor<B, T, D>;
    fn add_inplace<S2: Store>(&mut self, other: &B::Tensor<'_, T, S2, D>) where Self: OwnedTensor<B, T, D>;

    fn map(&self, f: impl Fn(T) -> T) -> B::Tensor<'a, T, OwnedStore, D>;
    fn map_inplace(&mut self, f: impl Fn(T) -> T) where Self: OwnedTensor<B, T, D>;

    fn map_threaded(&self, f: impl Fn(T) -> T + Send + Sync) -> B::Tensor<'a, T, OwnedStore, D>;
    fn map_inplace_threaded(&mut self, f: impl Fn(T) -> T + Send + Sync) where Self: OwnedTensor<B, T, D>;

    fn map_batched(&self, f: impl Fn(&[T], &mut [T]) + Send + Sync, batch_size: usize) -> B::Tensor<'a, T, OwnedStore, D>;
    fn map_inplace_batched(&mut self, f: impl Fn(&mut [T]) + Send + Sync, batch_size: usize) where Self: OwnedTensor<B, T, D>;

    fn select(&self, indices: &[usize]) -> B::Tensor<'a, T, OwnedStore, D>;

    fn select_from_start(&self, count: usize) -> B::Tensor<'a, T, RefStore, D>;

    fn slice(&self, ranges: &[std::ops::Range<usize>]) -> B::Tensor<'a, T, OwnedStore, D>;
    
    fn from_slice(data: &'a [T], shape: impl Into<D>) -> B::Tensor<'a, T, RefStore, D>;
    
    fn from_vec(data: Vec<T>, shape: impl Into<D>) -> B::Tensor<'a, T, OwnedStore, D>;
}