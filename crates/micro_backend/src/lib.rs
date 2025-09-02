#![allow(warnings)]
mod tensors;

use std::fmt::{Debug, Display};
pub use tensors::*;

use std::ops::{Add, Mul};
use half::f16;
use num_traits::{Float, FromPrimitive, One, Zero};

pub trait Backend: Sized {
    type RefStore: Store;
    type OwnedStore: Store;
    type LoadStore: Store;
    type SharedStore: Store;

    type Tensor<'a, T: DType, S: Store, D: Dim>: Tensor<'a, T, Self, S, D> where Self: Supports<T, S, D>;
}


pub trait SupportsDType<T: DType> {}
pub trait SupportsStore<S: Store> {}
pub trait SupportsDim<D: Dim> {}


pub trait Supports<T: DType, S: Store, D: Dim>: Backend {
    type _Tensor<'a>: Tensor<'a, T, Self, S, D> where Self: Sized;
}

// impl <B: SupportsDType<T> + SupportsStore<S> + SupportsDim<D>, T: DType, S: Store, D: Dim> Supports<T, S, D> for B {}


pub trait DType: 'static + Display + Debug + Copy + Float + FromPrimitive + Zero + One + Send + Sync {}

impl DType for f32 {}
impl DType for f64 {}
impl DType for f16 {}

pub trait Store {}
pub trait Dim {
    type Larger: Dim;
    type Smaller: Dim;
}



pub trait OwnedTensor<B: Backend, T: DType, D: Dim> {}

impl<B: Backend + Supports<T, B::OwnedStore, D>, T: DType, D: Dim> OwnedTensor<B, T, D> for B::Tensor<'_, T, B::OwnedStore, D> {}

pub trait Tensor<'a, T: DType, B: Backend + Supports<T, S, D>, S: Store, D: Dim>: Sized + Debug + Clone {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> B::Tensor<'a, T, B::RefStore, D>;

    fn to_dtype<'b, U: DType>(&self) -> B::Tensor<'b, U, B::OwnedStore, D> where B: SupportsDType<U>;
    fn to_owned<'b>(&self) -> B::Tensor<'b, T, B::OwnedStore, D>;
    fn as_ref(&self) -> B::Tensor<'a, T, B::RefStore, D>;

    fn add<'b, S2: Store>(&self, other: &B::Tensor<'_, T, S2, D>) -> B::Tensor<'b, T, B::OwnedStore, D> where B: SupportsStore<S2>;
    fn mul<'b, S2: Store>(&self, other: &B::Tensor<'_, T, S2, D>) -> B::Tensor<'b, T, B::OwnedStore, D> where B: SupportsStore<S2>;

    fn mul_inplace<S2: Store>(&mut self, other: &B::Tensor<'_, T, S2, D>) where Self: OwnedTensor<B, T, D>, B: SupportsStore<S2>;
    fn add_inplace<S2: Store>(&mut self, other: &B::Tensor<'_, T, S2, D>) where Self: OwnedTensor<B, T, D>, B: SupportsStore<S2>;

    fn map(&self, f: impl Fn(T) -> T) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace(&mut self, f: impl Fn(T) -> T) where Self: OwnedTensor<B, T, D>;

    fn map_threaded(&self, f: impl Fn(T) -> T + Send + Sync) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace_threaded(&mut self, f: impl Fn(T) -> T + Send + Sync) where Self: OwnedTensor<B, T, D>;

    fn map_batched(&self, f: impl Fn(&[T], &mut [T]) + Send + Sync, batch_size: usize) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace_batched(&mut self, f: impl Fn(&mut [T]) + Send + Sync, batch_size: usize) where Self: OwnedTensor<B, T, D>;

    fn select(&self, indices: &[usize]) -> B::Tensor<'a, T, B::OwnedStore, D>;

    fn select_from_start(&self, count: usize) -> B::Tensor<'a, T, B::RefStore, D>;

    fn slice(&self, ranges: &[std::ops::Range<usize>]) -> B::Tensor<'a, T, B::OwnedStore, D>;

}

// pub trait OwnedTensor<'a, T: DType, B: Backend + SupportsDType<T>, D: Dim<B>>: Tensor<'a, T, B, B::OwnedStore, D> {
//     fn mul_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
//     fn add_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
// }