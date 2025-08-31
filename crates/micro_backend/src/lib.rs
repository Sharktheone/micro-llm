mod tensors;

use std::fmt::{Debug, Display};
pub use tensors::*;

use std::ops::{Add, Mul};
use num_traits::{Float, FromPrimitive, One, Zero};

pub trait Backend: Sized {
    type RefStore: Store<Self>;
    type OwnedStore: Store<Self>;
    type LoadStore: Store<Self>;
    type SharedStore: Store<Self>;

    type Tensor<'a, T: DType, S: Store<Self>, D: Dim<Self>>: Tensor<'a, T, Self, S, D> where Self: SupportsDType<T>;
}



pub trait SupportsDType<D: DType>: Backend {}

pub trait DType: 'static + Display + Debug + Copy + Float + FromPrimitive + Zero + One + Send + Sync {}

pub trait Store<B: Backend> {}
pub trait Dim<B: Backend> {}



pub trait OwnedTensor<B: Backend, T: DType, D: Dim<B>> {}

impl<B: Backend + SupportsDType<T>, T: DType, D: Dim<B>> OwnedTensor<B, T, D> for B::Tensor<'_, T, B::OwnedStore, D> {}

pub trait Tensor<'a, T: DType, B: Backend + SupportsDType<T>, S: Store<B>, D: Dim<B>>: Sized + Debug + Clone {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> B::Tensor<'a, T, B::RefStore, D>;

    fn to_dtype<'b, U: DType>(&self) -> B::Tensor<'b, U, B::OwnedStore, D> where B: SupportsDType<U>;
    fn to_owned<'b>(&self) -> B::Tensor<'b, T, B::OwnedStore, D>;
    fn as_ref(&self) -> B::Tensor<'a, T, B::RefStore, D>;

    fn add<'b>(&self, other: &B::Tensor<'_, T, S, D>) -> B::Tensor<'b, T, B::OwnedStore, D>;
    fn mul<'b>(&self, other: &B::Tensor<'_, T, S, D>) -> B::Tensor<'b, T, B::OwnedStore, D>;

    fn mul_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>) where Self: OwnedTensor<B, T, D>;
    fn add_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>) where Self: OwnedTensor<B, T, D>;

    fn map(&self, f: impl Fn(T) -> T) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace(&mut self, f: impl Fn(T) -> T) where Self: OwnedTensor<B, T, D>;

    fn map_threaded(&self, f: impl Fn(T) -> T + Send + Sync) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace_threaded(&mut self, f: impl Fn(T) -> T + Send + Sync) where Self: OwnedTensor<B, T, D>;

    fn map_batched(&self, f: impl Fn(&[T], &mut [T]) + Send + Sync, batch_size: usize) -> B::Tensor<'a, T, B::OwnedStore, D>;
    fn map_inplace_batched(&mut self, f: impl Fn(&mut [T]) + Send + Sync, batch_size: usize) where Self: OwnedTensor<B, T, D>;
}

// pub trait OwnedTensor<'a, T: DType, B: Backend + SupportsDType<T>, D: Dim<B>>: Tensor<'a, T, B, B::OwnedStore, D> {
//     fn mul_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
//     fn add_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
// }