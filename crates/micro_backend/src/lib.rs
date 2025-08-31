mod tensors;

use std::fmt::{Debug, Display};
pub use tensors::*;

use std::ops::{Add, Mul};

pub trait Backend: Sized {
    type RefStore: Store<Self>;
    type OwnedStore: Store<Self>;
    type LoadStore: Store<Self>;
    type SharedStore: Store<Self>;

    type Tensor<'a, T: DType, S: Store<Self>, D: Dim<Self>>: Tensor<'a, T, Self, S, D> where Self: SupportsDType<T>;
}



pub trait SupportsDType<D: DType>: Backend {}

pub trait DType: Display + Debug + Copy {}

pub trait Store<B: Backend> {}
pub trait Dim<B: Backend> {}



pub trait OwnedTensor<B: Backend, T: DType, D: Dim<B>> {}

impl<B: Backend + SupportsDType<T>, T: DType, D: Dim<B>> OwnedTensor<B, T, D> for B::Tensor<'_, T, B::OwnedStore, D> {}

pub trait Tensor<'a, T: DType, B: Backend + SupportsDType<T>, S: Store<B>, D: Dim<B>>: Sized + Debug + Clone {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> B::Tensor<'a, T, B::RefStore, D>;

    fn to_dtype<'b, U: DType>(&self) -> B::Tensor<'b, U, B::OwnedStore, D> where B: SupportsDType<U>;

    fn add<'b>(&self, other: &B::Tensor<'_, T, S, D>) -> B::Tensor<'b, T, B::OwnedStore, D>;
    fn mul<'b>(&self, other: &B::Tensor<'_, T, S, D>) -> B::Tensor<'b, T, B::OwnedStore, D>;

    fn mul_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>) where Self: OwnedTensor<B, T, D>;
    fn add_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>) where Self: OwnedTensor<B, T, D>;
}

// pub trait OwnedTensor<'a, T: DType, B: Backend + SupportsDType<T>, D: Dim<B>>: Tensor<'a, T, B, B::OwnedStore, D> {
//     fn mul_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
//     fn add_inplace(&mut self, other: &B::Tensor<'_, T, B::RefStore, D>);
// }