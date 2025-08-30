mod tensors;

pub use tensors::*;

use std::ops::{Add, Mul};

pub trait Backend: Sized {
    type RefStore: Store<Self>;
    type OwnedStore: Store<Self>;
    type LoadStore: Store<Self>;
    type SharedStore: Store<Self>;

    type Tensor<'a, T: DType<Self>, S: Store<Self>, D: Dim<Self>>: Tensor<'a, T, Self, S, D>;
}

pub trait DType<B: Backend> {}

pub trait Store<B: Backend> {}
pub trait Dim<B: Backend> {}

pub trait Tensor<'a, T: DType<B>, B: Backend, S: Store<B>, D: Dim<B>>: Sized + Add + Mul {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> Self;

    fn to_dtype<U: DType<B>>(&self) -> B::Tensor<'_, U, B::OwnedStore, D>;
}