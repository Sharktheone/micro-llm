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

pub trait Tensor<'a, T: DType, B: Backend + SupportsDType<T>, S: Store<B>, D: Dim<B>>: Sized + Add + Mul {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> Self;

    fn to_dtype<U: DType>(&self) -> B::Tensor<'_, U, B::OwnedStore, D> where B: SupportsDType<U>;
}