use crate::{Backend, DType, Dim, SupportsDType};

pub type Tensor1<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::OwnedStore, Dim1<B>>;
pub type Tensor2<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::OwnedStore, Dim2<B>>;
pub type Tensor3<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::OwnedStore, Dim3<B>>;

pub type RefTensor1<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::RefStore, Dim1<B>>;
pub type RefTensor2<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::RefStore, Dim2<B>>;
pub type RefTensor3<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::RefStore, Dim3<B>>;

pub type LoadTensor1<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::LoadStore, Dim1<B>>;
pub type LoadTensor2<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::LoadStore, Dim2<B>>;
pub type LoadTensor3<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::LoadStore, Dim3<B>>;

pub type SharedTensor1<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::SharedStore, Dim1<B>>;
pub type SharedTensor2<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::SharedStore, Dim2<B>>;
pub type SharedTensor3<'a, B: Backend + SupportsDType<T>, T: DType> = B::Tensor<'a, T, B::SharedStore, Dim3<B>>;


pub struct Dim1<B: Backend> {
    dim: [usize; 1],
    _marker: std::marker::PhantomData<B>,
}

pub struct Dim2<B: Backend> {
    dim: [usize; 2],
    _marker: std::marker::PhantomData<B>,
}

pub struct Dim3<B: Backend> {
    dim: [usize; 3],
    _marker: std::marker::PhantomData<B>,
}


impl<B: Backend> Dim for Dim1<B> {
    type Larger = Dim2<B>;
    type Smaller = Dim1<B>;
}
impl<B: Backend> Dim for Dim2<B> {
    type Larger = Dim3<B>;
    type Smaller = Dim1<B>;
}
impl<B: Backend> Dim for Dim3<B> {
    type Larger = Dim3<B>;
    type Smaller = Dim2<B>;
}