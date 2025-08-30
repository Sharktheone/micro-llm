use crate::{Backend, DType};

type Tensor1<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::OwnedStore, Dim1<B>>;
type Tensor2<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::OwnedStore, Dim2<B>>;
type Tensor3<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::OwnedStore, Dim3<B>>;

type RefTensor1<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::RefStore, Dim1<B>>;
type RefTensor2<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::RefStore, Dim2<B>>;
type RefTensor3<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::RefStore, Dim3<B>>;

type LoadTensor1<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::LoadStore, Dim1<B>>;
type LoadTensor2<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::LoadStore, Dim2<B>>;
type LoadTensor3<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::LoadStore, Dim3<B>>;

type SharedTensor1<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::SharedStore, Dim1<B>>;
type SharedTensor2<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::SharedStore, Dim2<B>>;
type SharedTensor3<'a, B: Backend, T: DType<B>> = B::Tensor<'a, T, B::SharedStore, Dim3<B>>;


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
