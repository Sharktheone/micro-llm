use crate::dim::{Dim1, Dim2, Dim3};
use crate::store::{LoadStore, OwnedStore, RefStore, SharedStore};
use crate::{Backend, DType};

pub type Tensor1<'a, B: Backend, T: DType> = B::Tensor<'a, T, OwnedStore, Dim1>;
pub type Tensor2<'a, B: Backend, T: DType> = B::Tensor<'a, T, OwnedStore, Dim2>;
pub type Tensor3<'a, B: Backend, T: DType> = B::Tensor<'a, T, OwnedStore, Dim3>;

pub type RefTensor1<'a, B: Backend, T: DType> = B::Tensor<'a, T, RefStore, Dim1>;
pub type RefTensor2<'a, B: Backend, T: DType> = B::Tensor<'a, T, RefStore, Dim2>;
pub type RefTensor3<'a, B: Backend, T: DType> = B::Tensor<'a, T, RefStore, Dim3>;

pub type LoadTensor1<'a, B: Backend, T: DType> = B::Tensor<'a, T, LoadStore, Dim1>;
pub type LoadTensor2<'a, B: Backend, T: DType> = B::Tensor<'a, T, LoadStore, Dim2>;
pub type LoadTensor3<'a, B: Backend, T: DType> = B::Tensor<'a, T, LoadStore, Dim3>;

pub type SharedTensor1<'a, B: Backend, T: DType> = B::Tensor<'a, T, SharedStore, Dim1>;
pub type SharedTensor2<'a, B: Backend, T: DType> = B::Tensor<'a, T, SharedStore, Dim2>;
pub type SharedTensor3<'a, B: Backend, T: DType> = B::Tensor<'a, T, SharedStore, Dim3>;
