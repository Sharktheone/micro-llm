#![feature(specialization)]
#![allow(warnings)]

mod loader;
mod store;
mod tensor;

use crate::loader::CpuLoader;
use crate::tensor::CpuTensor;
use half::f16;
use micro_backend::{Backend, DType, Dim, Store, SupportsDType, Tensor};

pub struct CpuBackend;

impl<T: DType> SupportsDType<T> for CpuBackend {
    type _Tensor<'a, S: Store, D: Dim> = CpuTensor<'a, T, S, D>;
}

impl Backend for CpuBackend {
    type Tensor<'a, T: DType, S: Store, D: Dim>
        = <Self as SupportsDType<T>>::_Tensor<'a, S, D>
    where
        Self: SupportsDType<T>;

    type Loader = CpuLoader;
}
