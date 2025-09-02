mod store;
mod tensor;

use half::f16;
use micro_backend::{Backend, DType, Dim, Store, Supports, SupportsDType, Tensor};
use crate::store::{CpuLoadStore, CpuOwnedStore, CpuRefStore, CpuSharedStore, CpuStore};
use crate::tensor::CpuTensor;

pub struct CpuBackend;





impl<T: DType, S: CpuStore, D: Dim> Supports<T, S, D> for CpuBackend
where
    CpuBackend: Supports<T, S, D>,
{
    type _Tensor<'a> = CpuTensor<'a, T, S, D>;
}

impl Backend for CpuBackend {
    type RefStore = CpuRefStore;
    type OwnedStore = CpuOwnedStore;
    type LoadStore = CpuLoadStore;
    type SharedStore = CpuSharedStore;
    type Tensor<'a, T: DType, S: Store, D: Dim>
    = <Self as Supports<T, S, D>>::_Tensor<'a>
    where Self: Supports<T, S, D>;
}


impl SupportsDType<f32> for CpuBackend {}
impl SupportsDType<f64> for CpuBackend {}
impl SupportsDType<f16> for CpuBackend {}