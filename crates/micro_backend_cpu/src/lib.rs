mod store;
mod tensor;

use half::f16;
use micro_backend::{Backend, DType, Dim, Store, SupportsDType};

pub struct CpuBackend;

impl Backend for CpuBackend {
    type RefStore = CpuRefStore;
    type OwnedStore = CpuOwnedStore;
    type LoadStore = CpuLoadStore;
    type SharedStore = CpuSharedStore;
    type Tensor<'a, T: DType, S: Store<Self>, D: Dim<Self>>
    where
        Self: SupportsDType<T>
    = CpuTensor<'a, T, S, D>;
}


impl SupportsDType<f32> for CpuBackend {}
impl SupportsDType<f64> for CpuBackend {}
impl SupportsDType<f16> for CpuBackend {}