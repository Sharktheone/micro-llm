mod store;
mod tensor;

use half::f16;
use micro_backend::{Backend, DType, Dim, Store, Supports, SupportsDType};
use crate::store::{CpuLoadStore, CpuOwnedStore, CpuRefStore, CpuSharedStore, CpuStore};
use crate::tensor::CpuTensor;

pub struct CpuBackend;

impl Backend for CpuBackend {
    type RefStore = CpuRefStore;
    type OwnedStore = CpuOwnedStore;
    type LoadStore = CpuLoadStore;
    type SharedStore = CpuSharedStore;
    type Tensor<'a, T: DType, S: CpuStore, D: Dim>
    = CpuTensor<'a, T, S, D>
    where Self: Supports<T, S, D>;
}


impl SupportsDType<f32> for CpuBackend {}
impl SupportsDType<f64> for CpuBackend {}
impl SupportsDType<f16> for CpuBackend {}