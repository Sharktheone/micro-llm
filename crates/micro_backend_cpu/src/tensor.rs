use micro_backend::{DType, Dim, Store};

pub struct CpuTensor<'a, T: DType, S: Store, D: Dim> {
    data: S,
}
