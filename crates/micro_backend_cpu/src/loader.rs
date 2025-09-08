use crate::CpuBackend;
use micro_backend::load::LoadResult;
use micro_backend::{Backend, DType, Dim, LoadStore, ModelLoader, SupportsDType};
use std::path::Path;

pub struct CpuLoader {}

impl ModelLoader<CpuBackend> for CpuLoader {
    fn load_model(backend: &mut CpuBackend, files: &[impl AsRef<Path>]) -> anyhow::Result<Self> {
        todo!()
    }

    fn to_dtype<T: DType>(self) -> Self
    where
        CpuBackend: SupportsDType<T>,
    {
        todo!()
    }

    fn load_tensor<T: DType, D: Dim>(
        &self,
        name: &str,
    ) -> LoadResult<<CpuBackend as Backend>::Tensor<'_, T, LoadStore, D>>
    where
        CpuBackend: SupportsDType<T>,
    {
        todo!()
    }
}
