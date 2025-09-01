use std::fmt::Debug;
use std::ops::Range;
use micro_backend::{DType, Dim, OwnedTensor, Store, Supports, SupportsDType, SupportsStore, Tensor};
use crate::CpuBackend;
use crate::store::{CpuOwnedStore, CpuRefStore, CpuStore};

pub struct CpuTensor<'a, T: DType, S: CpuStore, D: Dim> {
    data: S::DataStorage<'a, T>,
    dim: D,
}

impl<'a, T: DType, S: CpuStore, D: Dim> Debug for CpuTensor<'a, T, S, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuTensor")
            .finish()
    }
}

impl<'a, T: DType, S: CpuStore, D: Dim> Clone for CpuTensor<'a, T, S, D> {
    fn clone(&self) -> Self {
        todo!()
    }
}


impl<'a, T: DType, S: CpuStore, D: Dim> Tensor<'a, T, CpuBackend, S, D> for CpuTensor<'a, T, S, D>
where
    CpuBackend: Supports<T, S, D>,
    S: CpuStore,
{
    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn data(&self) -> &[T] {
        todo!()
    }

    fn t(&self) -> CpuTensor<'a, T, CpuRefStore, D> {
        todo!()
    }

    fn to_dtype<'b, U: DType>(&self) -> CpuTensor<'b, U, CpuOwnedStore, D>
    where
        CpuBackend: SupportsDType<U>
    {
        todo!()
    }

    fn to_owned<'b>(&self) -> CpuTensor<'b, T, CpuOwnedStore, D> {
        todo!()
    }

    fn as_ref(&self) -> CpuTensor<'a, T, CpuRefStore, D> {
        todo!()
    }

    fn add<'b, S2: Store>(&self, other: &CpuTensor<'_, T, S2, D>) -> CpuTensor<'b, T, CpuOwnedStore, D>
    where
        CpuBackend: SupportsStore<S2>,
        S2: CpuStore
    {
        todo!()
    }

    fn mul<'b, S2: Store>(&self, other: &CpuTensor<'_, T, S2, D>) -> CpuTensor<'b, T, CpuOwnedStore, D>
    where
        CpuBackend: SupportsStore<S2>,
        S2: CpuStore
    {
        todo!()
    }

    fn mul_inplace<S2: Store>(&mut self, other: &CpuTensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
        CpuBackend: SupportsStore<S2>,
        S2: CpuStore
    {
        todo!()
    }

    fn add_inplace<S2: Store>(&mut self, other: &CpuTensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
        CpuBackend: SupportsStore<S2>,
        S2: CpuStore
    {
        todo!()
    }

    fn map(&self, f: impl Fn(T) -> T) -> CpuTensor<'a, T, CpuOwnedStore, D> {
        todo!()
    }

    fn map_inplace(&mut self, f: impl Fn(T) -> T)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn map_threaded(&self, f: impl Fn(T) -> T + Send + Sync) -> CpuTensor<'a, T, CpuOwnedStore, D> {
        todo!()
    }

    fn map_inplace_threaded(&mut self, f: impl Fn(T) -> T + Send + Sync)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn map_batched(&self, f: impl Fn(&[T], &mut [T]) + Send + Sync, batch_size: usize) -> CpuTensor<'a, T, CpuOwnedStore, D> {
        todo!()
    }

    fn map_inplace_batched(&mut self, f: impl Fn(&mut [T]) + Send + Sync, batch_size: usize)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn select(&self, indices: &[usize]) -> CpuTensor<'a, T, CpuOwnedStore, D> {
        todo!()
    }

    fn select_from_start(&self, count: usize) -> CpuTensor<'a, T, CpuRefStore, D> {
        todo!()
    }

    fn slice(&self, ranges: &[Range<usize>]) -> CpuTensor<'a, T, CpuOwnedStore, D> {
        todo!()
    }
}