use crate::CpuBackend;
use crate::store::CpuStore;
use micro_backend::{Backend, DType, Dim, OwnedStore, OwnedTensor, RefStore, RefTensor1, Store, SupportsDType, Tensor};
use std::fmt::Debug;
use std::ops::Range;

pub struct CpuTensor<'a, T: DType, S: CpuStore, D: Dim> {
    data: S::DataStorage<'a, T>,
    dim: D,
}

impl<'a, T: DType, S: CpuStore, D: Dim> Debug for CpuTensor<'a, T, S, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuTensor").finish()
    }
}

impl<'a, T: DType, S: CpuStore, D: Dim> Clone for CpuTensor<'a, T, S, D> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'a, T: DType, S: Store + CpuStore, D: Dim> Tensor<'a, T, CpuBackend, S, D>
    for CpuTensor<'a, T, S, D>
{
    fn shape(&self) -> &[usize] {
        todo!()
    }

    fn data(&self) -> &[T] {
        todo!()
    }

    fn t(&self) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D> {
        todo!()
    }

    fn to_dtype<'b, U: DType>(&self) -> <CpuBackend as Backend>::Tensor<'b, U, OwnedStore, D>
    where
        CpuBackend: SupportsDType<U>,
    {
        todo!()
    }

    fn to_owned<'b>(&self) -> <CpuBackend as Backend>::Tensor<'b, T, OwnedStore, D> {
        todo!()
    }

    fn as_ref(&self) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D> {
        todo!()
    }

    fn add<'b, S2: Store>(
        &self,
        other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>,
    ) -> <CpuBackend as Backend>::Tensor<'b, T, OwnedStore, D> {
        todo!()
    }

    fn mul<'b, S2: Store>(
        &self,
        other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>,
    ) -> <CpuBackend as Backend>::Tensor<'b, T, OwnedStore, D> {
        todo!()
    }

    fn mul_inplace<S2: Store>(&mut self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
    {
        todo!()
    }

    fn add_inplace<S2: Store>(&mut self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
    {
        todo!()
    }

    fn map(&self, f: impl Fn(T) -> T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn map_inplace(&mut self, f: impl Fn(T) -> T)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
    {
        todo!()
    }

    fn map_threaded(
        &self,
        f: impl Fn(T) -> T + Send + Sync,
    ) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn map_inplace_threaded(&mut self, f: impl Fn(T) -> T + Send + Sync)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
    {
        todo!()
    }

    fn map_batched(
        &self,
        f: impl Fn(&[T], &mut [T]) + Send + Sync,
        batch_size: usize,
    ) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn map_inplace_batched(&mut self, f: impl Fn(&mut [T]) + Send + Sync, batch_size: usize)
    where
        Self: OwnedTensor<CpuBackend, T, D>,
    {
        todo!()
    }

    fn select(&self, indices: &[usize]) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn select_from_start(
        &self,
        count: usize,
    ) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D> {
        todo!()
    }

    fn slice(
        &self,
        ranges: &[Range<usize>],
    ) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn dim(&self) -> D {
        todo!()
    }

    fn sub<'b, S2: Store>(&self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>) -> <CpuBackend as Backend>::Tensor<'b, T, OwnedStore, D> {
        todo!()
    }

    fn div<'b, S2: Store>(&self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>) -> <CpuBackend as Backend>::Tensor<'b, T, OwnedStore, D> {
        todo!()
    }

    fn sub_inplace<S2: Store>(&mut self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn div_inplace<S2: Store>(&mut self, other: &<CpuBackend as Backend>::Tensor<'_, T, S2, D>)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn div_scalar(&self, scalar: T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn mul_scalar(&self, scalar: T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn add_scalar(&self, scalar: T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn sub_scalar(&self, scalar: T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn div_scalar_inplace(&mut self, scalar: T)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn mul_scalar_inplace(&mut self, scalar: T)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn add_scalar_inplace(&mut self, scalar: T)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn sub_scalar_inplace(&mut self, scalar: T)
    where
        Self: OwnedTensor<CpuBackend, T, D>
    {
        todo!()
    }

    fn map_axis(&self, axis: usize, f: impl Fn(RefTensor1<CpuBackend, T>) -> T) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D::Smaller> {
        todo!()
    }

    fn map_axis_threaded(&self, axis: usize, f: impl Fn(RefTensor1<CpuBackend, T>) -> T + Send + Sync) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D::Smaller> {
        todo!()
    }

    fn sum_axis(&self, axis: usize) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D::Smaller> {
        todo!()
    }

    fn sum(&self) -> T {
        todo!()
    }

    fn from_slice(data: &'a [T], shape: impl Into<D>) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D> {
        todo!()
    }

    fn from_vec(data: Vec<T>, shape: impl Into<D>) -> <CpuBackend as Backend>::Tensor<'a, T, OwnedStore, D> {
        todo!()
    }

    fn squeeze(&self, axis: usize) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D::Smaller> {
        todo!()
    }

    fn unsqueeze(&self, axis: usize) -> <CpuBackend as Backend>::Tensor<'a, T, RefStore, D::Larger> {
        todo!()
    }

    fn insert_axis(self, axis: usize) -> <CpuBackend as Backend>::Tensor<'a, T, S, D::Larger> {
        todo!()
    }

    fn remove_axis(self, axis: usize) -> <CpuBackend as Backend>::Tensor<'a, T, S, D::Smaller> {
        todo!()
    }
}
