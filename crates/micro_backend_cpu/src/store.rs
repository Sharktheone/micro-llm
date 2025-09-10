use micro_backend::{LoadStore, OwnedStore, RefStore, SharedStore, Store};
use std::ops::Deref;

pub trait CpuStore {
    type DataStorage<'a, T: 'a + Send + Sync>: AsRef<[T]> + Send + Sync;
}

impl<T: Store> CpuStore for T
where
    T: Store,
{
    default type DataStorage<'a, U: 'a + Send + Sync> = &'a [U];
}

impl CpuStore for RefStore {
    type DataStorage<'a, U: 'a + Send + Sync> = &'a [U];
}
impl CpuStore for OwnedStore {
    type DataStorage<'a, U: 'a + Send + Sync> = Vec<U>;
}

impl CpuStore for LoadStore {
    type DataStorage<'a, U: 'a + Send + Sync> = &'a [U];
}

impl CpuStore for SharedStore {
    type DataStorage<'a, U: 'a + Send + Sync> = &'a [U];
}
