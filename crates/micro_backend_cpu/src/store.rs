use std::ops::Deref;
use micro_backend::{Store, SupportsStore};
use crate::CpuBackend;

pub struct CpuRefStore;
pub struct CpuOwnedStore;
pub struct CpuLoadStore;

pub struct CpuSharedStore;



pub trait CpuStore: Store {
    type DataStorage<'a, T: 'a>: Deref<Target=[T]>;
}

impl<T: CpuStore> SupportsStore<T> for CpuBackend {}

impl CpuStore for CpuRefStore {
    type DataStorage<'a, U: 'a> = &'a [U];

}

impl Store for CpuRefStore {}

impl CpuStore for CpuOwnedStore {
    type DataStorage<'a, U: 'a> = Vec<U>;
}

impl Store for CpuOwnedStore {}

impl CpuStore for CpuLoadStore {
    type DataStorage<'a, U: 'a> = &'a [U];
}

impl Store for CpuLoadStore {}
impl CpuStore for CpuSharedStore {
    type DataStorage<'a, U: 'a> = &'a [U];
}

impl Store for CpuSharedStore {}