pub trait Store: private::Sealed {}

pub struct RefStore;
pub struct OwnedStore;
pub struct LoadStore;
pub struct SharedStore;

impl Store for RefStore {}
impl Store for OwnedStore {}
impl Store for LoadStore {}
impl Store for SharedStore {}

impl private::Sealed for RefStore {}
impl private::Sealed for OwnedStore {}
impl private::Sealed for LoadStore {}
impl private::Sealed for SharedStore {}

mod private {
    pub trait Sealed {}
}
