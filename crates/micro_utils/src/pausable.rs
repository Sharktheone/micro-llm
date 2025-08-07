use std::error::Error;

pub trait Pausable: Sized {
    type Error: Error;

    fn pause(self) -> Vec<u8>;
    fn resume(buf: Vec<u8>) -> Result<Self, Self::Error>;
}
