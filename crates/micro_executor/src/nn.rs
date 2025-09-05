mod embedding;
mod gelu;
mod linear;
mod rms_norm;
mod silu;
mod softmax;
pub mod ndarray;


pub use embedding::*;
pub use gelu::*;
pub use linear::*;
pub use rms_norm::*;
pub use silu::*;
pub use softmax::*;

use safetensors::SafeTensors;

pub trait Layer<'a>: Sized {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self>;
}
