mod embedding;
mod gelu;
mod layer_norm;
mod linear;
mod multinomial;
mod softmax;
mod silu;

pub use embedding::*;
pub use gelu::*;
pub use layer_norm::*;
pub use linear::*;
pub use multinomial::*;
pub use softmax::*;
pub use silu::*;


use safetensors::SafeTensors;

pub trait Layer<'a>: Sized {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self>;
}
