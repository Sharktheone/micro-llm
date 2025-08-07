mod embedding;
mod gelu;
mod layer_norm;
pub mod linear;
mod multinomial;
mod softmax;
mod silu;

use safetensors::SafeTensors;

pub trait Layer<'a>: Sized {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self>;
}
