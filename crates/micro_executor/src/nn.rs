pub mod linear;
mod embedding;

use safetensors::SafeTensors;

pub trait Layer<'a>: Sized {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self>;
}