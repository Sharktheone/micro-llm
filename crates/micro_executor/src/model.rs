mod llama;

use safetensors::SafeTensors;

pub trait Model<'a>: Sized {

    type Cache;

    fn from_safetensors(tensors: SafeTensors<'a>) -> anyhow::Result<Self>;

    fn new_cache(&self) -> Self::Cache;

    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache) -> anyhow::Result<u32>;
}