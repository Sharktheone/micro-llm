use micro_backend::{
    Backend, DType, LoadTensor2, ModelLoader, RefTensor2, SupportsDType, Tensor, Tensor2, load,
};

pub struct Embedding<'a, B: Backend + SupportsDType<T>, T: DType> {
    weight: LoadTensor2<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> Embedding<'a, B, T> {
    pub fn load(loader: &'a B::Loader, prefix: &str) -> load::LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        Ok(Embedding { weight })
    }

    pub fn forward(&self, input: &[usize]) -> Tensor2<'a, B, T> {
        self.weight.select(input)
    }

    pub fn position_forward(&self, input: usize) -> RefTensor2<'a, B, T> {
        self.weight.select_from_start(input)
    }
}
