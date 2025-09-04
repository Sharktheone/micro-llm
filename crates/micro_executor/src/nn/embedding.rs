use crate::load::{DType, LoadResult, Loadable, load_array2};
use micro_backend::{Backend, LoadTensor2, ModelLoader, RefTensor2, Tensor, Tensor2, load};
use ndarray::{Array2, ArrayView2, Axis, s};

pub struct Embedding<'a, T> {
    weight: ArrayView2<'a, T>,
}

impl<'a, T: Loadable> Embedding<'a, T> {
    pub fn from_safe_tensors(
        model: &safetensors::SafeTensors<'a>,
        prefix: &str,
    ) -> LoadResult<Self> {
        let weight = load_array2(model, &format!("{}weight", prefix))?;
        Ok(Embedding { weight })
    }
}

impl<'a, T: Clone> Embedding<'a, T> {
    pub fn forward(&self, input: &[usize]) -> Array2<T> {
        self.weight.select(Axis(0), input)
    }
}

impl<'a, T> Embedding<'a, T> {
    pub fn position_forward(&self, input: usize) -> ArrayView2<'_, T> {
        self.weight.slice(s![..input, ..])
    }
}

pub struct Embedding2<'a, B: Backend, T: DType> {
    weight: LoadTensor2<'a, B, T>,
}

impl<'a, B: Backend, T: DType> Embedding2<'a, B, T> {
    pub fn load(loader: &B::Loader, prefix: &str) -> load::LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        Ok(Embedding2 { weight })
    }

    pub fn forward(&self, input: &[usize]) -> Tensor2<'a, B, T> {
        self.weight.select(input)
    }

    pub fn position_forward(&self, input: usize) -> RefTensor2<'a, B, T> {
        self.weight.select_from_start(input)
    }
}
