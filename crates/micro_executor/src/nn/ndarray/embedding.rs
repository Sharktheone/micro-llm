use crate::load::{DType, LoadResult, Loadable, load_array2};
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
