use crate::nn::Layer;
use micro_backend::{
    Backend, DType, LoadTensor2, ModelLoader, RefTensor2, SupportsDType, Tensor, Tensor2, load,
};

pub struct Linear<'a, B: Backend + SupportsDType<T>, T: DType> {
    weight: LoadTensor2<'a, B, T>,
    bias: LoadTensor2<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> Linear<'a, B, T> {
    pub fn load(loader: &'a B::Loader, prefix: &str) -> load::LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        let bias = loader.load_tensor(&format!("{prefix}bias"))?;

        Ok(Linear { weight, bias })
    }
    pub fn forward(&self, input: RefTensor2<'_, B, T>) -> Tensor2<'_, B, T> {
        input.mul(&self.weight.t()).add(&self.bias)
    }

    pub fn forward_inplace(&self, input: &mut Tensor2<'_, B, T>) {
        input.mul_inplace(&self.weight.t());
        input.add_inplace(&self.bias);
    }
}

pub struct LinearNoBias<'a, B: Backend + SupportsDType<T>, T: DType> {
    weight: LoadTensor2<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LinearNoBias<'a, B, T> {
    pub fn load(loader: &'a B::Loader, prefix: &str) -> load::LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        Ok(LinearNoBias { weight })
    }
    pub fn forward(&self, input: RefTensor2<'_, B, T>) -> Tensor2<'_, B, T> {
        input.mul(&self.weight.t())
    }

    pub fn forward_inplace(&self, input: &mut Tensor2<'_, B, T>) {
        input.mul_inplace(&self.weight.t());
    }
}
