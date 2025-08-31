use crate::load::{LoadResult, Loadable, load_array1, load_array2};
use crate::nn::Layer;
use ndarray::{Array2, ArrayView1, ArrayView2, LinalgScalar};
use safetensors::SafeTensors;
use micro_backend::{Backend, DType, LoadTensor2, RefTensor2, SupportsDType, Tensor, Tensor2};

pub struct Linear<'a, T> {
    weight: ArrayView2<'a, T>,
    bias: ArrayView1<'a, T>,
}

pub struct LinearNoBias<'a, T> {
    weight: ArrayView2<'a, T>,
}

impl<'a, T: Loadable> Linear<'a, T> {
    pub fn from_safe_tensors(model: &SafeTensors<'a>, prefix: &str) -> LoadResult<Self> {
        let weight = load_array2(model, &format!("{}weight", prefix))?;
        let bias = load_array1(model, &format!("{}bias", prefix))?;
        Ok(Linear { weight, bias })
    }
}

impl<'a, T: Loadable> Layer<'a> for Linear<'a, T> {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self::from_safe_tensors(&safe_tensors, prefix)?)
    }
}

impl<'a, T: Loadable> LinearNoBias<'a, T> {
    pub fn from_safe_tensors(model: &SafeTensors<'a>, prefix: &str) -> LoadResult<Self> {
        let weight = load_array2(model, &format!("{}weight", prefix))?;
        Ok(LinearNoBias { weight })
    }
}

impl<'a, T: Loadable> Layer<'a> for LinearNoBias<'a, T> {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self::from_safe_tensors(&safe_tensors, prefix)?)
    }
}

impl<T: LinalgScalar> Linear<'_, T> {
    pub fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.dot(&self.weight.t()) + &self.bias
    }
}

impl<T: LinalgScalar> LinearNoBias<'_, T> {
    pub fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.dot(&self.weight.t())
    }

    #[allow(dead_code)]
    pub fn forward_view(&self, input: &ArrayView2<T>) -> Array2<T> {
        input.dot(&self.weight.t())
    }
}


pub struct LinearNoBiasB<'a, B: Backend + SupportsDType<T>, T: DType> {
    weight: LoadTensor2<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LinearNoBiasB<'a, B, T> {
    pub fn forward(&self, input: RefTensor2<'_, B, T>) -> Tensor2<'_, B, T> {
        input.mul(&self.weight.t())
    }

    pub fn forward_inplace(&self, input: &mut Tensor2<'_, B, T>) {
        input.mul_inplace(&self.weight.t());
    }
}