use bytemuck::AnyBitPattern;
use ndarray::{Array2, ArrayView1, ArrayView2, LinalgScalar};
use num_traits::FromPrimitive;
use safetensors::SafeTensors;
use crate::load::{load_array1, load_array2, DType, LoadResult};
use crate::nn::Layer;

pub struct Linear<'a, T> {
    weight: ArrayView2<'a, T>,
    bias: ArrayView1<'a, T>,
}

pub struct LinearNoBias<'a, T> {
    weight: ArrayView2<'a, T>,
}

impl<'a, T: FromPrimitive + DType + AnyBitPattern> Linear<'a, T> {
    pub fn from_safe_tensors(model: &SafeTensors<'a>, prefix: &str) -> LoadResult<Self> {
        let weight = load_array2(model, &format!("{}weight", prefix))?;
        let bias = load_array1(model, &format!("{}bias", prefix))?;
        Ok(Linear { weight, bias })
    }
}

impl<'a, T: FromPrimitive + DType + AnyBitPattern> Layer<'a> for Linear<'a, T> {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self::from_safe_tensors(&safe_tensors, prefix)?)
    }
}

impl<'a, T: FromPrimitive + DType + AnyBitPattern> LinearNoBias<'a, T> {
    pub fn from_safe_tensors(model: &SafeTensors<'a>, prefix: &str) -> LoadResult<Self> {
        let weight = load_array2(model, &format!("{}weight", prefix))?;
        Ok(LinearNoBias { weight })
    }
}

impl<'a, T: FromPrimitive + DType + AnyBitPattern> Layer<'a> for LinearNoBias<'a, T> {
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
