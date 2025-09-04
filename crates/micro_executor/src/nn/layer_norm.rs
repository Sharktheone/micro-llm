use crate::load::{LoadResult, Loadable, load_array1};
use crate::nn::Layer;
use ndarray::{Array2, ArrayView1, ScalarOperand};
use num_traits::{Float, FromPrimitive, Zero};
use safetensors::SafeTensors;

pub struct LayerNorm<'a, T> {
    pub weight: ArrayView1<'a, T>,
    pub bias: ArrayView1<'a, T>,
    pub eps: T,
}

impl<'a, T: Loadable> LayerNorm<'a, T> {
    pub fn from_safe_tensors(model: &SafeTensors<'a>, prefix: &str) -> LoadResult<Self> {
        let weight = load_array1(model, &format!("{}weight", prefix))?;
        let bias = load_array1(model, &format!("{}bias", prefix))?;
        let eps = T::from_f32(1e-5).expect("failed to convert to float");

        Ok(LayerNorm { weight, bias, eps })
    }
}

impl<'a, T: Loadable> Layer<'a> for LayerNorm<'a, T> {
    fn load(safe_tensors: SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self::from_safe_tensors(&safe_tensors, prefix)?)
    }
}

impl<T> LayerNorm<'_, T>
where
    T: Clone + Zero + FromPrimitive + Float + ScalarOperand,
{
    pub fn forward(&self, x: &Array2<T>) -> anyhow::Result<Array2<T>> {
        let mean = x
            .mean_axis(ndarray::Axis(1))
            .ok_or(anyhow::anyhow!("mean error"))?;

        let variance = x.var_axis(ndarray::Axis(1), T::zero());

        let mean = mean.insert_axis(ndarray::Axis(1));
        let mean = mean
            .broadcast((x.shape()[0], x.shape()[1]))
            .ok_or(anyhow::anyhow!("broadcast error"))?;

        let a = (variance + self.eps).sqrt();

        let a = a.insert_axis(ndarray::Axis(1));
        let a = a
            .broadcast((x.shape()[0], x.shape()[1]))
            .ok_or(anyhow::anyhow!("broadcast error"))?;

        let norm = (x - &mean) / a;

        let y = &self.weight * &norm + &self.bias;

        Ok(y)
    }
}
