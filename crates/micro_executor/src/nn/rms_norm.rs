use std::fmt::{Debug, Display};
use ndarray::{Array2, ArrayView1, ScalarOperand, Axis};
use num_traits::{AsPrimitive, Float, FromPrimitive, Zero};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use micro_backend::{Backend, DType, Dim, LoadTensor1, ModelLoader, RefTensor2, Tensor, Tensor2};
use micro_backend::load::LoadResult;
use crate::load::{load_array1, Loadable};

pub struct RmsNorm<'a, T> {
    pub weight: ArrayView1<'a, T>,
    pub eps: f32,
}

impl<'a, T: AsPrimitive<f32>> RmsNorm<'a, T> {
    pub fn new(weight: ArrayView1<'a, T>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array2<T>) -> anyhow::Result<Array2<T>>
    where
        T: Clone + Zero + FromPrimitive + Float + ScalarOperand + Display + Debug,
    {
        let (rows, cols) = x.dim();
        if rows == 0 || cols == 0 {
            anyhow::bail!("RmsNorm: input has zero-sized dimension: ({rows}, {cols})");
        }
        if self.weight.len() != cols {
            anyhow::bail!(
                "RmsNorm: weight length {} does not match input cols {}",
                self.weight.len(),
                cols
            );
        }

        let x = x.as_standard_layout();
        let weight = self.weight.as_standard_layout();

        let x_view = x.as_slice().unwrap();
        let weight_view = weight.as_slice().unwrap();

        let mut output = vec![T::zero(); x_view.len()];

        x_view.chunks(cols)
            .zip(output.chunks_mut(cols))
            .for_each(|(src, dst)| {
                let sum2 = src
                    .iter()
                    .map(|&v| {
                        let v = v.as_();
                        v * v
                    })
                    .sum::<f32>();

                let m = (sum2 / cols as f32 + self.eps).sqrt();

                let m = T::from_f32(m).unwrap_or_else(T::nan);
                for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(weight_view) {
                    *d = *s / m * *alpha
                }
            });

        Ok(Array2::from_shape_vec((rows, cols), output)?)
    }
}

impl<'a, T: Loadable> RmsNorm<'a, T> {
    pub fn from_safe_tensors(model: &safetensors::SafeTensors<'a>, prefix: &str, eps: f32) -> anyhow::Result<Self> {
        let weight = load_array1(model, &format!("{}weight", prefix))?;

        Ok(RmsNorm { weight, eps })
    }
}

pub struct RmsNorm2<'a, B: Backend, T: DType> {
    pub weight: LoadTensor1<'a, B, T>,
    pub eps: f32,
}

impl<'a,  B: Backend, T: DType> RmsNorm2<'a, B, T> {
    pub fn load(loader: &B::Loader, prefix: &str, eps: f32) -> LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        Ok(RmsNorm2 { weight, eps })
    }

    pub fn forward(&self, x: &RefTensor2<'_, B, T>) -> anyhow::Result<Tensor2<'_, B, T>> {
        let [rows, cols] = x.dim().pattern();
        
        if rows == 0 || cols == 0 {
            anyhow::bail!("RmsNorm: input has zero-sized dimension: ({rows}, {cols})");
        }
        if self.weight.len() != cols {
            anyhow::bail!(
                "RmsNorm: weight length {} does not match input cols {}",
                self.weight.len(),
                cols
            );
        }

        let x_view = x.data();
        let weight_view = self.weight.data();

        let mut output = vec![T::zero(); x_view.len()];

        x_view.par_chunks(cols)
            .zip(output.par_chunks_mut(cols))
            .for_each(|(src, dst)| {
                let sum2 = src
                    .iter()
                    .map(|&v| {
                        let v = v.as_();
                        v * v
                    })
                    .sum::<f32>();

                let m = (sum2 / cols as f32 + self.eps).sqrt();

                let m = T::from_f32(m).unwrap_or_else(T::nan);
                for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(weight_view) {
                    *d = *s / m * *alpha
                }
            });

        Ok(Tensor2::<B, T>::from_vec((rows, cols), output)?)
    }
}
