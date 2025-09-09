use micro_backend::load::LoadResult;
use micro_backend::{
    Backend, DType, Dim, LoadTensor1, ModelLoader, RefTensor2, SupportsDType, Tensor, Tensor2,
};
use num_traits::{AsPrimitive, Float, FromPrimitive, Zero};
use rayon::slice::{ParallelSlice, ParallelSliceMut};
use std::fmt::{Debug, Display};

pub struct RmsNorm<'a, B: Backend + SupportsDType<T>, T: DType> {
    pub weight: LoadTensor1<'a, B, T>,
    pub eps: f32,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> RmsNorm<'a, B, T> {
    pub fn load(loader: &'a B::Loader, prefix: &str, eps: f32) -> LoadResult<Self> {
        let weight = loader.load_tensor(&format!("{prefix}weight"))?;

        Ok(RmsNorm { weight, eps })
    }

    pub fn forward(&self, x: &RefTensor2<'_, B, T>) -> anyhow::Result<Tensor2<'_, B, T>> {
        let [rows, cols] = x.dim().pattern();

        if rows == 0 || cols == 0 {
            anyhow::bail!("RmsNorm: input has zero-sized dimension: ({rows}, {cols})");
        }
        if self.weight.data().len() != cols {
            anyhow::bail!(
                "RmsNorm: weight length {} does not match input cols {}",
                self.weight.data().len(),
                cols
            );
        }

        let x_view = x.data();
        let weight_view = self.weight.data();

        let mut output = vec![T::zero(); x_view.len()];

        x_view
            .par_chunks(cols)
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

        Ok(Tensor2::<B, T>::from_vec((rows, cols), output))
    }
}
