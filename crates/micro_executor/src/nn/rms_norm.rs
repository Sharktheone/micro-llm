use ndarray::{Array2, ArrayView1, ScalarOperand, Axis};
use num_traits::{Float, FromPrimitive, Zero};

pub struct RmsNorm<'a, T> {
    pub weight: ArrayView1<'a, T>,
    pub eps: T,
}

impl<'a, T> RmsNorm<'a, T> {
    pub fn new(weight: ArrayView1<'a, T>, eps: T) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array2<T>) -> anyhow::Result<Array2<T>>
    where
        T: Clone + Zero + FromPrimitive + Float + ScalarOperand,
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

        let x2 = x.mapv(|v| v * v);
        let mean = x2
            .mean_axis(Axis(1))
            .ok_or_else(|| anyhow::anyhow!("RmsNorm: failed to compute mean over axis"))?;

        let denom = mean.mapv(|m| (m + self.eps).sqrt()).insert_axis(Axis(1));

        let normalized = x / &denom;
        let weight_b = self
            .weight
            .broadcast((rows, cols))
            .ok_or_else(|| anyhow::anyhow!("RmsNorm: failed to broadcast weight"))?;

        Ok(normalized * &weight_b)
    }
}