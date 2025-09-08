use num_traits::{Float, FromPrimitive, One};

pub fn silu<T: Float + FromPrimitive + One>(mut x: ndarray::Array2<T>) -> ndarray::Array2<T> {
    let one = T::one();

    x.mapv_inplace(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    });

    x
}
