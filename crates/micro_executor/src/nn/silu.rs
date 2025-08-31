use num_traits::{Float, FromPrimitive, One};
use micro_backend::{Backend, DType, RefTensor2, SupportsDType, Tensor, Tensor2};

pub fn silu<T: Float + FromPrimitive + One>(mut x: ndarray::Array2<T>) -> ndarray::Array2<T> {
    let one = T::one();

    x.mapv_inplace(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    });

    x
}

pub fn silu2<B: Backend + SupportsDType<T>, T: DType>(x: RefTensor2<B, T>) -> Tensor2<B, T> {
    let one = T::one();

    x.map_threaded(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    })
}

pub fn silu_inplace<B: Backend + SupportsDType<T>, T: DType>(x: &mut Tensor2<B, T>) {
    let one = T::one();

    x.map_inplace_threaded(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    });
}
