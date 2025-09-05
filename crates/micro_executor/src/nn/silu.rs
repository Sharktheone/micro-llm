use micro_backend::{Backend, DType, RefTensor2, SupportsDType, Tensor, Tensor2};
use num_traits::{Float, FromPrimitive, One};

pub fn silu_backend<B: Backend + SupportsDType<T>, T: DType>(x: RefTensor2<B, T>) -> Tensor2<B, T> {
    let one = T::one();

    x.map_threaded(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    })
}

pub fn silu_inplace_backend<B: Backend + SupportsDType<T>, T: DType>(x: &mut Tensor2<B, T>) {
    let one = T::one();

    x.map_inplace_threaded(|v| {
        let sigmoid = one / (one + (-v).exp());
        v * sigmoid
    });
}
