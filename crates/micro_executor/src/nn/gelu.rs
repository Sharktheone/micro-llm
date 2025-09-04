use micro_backend::{Backend, DType, RefTensor2, SupportsDType, Tensor, Tensor2};
use num_traits::{Float, FloatConst, FromPrimitive};

pub fn gelu<T: Float + FromPrimitive>(x: ndarray::Array2<T>) -> ndarray::Array2<T> {
    //TODO: this currently converts to f32, but we should be able to do this without converting

    let x = x.mapv(|x| x.to_f32().unwrap());
    let x = x.mapv(|x| x * 0.5 * (1.0 + libm::erff(x * f32::FRAC_1_SQRT_2())));
    x.mapv(|x| T::from_f32(x).unwrap())
}

pub fn gelu2<B: Backend + SupportsDType<T>, T: DType>(x: RefTensor2<B, T>) -> Tensor2<B, T> {
    x.map_threaded(|x| {
        let x = x.to_f32().unwrap();
        let x = x * 0.5 * (1.0 + libm::erff(x * f32::FRAC_1_SQRT_2()));
        T::from_f32(x).unwrap()
    })
}

pub fn gelu_inplace<B: Backend + SupportsDType<T>, T: DType>(x: &mut Tensor2<B, T>) {
    x.map_inplace_threaded(|v| {
        let x = v.to_f32().unwrap();
        let x = x * 0.5 * (1.0 + libm::erff(x * f32::FRAC_1_SQRT_2()));
        T::from_f32(x).unwrap()
    });
}
