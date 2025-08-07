use num_traits::{Float, FloatConst, FromPrimitive};

pub fn gelu<T: Float + FromPrimitive>(x: ndarray::Array2<T>) -> ndarray::Array2<T> {
    //TODO: this currently converts to f32, but we should be able to do this without converting

    let x = x.mapv(|x| x.to_f32().unwrap());
    let x = x.mapv(|x| {
        x * 0.5 * (1.0 + libm::erff(x * f32::FRAC_1_SQRT_2()))
    });
    x.mapv(|x| T::from_f32(x).unwrap())
}
