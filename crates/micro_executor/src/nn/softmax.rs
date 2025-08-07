use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::Float;

pub fn softmax3<T: LinalgScalar + Float>(
    input: ndarray::Array3<T>,
    axis: usize,
) -> anyhow::Result<ndarray::Array3<T>> {
    let max = input.map_axis(ndarray::Axis(axis), |col| {
        col.iter().cloned().fold(T::min_value(), T::max)
    });
    let max = max.insert_axis(ndarray::Axis(axis));
    let exp = (input - &max).mapv(T::exp);
    let sum = exp.sum_axis(ndarray::Axis(axis));
    let sum = sum.insert_axis(ndarray::Axis(axis));
    Ok(exp / &sum)
}

pub fn softmax2<T: LinalgScalar + Float>(
    input: ndarray::ArrayView2<T>,
    axis: usize,
) -> anyhow::Result<ndarray::Array2<T>> {
    let max = input.map_axis(ndarray::Axis(axis), |col| {
        col.iter().cloned().fold(T::min_value(), T::max)
    });
    let max = max.insert_axis(ndarray::Axis(axis));
    let exp = (&input - &max).mapv(T::exp);
    let sum = exp.sum_axis(ndarray::Axis(axis));
    let sum = sum.insert_axis(ndarray::Axis(axis));
    Ok(exp / &sum)
}

pub fn softmax1<T: LinalgScalar + Float + ScalarOperand>(
    input: ndarray::ArrayView1<T>,
) -> anyhow::Result<ndarray::Array1<T>> {
    let max = input.iter().cloned().fold(T::min_value(), T::max);
    let exp = input.mapv(|x| (x - max).exp());
    let sum = exp.sum();
    Ok(exp / sum)
}
