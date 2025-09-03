use ndarray::{LinalgScalar, ScalarOperand};
use num_traits::Float;
use micro_backend::{Backend, DType, RefTensor1, RefTensor2, RefTensor3, Tensor, Tensor1, Tensor2, Tensor3};

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


pub fn softmax1_<B: Backend, T: DType>(input: &RefTensor1<B, T>) -> RefTensor1<B, T> {
    let max = input.data().iter().cloned().fold(T::min_value(), T::max);

    let mut exp = input.sub_scalar(max);

    exp.map_inplace_threaded(T::exp);

    let sum = exp.sum();

    exp.div_scalar_inplace(sum);

    exp
}

pub fn softmax1_inplace<B: Backend, T: DType>(input: &mut Tensor1<B, T>) {
    let max = input.data().iter().cloned().fold(T::min_value(), T::max);

    input.sub_scalar_inplace(max);

    input.map_inplace_threaded(T::exp);

    let sum = input.sum();

    input.div_scalar_inplace(sum);
}

fn softmax2_<B: Backend, T: DType>(input: &RefTensor2<B, T>, axis: usize) -> Tensor2<B, T> {
    let max = input.map_axis_threaded(axis, |col| {
        col.data().iter().cloned().fold(T::min_value(), T::max)
    });

    let max = max.insert_axis(axis);
    let mut exp = input.sub(&max);
    exp.map_inplace_threaded(T::exp);
    let sum = exp.sum_axis(axis);
    let sum = sum.insert_axis(axis);
    exp.div_inplace(&sum);

    exp
}

fn softmax2_inplace<B: Backend, T: DType>(input: &mut Tensor2<B, T>, axis: usize) {
    let max = input.map_axis_threaded(axis, |col| {
        col.data().iter().cloned().fold(T::min_value(), T::max)
    });

    let max = max.insert_axis(axis);
    input.sub_inplace(&max);
    input.map_inplace_threaded(T::exp);
    let sum = input.sum_axis(axis);
    let sum = sum.insert_axis(axis);
    input.div_inplace(&sum);
}
fn softmax3_<B: Backend, T: DType>(input: &RefTensor3<B, T>, axis: usize) -> Tensor3<B, T> {
    let max = input.map_axis_threaded(axis, |col| {
        col.data().iter().cloned().fold(T::min_value(), T::max)
    });

    let max = max.insert_axis(axis);
    let mut exp = input.sub(&max);
    exp.map_inplace_threaded(T::exp);
    let sum = exp.sum_axis(axis);
    let sum = sum.insert_axis(axis);
    exp.div_inplace(&sum);

    exp
}

fn softmax3_inplace<B: Backend, T: DType>(input: &mut Tensor3<B, T>, axis: usize) {
    let max = input.map_axis_threaded(axis, |col| {
        col.data().iter().cloned().fold(T::min_value(), T::max)
    });

    let max = max.insert_axis(axis);
    input.sub_inplace(&max);
    input.map_inplace_threaded(T::exp);
    let sum = input.sum_axis(axis);
    let sum = sum.insert_axis(axis);
    input.div_inplace(&sum);
}
