use micro_backend::{Backend, DType, RefTensor1, RefTensor2, RefTensor3, SupportsDType, Tensor, Tensor1, Tensor2, Tensor3};
use num_traits::Float;


pub fn softmax1<B: Backend + SupportsDType<T>, T: DType>(input: &RefTensor1<B, T>) -> RefTensor1<B, T> {
    let max = input.data().iter().cloned().fold(T::min_value(), T::max);

    let mut exp = input.sub_scalar(max);

    exp.map_inplace_threaded(T::exp);

    let sum = exp.sum();

    exp.div_scalar_inplace(sum);

    exp
}

pub fn softmax1_inplace<B: Backend  + SupportsDType<T>, T: DType>(input: &mut Tensor1<B, T>) {
    let max = input.data().iter().cloned().fold(T::min_value(), T::max);

    input.sub_scalar_inplace(max);

    input.map_inplace_threaded(T::exp);

    let sum = input.sum();

    input.div_scalar_inplace(sum);
}

fn softmax2<B: Backend + SupportsDType<T>, T: DType>(input: &RefTensor2<B, T>, axis: usize) -> Tensor2<B, T> {
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

fn softmax2_inplace<B: Backend + SupportsDType<T>, T: DType>(input: &mut Tensor2<B, T>, axis: usize) {
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
fn softmax3<B: Backend + SupportsDType<T>, T: DType>(input: &RefTensor3<B, T>, axis: usize) -> Tensor3<B, T> {
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

fn softmax3_inplace<B: Backend + SupportsDType<T>, T: DType>(input: &mut Tensor3<B, T>, axis: usize) {
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
