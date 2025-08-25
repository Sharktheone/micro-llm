


pub trait Backend: Sized {
    type Tensor<'a, T: DType<Self>>: Tensor<'a, T, Self>;
}

pub trait DType<B: Backend> {}

pub trait Tensor<'a, T: DType<B>, B: Backend>: Sized {
    fn shape(&self) -> &[usize];
    fn data(&self) -> &[T];

    fn t(&self) -> Self;

    fn to_dtype<U: DType<B>>(&self) -> B::Tensor<'_, U>;
}