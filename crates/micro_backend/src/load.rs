use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("Tensor not found")]
    NotFound,
    #[error("Invalid shape")]
    Shape(ShapeError),
    #[error("Invalid dtype, got {0:?}, expected {1:?}")]
    DType(Dtype, Dtype),
    #[error("Invalid dimensions, got {0:?}, expected {1:?}")]
    Dim(u8, u8),
    #[error("{0}")]
    Other(anyhow::Error)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    IncompatibleShape = 1,
    IncompatibleLayout,
    RangeLimited,
    OutOfBounds,
    Unsupported,
    Overflow,
}


pub type LoadResult<T> = Result<T, LoadError>;
