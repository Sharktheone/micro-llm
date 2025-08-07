use bytemuck::{cast_slice, AnyBitPattern};
use half::{bf16, f16};
use ndarray::{Array1, ArrayView1, ArrayView2, ShapeError};
use num_traits::FromPrimitive;
use safetensors::{Dtype, SafeTensorError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoadError {
    #[error("Tensor not found")]
    NotFound(#[from] SafeTensorError),
    #[error("Invalid shape")]
    Shape(#[from] ShapeError),
    #[error("Invalid dtype, got {0:?}, expected {1:?}")]
    DType(Dtype, Dtype),
    #[error("Invalid dimensions, got {0:?}, expected {1:?}")]
    Dim(u8, u8),
}

pub type LoadResult<T> = Result<T, LoadError>;


pub fn load_array1<'a, T: FromPrimitive + DType + AnyBitPattern>(
    model: &safetensors::SafeTensors<'a>,
    prefix: &str,
) -> LoadResult<ArrayView1<'a, T>> {
    let layer = model.tensor(prefix)?;

    if layer.dtype() != T::dtype() {
        return Err(LoadError::DType(layer.dtype(), T::dtype()));
    }

    if layer.shape().len() != 1 {
        return Err(LoadError::Dim(layer.shape().len() as u8, 1));
    }

    let data = cast_slice::<u8, T>(layer.data());

    Ok(ArrayView1::from_shape(layer.shape()[0], data)?)
}


pub fn load_array2<'a, T: FromPrimitive + DType + AnyBitPattern>(
    model: &safetensors::SafeTensors<'a>,
    prefix: &str,
) -> LoadResult<ArrayView2<'a, T>> {
    let layer = model.tensor(prefix)?;

    if layer.dtype() != T::dtype() {
        return Err(LoadError::DType(layer.dtype(), T::dtype()));
    }

    if layer.shape().len() != 2 {
        return Err(LoadError::Dim(layer.shape().len() as u8, 2));
    }

    let data = cast_slice::<u8, T>(layer.data());
    
    let shape = layer.shape();

    Ok(ArrayView2::from_shape((shape[0], shape[1]), data)?)
}


pub trait DType {
    fn dtype() -> Dtype;
}

macro_rules! impl_dtype {
    ($t:ty, $dtype:expr) => {
        impl DType for $t {
            fn dtype() -> Dtype {
                use safetensors::Dtype::*;
                
                $dtype
            }
        }
    };
}

impl_dtype!(bf16, BF16);
impl_dtype!(f16, F16);
impl_dtype!(f32, F32);
impl_dtype!(f64, F64);
