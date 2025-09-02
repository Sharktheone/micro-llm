pub trait Dim: private::Sealed {
    type Larger: Dim;
    type Smaller: Dim;
}

pub struct Dim1 {
    dim: [usize; 1],
}

impl Dim for Dim1 {
    type Larger = Dim2;
    type Smaller = Dim1;
}

pub struct Dim2 {
    dim: [usize; 2],
}

impl Dim for Dim2 {
    type Larger = Dim3;
    type Smaller = Dim1;
}

pub struct Dim3 {
    dim: [usize; 3],
}

impl Dim for Dim3 {
    type Larger = Dim3;
    type Smaller = Dim2;
}


impl private::Sealed for Dim1 {}
impl private::Sealed for Dim2 {}
impl private::Sealed for Dim3 {}

mod private {
    pub trait Sealed {}
}
