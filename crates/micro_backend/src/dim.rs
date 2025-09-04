pub trait Dim: private::Sealed {
    type Pattern;

    fn pattern(&self) -> Self::Pattern;

    type Larger: Dim;
    type Smaller: Dim;
}

pub struct Dim1 {
    dim: [usize; 1],
}

impl Dim for Dim1 {
    type Pattern = [usize; 1];

    fn pattern(&self) -> Self::Pattern {
        self.dim
    }

    type Larger = Dim2;
    type Smaller = Dim1;
}

impl From<[usize; 1]> for Dim1 {
    fn from(pattern: [usize; 1]) -> Self {
        Self { dim: pattern }
    }
}

impl From<(usize,)> for Dim1 {
    fn from(pattern: (usize,)) -> Self {
        Self { dim: [pattern.0] }
    }
}

impl From<usize> for Dim1 {
    fn from(pattern: usize) -> Self {
        Self { dim: [pattern] }
    }
}

pub struct Dim2 {
    dim: [usize; 2],
}

impl Dim for Dim2 {
    type Pattern = [usize; 2];

    fn pattern(&self) -> Self::Pattern {
        self.dim
    }

    type Larger = Dim3;
    type Smaller = Dim1;
}

impl From<[usize; 2]> for Dim2 {
    fn from(pattern: [usize; 2]) -> Self {
        Self { dim: pattern }
    }
}

impl From<(usize, usize)> for Dim2 {
    fn from(pattern: (usize, usize)) -> Self {
        Self {
            dim: [pattern.0, pattern.1],
        }
    }
}

pub struct Dim3 {
    dim: [usize; 3],
}

impl Dim for Dim3 {
    type Pattern = [usize; 3];

    fn pattern(&self) -> Self::Pattern {
        self.dim
    }

    type Larger = Dim3;
    type Smaller = Dim2;
}

impl From<[usize; 3]> for Dim3 {
    fn from(pattern: [usize; 3]) -> Self {
        Self { dim: pattern }
    }
}

impl From<(usize, usize, usize)> for Dim3 {
    fn from(pattern: (usize, usize, usize)) -> Self {
        Self {
            dim: [pattern.0, pattern.1, pattern.2],
        }
    }
}

impl private::Sealed for Dim1 {}
impl private::Sealed for Dim2 {}
impl private::Sealed for Dim3 {}

mod private {
    pub trait Sealed {}
}
