mod convolution;
mod elementwise;
mod iter;
mod matmul;
#[cfg(feature = "num")]
mod num_based;
#[cfg(feature = "rand")]
mod random;
mod slice;

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, MulAssign};

/// Matrix is just a wrapper on an array of R * C elements.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<T, const R: usize, const C: usize>([T; R * C])
where
    T: Default + Copy,
    [(); R * C]: ;

impl<T, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C>
where
    T: Default + Copy + fmt::Debug,
    [(); R * C]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..R {
            fmt::Debug::fmt(&self.0[(i * C)..((i + 1) * C)], f)?;
            writeln!(f,)?;
        }
        Ok(())
    }
}

impl<T, const R: usize, const C: usize> Default for Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    fn default() -> Self {
        Matrix([T::default(); R * C])
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    /// Create a Matrix. This function is here to keep the inside data private.
    pub fn new(data: [T; R * C]) -> Self {
        Self(data)
    }

    /// Change the shape of the matrix, keeping the elements in the same order.
    pub fn reshape<const NEW_R: usize, const NEW_C: usize>(self) -> Matrix<T, NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        assert_eq!(NEW_R * NEW_C, R * C);
        let mut data = [T::default(); NEW_R * NEW_C];
        for (elem, &val) in data.iter_mut().zip(self.0.iter()) {
            *elem = val;
        }
        Matrix(data)
    }

    /// Get the transpose of a matrix.
    pub fn transpose(self) -> Matrix<T, C, R>
    where
        [(); C * R]: ,
    {
        let mut data = [T::default(); C * R];
        for (i, elem) in data.iter_mut().enumerate() {
            let r = i % R;
            let c = i / R;
            *elem = self.0[r * C + c];
        }
        Matrix(data)
    }
}
