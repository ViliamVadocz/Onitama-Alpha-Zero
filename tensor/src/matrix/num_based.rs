use super::*;
use num::{One, Zero};

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Default + Copy + Zero + One,
    [(); S * S]: ,
{
    /// Get the identity matrix. Only implemented for square matrices.
    pub fn identity() -> Self {
        let mut data = [T::zero(); S * S];
        for i in 0..S {
            data[i * (S + 1)] = T::one();
        }
        Matrix(data)
    }
}
