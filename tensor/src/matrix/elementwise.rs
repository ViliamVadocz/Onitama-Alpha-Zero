use super::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    /// Multiply all elements in the matrix by a scalar.
    pub fn scale(mut self, scalar: T) -> Self
    where
        T: MulAssign,
    {
        for elem in self.0.iter_mut() {
            *elem *= scalar
        }
        self
    }
}

/// Implement element-wise operations.
macro_rules! impl_op {
    ($bound:ident, $method:ident) => {
        impl<T, const R: usize, const C: usize> $bound for Matrix<T, R, C>
        where
            T: Default + Copy + $bound<Output = T>,
            [(); R * C]: ,
        {
            type Output = Self;

            fn $method(mut self, other: Self) -> Self {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
                    *elem = $bound::$method(*elem, val);
                }
                self
            }
        }
    };
}

/// Implement element-wise assign operations.
macro_rules! impl_op_assign {
    ($bound:ident, $method:ident) => {
        impl<T, const R: usize, const C: usize> $bound for Matrix<T, R, C>
        where
            T: Default + Copy + $bound,
            [(); R * C]: ,
        {
            fn $method(&mut self, other: Self) {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
                    $bound::$method(elem, val);
                }
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);
impl_op_assign!(DivAssign, div_assign);
