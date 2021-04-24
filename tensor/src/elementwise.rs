use super::*;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

pub trait ElementWiseTensor<T, const X: usize>: Tensor<T, X>
where
    T: Default + Copy + Debug,
    [(); X]: ,
{
    /// Multiply all elements by a scalar.
    fn scale(mut self, scalar: T) -> Self
    where
        T: MulAssign,
    {
        for elem in self.get_data_mut().iter_mut() {
            *elem *= scalar;
        }
        self
    }
}

impl<T, const L: usize> ElementWiseTensor<T, L> for Tensor1<T, L> where T: Default + Copy + Debug {}

impl<T, const R: usize, const C: usize> ElementWiseTensor<T, { R * C }> for Tensor2<T, R, C> where
    T: Default + Copy + Debug
{
}

impl<T, const D1: usize, const D2: usize, const D3: usize> ElementWiseTensor<T, { D1 * D2 * D3 }>
    for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
{
}

// Define macros to help implement all elementwise operation traits.
macro_rules! impl_op_inner {
    (
        impl[$($generics:tt)*] $bound:ident for $type:ty
        $( where($($predicates:tt)*) )?
        {$method:ident}
    ) => {
        impl<T, $($generics)*> $bound for $type
        where
            T: Default + Copy + Debug + $bound<Output = T>,
            $($($predicates)*)?
        {
            type Output = Self;

            fn $method(mut self, other: Self) -> Self {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
                    *elem = $bound::$method(*elem, val);
                }
                self
            }
        }
    }
}

macro_rules! impl_op_assign_inner {
    (
        impl[$($generics:tt)*] $bound:ident for $type:ty
        $( where($($predicates:tt)*) )?
        {$method:ident}
    ) => {
        impl<T, $($generics)*> $bound for $type
        where
            T: Default + Copy + Debug + $bound,
            $($($predicates)*)?
        {
            fn $method(&mut self, other: Self) {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
                    $bound::$method(elem, val);
                }
            }
        }
    }
}

macro_rules! impl_op_1 {
    ($bound:ident, $method:ident) => {
        impl_op_inner! {
            impl[const L: usize] $bound for Tensor1<T, L> {
                $method
            }
        }
    };
}

macro_rules! impl_op_2 {
    ($bound:ident, $method:ident) => {
        impl_op_inner! {
            impl[const R: usize, const C: usize] $bound for Tensor2<T, R, C>
            where( [(); R * C]: , )
            {$method}
        }
    };
}

macro_rules! impl_op_3 {
    ($bound:ident, $method:ident) => {
        impl_op_inner!{
            impl[const D1: usize, const D2: usize, const D3: usize] $bound for Tensor3<T, D1, D2, D3>
            where( [(); D1 * D2 * D3]: , )
            {$method}
        }
    };
}

macro_rules! impl_op_assign_1 {
    ($bound:ident, $method:ident) => {
        impl_op_assign_inner! {
            impl[const L: usize] $bound for Tensor1<T, L>
            where( )
            {$method}
        }
    };
}

macro_rules! impl_op_assign_2 {
    ($bound:ident, $method:ident) => {
        impl_op_assign_inner! {
            impl[const R: usize, const C: usize] $bound for Tensor2<T, R, C>
            where( [(); R * C]: , )
            {$method}
        }
    };
}

macro_rules! impl_op_assign_3 {
    ($bound:ident, $method:ident) => {
        impl_op_assign_inner!{
            impl[const D1: usize, const D2: usize, const D3: usize] $bound for Tensor3<T, D1, D2, D3>
            where( [(); D1 * D2 * D3]: , )
            {$method}
        }
    };
}

// Implement elementwise operations
impl_op_1!(Add, add);
impl_op_1!(Sub, sub);
impl_op_1!(Mul, mul);
impl_op_1!(Div, div);
impl_op_assign_1!(AddAssign, add_assign);
impl_op_assign_1!(SubAssign, sub_assign);
impl_op_assign_1!(MulAssign, mul_assign);
impl_op_assign_1!(DivAssign, div_assign);
impl_op_2!(Add, add);
impl_op_2!(Sub, sub);
impl_op_2!(Mul, mul);
impl_op_2!(Div, div);
impl_op_assign_2!(AddAssign, add_assign);
impl_op_assign_2!(SubAssign, sub_assign);
impl_op_assign_2!(MulAssign, mul_assign);
impl_op_assign_2!(DivAssign, div_assign);
impl_op_3!(Add, add);
impl_op_3!(Sub, sub);
impl_op_3!(Mul, mul);
impl_op_3!(Div, div);
impl_op_assign_3!(AddAssign, add_assign);
impl_op_assign_3!(SubAssign, sub_assign);
impl_op_assign_3!(MulAssign, mul_assign);
impl_op_assign_3!(DivAssign, div_assign);
