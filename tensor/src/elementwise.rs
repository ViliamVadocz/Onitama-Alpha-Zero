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

    /// Map a function onto each element.
    fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(T) -> T,
    {
        for elem in self.get_data_mut().iter_mut() {
            *elem = f(*elem);
        }
        self
    }
}

impl<T, const L: usize> ElementWiseTensor<T, L> for Tensor1<T, L> where T: Default + Copy + Debug {}

impl<T, const C: usize, const R: usize> ElementWiseTensor<T, { C * R }> for Tensor2<T, C, R> where
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
            impl[const C: usize, const R: usize] $bound for Tensor2<T, C, R>
            where( [(); C * R]: , )
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
            impl[const C: usize, const R: usize] $bound for Tensor2<T, C, R>
            where( [(); C * R]: , )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_tensor1() {
        let a = Vector::new([1, 2, 3, 4, 5, 6, 7]);
        let b = Vector::new([7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(a + b, Vector::new([8; 7]));
        assert_eq!(a - b, Vector::new([-6, -4, -2, 0, 2, 4, 6]));
        assert_eq!(a * b, Vector::new([7, 12, 15, 16, 15, 12, 7]));
        assert_eq!(a / b, Vector::new([0, 0, 0, 1, 1, 3, 7]));
        assert_eq!(a.scale(3), Vector::new([3, 6, 9, 12, 15, 18, 21]));
    }

    #[test]
    fn elementwise_tensor2() {
        let a: Matrix<_, 3, 3> = Matrix::new([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let b: Matrix<_, 3, 3> = Matrix::new([9, 8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(a + b, Matrix::new([10; 9]));
        assert_eq!(a - b, Matrix::new([-8, -6, -4, -2, 0, 2, 4, 6, 8]));
        assert_eq!(a * b, Matrix::new([9, 16, 21, 24, 25, 24, 21, 16, 9]));
        assert_eq!(a / b, Matrix::new([0, 0, 0, 0, 1, 1, 2, 4, 9]));
        assert_eq!(a.scale(3), Matrix::new([3, 6, 9, 12, 15, 18, 21, 24, 27]));
    }

    #[test]
    fn elementwise_tensor3() {
        let a: Tensor3<_, 2, 2, 2> = Tensor3::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let b: Tensor3<_, 2, 2, 2> = Tensor3::new([8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(a + b, Tensor3::new([9; 8]));
        assert_eq!(a - b, Tensor3::new([-7, -5, -3, -1, 1, 3, 5, 7]));
        assert_eq!(a * b, Tensor3::new([8, 14, 18, 20, 20, 18, 14, 8]));
        assert_eq!(a / b, Tensor3::new([0, 0, 0, 0, 1, 2, 3, 8]));
        assert_eq!(a.scale(3), Tensor3::new([3, 6, 9, 12, 15, 18, 21, 24]));
    }

    #[test]
    fn elementwise_assign_tensor1() {
        let a = Vector::new([1, 2, 3, 4, 5, 6, 7]);
        let mut b = a;
        b += Vector::new([4; 7]);
        b *= Vector::new([2; 7]);
        b /= Vector::new([2; 7]);
        b -= Vector::new([4; 7]);
        assert_eq!(a, b);
    }

    #[test]
    fn elementwise_assign_tensor2() {
        let a: Matrix<_, 3, 3> = Matrix::new([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut b = a;
        b += Matrix::new([4; 9]);
        b *= Matrix::new([2; 9]);
        b /= Matrix::new([2; 9]);
        b -= Matrix::new([4; 9]);
        assert_eq!(a, b);
    }

    #[test]
    fn elementwise_assign_tensor3() {
        let a: Tensor3<_, 2, 2, 2> = Tensor3::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut b = a;
        b += Tensor3::new([4; 8]);
        b *= Tensor3::new([2; 8]);
        b /= Tensor3::new([2; 8]);
        b -= Tensor3::new([4; 8]);
        assert_eq!(a, b);
    }

    #[test]
    fn elementwise_map() {
        let a = Tensor3::<_, 2, 2, 2>::new([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            a.map(|x| x + 1),
            Tensor3::<_, 2, 2, 2>::new([2, 3, 4, 5, 6, 7, 8, 9])
        );
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use test::{black_box, Bencher};

    #[bench]
    fn elementwise_op(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        let b = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a * b);
    }

    #[bench]
    fn elementwise_scale(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.scale(5.));
    }

    #[bench]
    fn elementwise_map(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        let f = black_box(|x| x + 1.);
        ben.iter(|| a.map(f));
    }
}
