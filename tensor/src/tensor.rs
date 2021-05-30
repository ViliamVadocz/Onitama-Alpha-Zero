use std::{array::IntoIter, iter::Sum, slice::Iter};

use super::*;

pub type Vector<T, const L: usize> = Tensor1<T, L>;
pub type Matrix<T, const C: usize, const R: usize> = Tensor2<T, C, R>;

pub trait Tensor<T, const X: usize>: Default + Display + Copy
where
    T: Default + Copy + Debug,
{
    fn new(data: [T; X]) -> Self;
    fn get_data(self) -> [T; X];
    fn get_data_ref(&self) -> &[T; X];
    fn get_data_mut(&mut self) -> &mut [T; X];
    fn nth(&self, n: usize) -> T;
    fn iter(&self) -> Iter<'_, T> {
        self.get_data_ref().iter()
    }
    fn into_iter(self) -> IntoIter<T, X> {
        IntoIter::new(self.get_data())
    }
    fn sum(self) -> T
    where
        T: Sum<T>,
    {
        self.into_iter().sum()
    }
    fn rev(self) -> Self {
        let mut data = self.get_data();
        data.reverse();
        Self::new(data)
    }
    fn reshape<G: Tensor<T, X>>(self) -> G {
        G::new(self.get_data())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor1<T, const L: usize>(pub(crate) [T; L])
where
    T: Default + Copy + Debug;

impl<T, const L: usize> Tensor<T, L> for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn new(data: [T; L]) -> Self {
        Tensor1(data)
    }

    fn get_data(self) -> [T; L] {
        self.0
    }

    fn get_data_ref(&self) -> &[T; L] {
        &self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; L] {
        &mut self.0
    }

    fn nth(&self, n: usize) -> T {
        self.0[n]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor2<T, const C: usize, const R: usize>(pub(crate) [T; C * R])
where
    T: Default + Copy + Debug,
    [(); C * R]: ;

impl<T, const C: usize, const R: usize> Tensor<T, { C * R }> for Tensor2<T, C, R>
where
    T: Default + Copy + Debug,
    [(); C * R]: ,
{
    fn new(data: [T; C * R]) -> Self {
        Tensor2(data)
    }

    fn get_data(self) -> [T; C * R] {
        self.0
    }

    fn get_data_ref(&self) -> &[T; C * R] {
        &self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; C * R] {
        &mut self.0
    }

    fn nth(&self, n: usize) -> T {
        self.0[n]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tensor3<T, const D1: usize, const D2: usize, const D3: usize>(pub(crate) [T; D1 * D2 * D3])
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ;

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor<T, { D1 * D2 * D3 }>
    for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn new(data: [T; D1 * D2 * D3]) -> Self {
        Tensor3(data)
    }

    fn get_data(self) -> [T; D1 * D2 * D3] {
        self.0
    }

    fn get_data_ref(&self) -> &[T; D1 * D2 * D3] {
        &self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; D1 * D2 * D3] {
        &mut self.0
    }

    fn nth(&self, n: usize) -> T {
        self.0[n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_tensor1() {
        let _ = Vector::<(), 0>::new([]);
        let _ = Vector::new([0, 1, 2, 3, 4, 5, 6, 7]);
        let _ = Vector::new([0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn create_tensor2() {
        let _ = Matrix::<(), 0, 0>::new([]);
        let _ = Matrix::<_, 2, 4>::new([0, 1, 2, 3, 4, 5, 6, 7]);
        let _ = Matrix::<_, 3, 2>::new([0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn create_tensor3() {
        let _ = Tensor3::<(), 0, 0, 0>::new([]);
        let _ = Tensor3::<_, 2, 2, 2>::new([0, 1, 2, 3, 4, 5, 6, 7]);
        let _ = Tensor3::<_, 1, 2, 3>::new([0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor1::new([0, 1, 2, 3, 4, 4, 5, 6, 7, 9, 10, 11]);
        let b = a
            .reshape::<Tensor2<_, 2, 6>>()
            .reshape::<Tensor3<_, 2, 3, 2>>()
            .reshape::<Tensor2<_, 4, 3>>()
            .reshape();
        assert_eq!(a, b);
    }

    #[test]
    fn nth() {
        let a = Tensor1::new([0, 1, 2, 3, 4]);
        assert_eq!(a.nth(1), 1);
        let a = Tensor2::<_, 2, 3>::new([0, 1, 2, 3, 4, 5]);
        assert_eq!(a.nth(2), 2);
        let a = Tensor3::<_, 2, 2, 1>::new([0, 1, 2, 3]);
        assert_eq!(a.nth(3), 3);
    }

    #[test]
    fn rev() {
        assert_eq!(Tensor1::new([1, 2, 3, 4]).rev(), Tensor1::new([4, 3, 2, 1]));
        assert_eq!(
            Tensor2::<_, 2, 2>::new([1, 2, 3, 4]).rev(),
            Tensor2::new([4, 3, 2, 1])
        );
        assert_eq!(
            Tensor3::<_, 1, 2, 2>::new([1, 2, 3, 4]).rev(),
            Tensor3::new([4, 3, 2, 1])
        );
    }
}
