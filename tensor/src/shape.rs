use super::*;

impl<T, const L: usize> Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    pub fn shape(&self) -> usize {
        L
    }
}

impl<T, const C: usize, const R: usize> Tensor2<T, C, R>
where
    T: Default + Copy + Debug,
    [(); C * R]: ,
{
    pub fn shape(&self) -> (usize, usize) {
        (C, R)
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    pub fn shape(&self) -> (usize, usize, usize) {
        (D1, D2, D3)
    }
}
