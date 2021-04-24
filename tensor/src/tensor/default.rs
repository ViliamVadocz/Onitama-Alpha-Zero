use super::*;

impl<T, const L: usize> Default for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn default() -> Self {
        Tensor1::new([T::default(); L])
    }
}

impl<T, const R: usize, const C: usize> Default for Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    fn default() -> Self {
        Tensor2::new([T::default(); R * C])
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Default for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn default() -> Self {
        Tensor3::new([T::default(); D1 * D2 * D3])
    }
}
