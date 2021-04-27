use super::*;

use std::fmt::Debug;
use std::ops::{AddAssign, MulAssign, Add};
use std::cmp::PartialOrd;

fn relu<T: Default + PartialOrd>(x: T) -> T {
    if x < T::default() {
        T::default()
    } else {
        x
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor3<T, D1, D2, D3> 
where
    T: Copy + Default + Debug,
    [(); D1 * D2 * D3]: ,
{
    pub fn convolution_pass<const N: usize, const K_D1: usize, const K_D2: usize>(
        self,
        kernels: [Tensor3<T, K_D1, K_D2, D3>; N],
        biases: Tensor3<T, D1, D2, N>,
    ) -> Tensor3<T, D1, D2, N>
    where
        T: AddAssign + MulAssign + Add<Output = T> + PartialOrd,
        [(); D1 * D2 * D3]: ,
        [(); K_D1 * K_D2 * D3]: ,
        [(); D1 * D2 * N]: ,
    {
        let conv_out = kernels.map(|kernel| self.convolve_with_pad(kernel).get_data());
        let data = unsafe { std::mem::transmute_copy(&conv_out) };
        (Tensor3::new(data) + biases).map(relu)
    }
}

impl<T, const L: usize> Tensor1<T, L> 
where
    T: Copy + Default + Debug,
{
    pub fn fully_connected_pass<const N: usize>(self, weights: Tensor2<T, L, N>, biases: Tensor1<T, N>) -> Tensor1<T, N>
    where
        [(); L * N]: ,
    {
        let mut data = [T::default(); N * L];
        unimplemented!()
    }
}
