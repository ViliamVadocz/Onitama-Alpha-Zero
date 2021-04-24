use super::*;
use std::ops::{AddAssign, MulAssign};

impl<T, const L: usize> Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    /// Naive convolution.
    pub fn convolve<const KERN_L: usize>(
        self,
        kernel: Tensor1<T, KERN_L>,
    ) -> Tensor1<T, { L + 1 - KERN_L }>
    where
        T: AddAssign + MulAssign,
    {
        let mut res = Tensor1::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            res += self.slice(i).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to preserve shape.
    pub fn convolve_with_pad<const KERN_L: usize>(self, kernel: Tensor1<T, KERN_L>) -> Tensor1<T, L>
    where
        T: AddAssign + MulAssign,
    {
        let mut res = Tensor1::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            res += self.slice_with_pad(i - KERN_L as isize / 2).scale(val)
        }
        res
    }
}

impl<T, const R: usize, const C: usize> Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    /// Naive convolution.
    pub fn convolve<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Tensor2<T, KERN_R, KERN_C>,
    ) -> Tensor2<T, { R + 1 - KERN_R }, { C + 1 - KERN_C }>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + 1 - KERN_R) * (C + 1 - KERN_C)]: ,
    {
        let mut res = Tensor2::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let r = i / KERN_C;
            let c = i % KERN_C;
            res += self.slice(r, c).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to preserve shape.
    pub fn convolve_with_pad<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Tensor2<T, KERN_R, KERN_C>,
    ) -> Tensor2<T, R, C>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + KERN_R - 1) * (C + KERN_C - 1)]: ,
    {
        let mut res = Tensor2::default();
        let kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let r = i / kern_c as isize - kern_r / 2;
            let c = i % kern_c as isize - kern_c / 2;
            res += self.slice_with_pad(r, c).scale(val);
        }
        res
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    /// Naive convolution.
    pub fn convolve<const KERN_D1: usize, const KERN_D2: usize, const KERN_D3: usize>(
        self,
        kernel: Tensor3<T, KERN_D1, KERN_D2, KERN_D3>,
    ) -> Tensor3<T, { D1 + 1 - KERN_D1 }, { D2 + 1 - KERN_D2 }, { D3 + 1 - KERN_D3 }>
    where
        T: AddAssign + MulAssign,
        [(); KERN_D1 * KERN_D2 * KERN_D3]: ,
        [(); (D1 + 1 - KERN_D1) * (D2 + 1 - KERN_D2) * (D3 + 1 - KERN_D3)]: ,
    {
        let mut res = Tensor3::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let d1 = i % KERN_D1;
            let d2 = (i / KERN_D1) % KERN_D2;
            let d3 = i / (KERN_D1 * KERN_D2);
            res += self.slice(d1, d2, d3).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to preserve shape.
    pub fn convolve_with_pad<const KERN_D1: usize, const KERN_D2: usize, const KERN_D3: usize>(
        self,
        kernel: Tensor3<T, KERN_D1, KERN_D2, KERN_D3>,
    ) -> Tensor3<T, D1, D2, D3>
    where
        T: AddAssign + MulAssign,
        [(); KERN_D1 * KERN_D2 * KERN_D3]: ,
    {
        let mut res = Tensor3::default();
        let kern_d1 = KERN_D1 as isize;
        let kern_d2 = KERN_D2 as isize;
        let kern_d3 = KERN_D3 as isize;
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let d1 = i % kern_d1 - kern_d1 / 2;
            let d2 = (i / kern_d1) % kern_d2 - kern_d2 / 2;
            let d3 = i / (kern_d1 * kern_d2) - kern_d3 / 2;
            res += self.slice_with_pad(d1, d2, d3).scale(val);
        }
        res
    }
}
