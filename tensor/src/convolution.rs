use std::ops::{AddAssign, MulAssign};

use super::*;

impl<T, const L: usize> Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    /// Naive convolution.
    pub fn convolve<const KERN_L: usize>(&self, kernel: &Tensor1<T, KERN_L>) -> Tensor1<T, { L + 1 - KERN_L }>
    where
        T: AddAssign + MulAssign,
    {
        let mut res = Tensor1::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            res += &self.slice(i).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to **preserve** shape.
    pub fn convolve_with_pad<const KERN_L: usize>(&self, kernel: &Tensor1<T, KERN_L>) -> Tensor1<T, L>
    where
        T: AddAssign + MulAssign,
    {
        let mut res = Tensor1::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            res += &self.slice_with_pad(i - KERN_L as isize / 2).scale(val)
        }
        res
    }

    /// Naive convolution with default padding to **change** shape.
    pub fn convolve_with_pad_to<const KERN_L: usize, const OUT_L: usize>(
        &self,
        kernel: &Tensor1<T, KERN_L>,
    ) -> Tensor1<T, OUT_L>
    where
        T: AddAssign + MulAssign,
    {
        let pad = (OUT_L + KERN_L - L - 1) as isize;
        let mut res = Tensor1::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            res += &self.slice_with_pad(i - pad / 2).scale(val)
        }
        res
    }
}

impl<T, const C: usize, const R: usize> Tensor2<T, C, R>
where
    T: Default + Copy + Debug,
    [(); C * R]: ,
{
    /// Naive convolution.
    pub fn convolve<const KERN_C: usize, const KERN_R: usize>(
        &self,
        kernel: &Tensor2<T, KERN_C, KERN_R>,
    ) -> Tensor2<T, { C + 1 - KERN_C }, { R + 1 - KERN_R }>
    where
        T: AddAssign + MulAssign,
        [(); KERN_C * KERN_R]: ,
        [(); (C + 1 - KERN_C) * (R + 1 - KERN_R)]: ,
    {
        let mut res = Tensor2::default();
        for (i, &val) in kernel.0.iter().enumerate() {
            let c = i % KERN_C;
            let r = i / KERN_C;
            res += &self.slice(c, r).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to **preserve** shape.
    pub fn convolve_with_pad<const KERN_C: usize, const KERN_R: usize>(
        &self,
        kernel: &Tensor2<T, KERN_C, KERN_R>,
    ) -> Tensor2<T, C, R>
    where
        T: AddAssign + MulAssign,
        [(); KERN_C * KERN_R]: ,
    {
        let mut res = Tensor2::default();
        let kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let c = i % kern_c - kern_c / 2;
            let r = i / kern_c - kern_r / 2;
            res += &self.slice_with_pad(c, r).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to **change** shape.
    pub fn convolve_with_pad_to<
        const KERN_C: usize,
        const KERN_R: usize,
        const OUT_C: usize,
        const OUT_R: usize,
    >(
        &self,
        kernel: &Tensor2<T, KERN_C, KERN_R>,
    ) -> Tensor2<T, OUT_C, OUT_R>
    where
        T: AddAssign + MulAssign,
        [(); KERN_C * KERN_R]: ,
        [(); OUT_C * OUT_R]: ,
    {
        let pad_c = (OUT_C + KERN_C - C - 1) as isize;
        let pad_r = (OUT_R + KERN_R - R - 1) as isize;
        let mut res = Tensor2::default();
        let _kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let c = i % kern_c - pad_c / 2;
            let r = i / kern_c - pad_r / 2;
            res += &self.slice_with_pad(c, r).scale(val);
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
        &self,
        kernel: &Tensor3<T, KERN_D1, KERN_D2, KERN_D3>,
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
            res += &self.slice(d1, d2, d3).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to **preserve** shape.
    pub fn convolve_with_pad<const KERN_D1: usize, const KERN_D2: usize, const KERN_D3: usize>(
        &self,
        kernel: &Tensor3<T, KERN_D1, KERN_D2, KERN_D3>,
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
            res += &self.slice_with_pad(d1, d2, d3).scale(val);
        }
        res
    }

    /// Naive convolution with default padding to **change** shape.
    pub fn convolve_with_pad_to<
        const KERN_D1: usize,
        const KERN_D2: usize,
        const KERN_D3: usize,
        const OUT_D1: usize,
        const OUT_D2: usize,
        const OUT_D3: usize,
    >(
        &self,
        kernel: &Tensor3<T, KERN_D1, KERN_D2, KERN_D3>,
    ) -> Tensor3<T, OUT_D1, OUT_D2, OUT_D3>
    where
        T: AddAssign + MulAssign,
        [(); KERN_D1 * KERN_D2 * KERN_D3]: ,
        [(); OUT_D1 * OUT_D2 * OUT_D3]: ,
    {
        let pad_d1 = (OUT_D1 + KERN_D1 - D1 - 1) as isize;
        let pad_d2 = (OUT_D2 + KERN_D2 - D2 - 1) as isize;
        let pad_d3 = (OUT_D3 + KERN_D3 - D3 - 1) as isize;
        let mut res = Tensor3::default();
        let kern_d1 = KERN_D1 as isize;
        let kern_d2 = KERN_D2 as isize;
        let _kern_d3 = KERN_D3 as isize;
        for (i, &val) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let d1 = i % kern_d1 - pad_d1 / 2;
            let d2 = (i / kern_d1) % kern_d2 - pad_d2 / 2;
            let d3 = i / (kern_d1 * kern_d2) - pad_d3 / 2;
            res += &self.slice_with_pad(d1, d2, d3).scale(val);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_tensor1() {
        let a = Vector::new([1, 2, 3, 4, 5, 6, 7]);
        let kernel = Vector::new([1, 2, 3]);
        assert_eq!(a.convolve(&kernel), Vector::new([14, 20, 26, 32, 38]));
        assert_eq!(
            a.convolve_with_pad(&kernel),
            Vector::new([8, 14, 20, 26, 32, 38, 20])
        );
        assert_eq!(
            a.convolve_with_pad_to(&kernel),
            Vector::new([3, 8, 14, 20, 26, 32, 38, 20, 7])
        );

        let b = Vector::new([1, 2, 3, 4, 5, 6]);
        let kernel = Vector::new([1, 2, 3, 4]);
        assert_eq!(b.convolve(&kernel), Vector::new([30, 40, 50]));
        assert_eq!(
            b.convolve_with_pad(&kernel),
            Vector::new([11, 20, 30, 40, 50, 32])
        );
        assert_eq!(b.convolve_with_pad_to(&kernel), Vector::new([20, 30, 40, 50, 32]))
    }

    #[test]
    fn convolution_tensor2() {
        #[rustfmt::skip]
        let a = Matrix::<_, 3, 3>::new([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);
        #[rustfmt::skip]
        let kernel = Matrix::<_, 2, 2>::new([
            1, 2,
            3, 4,
        ]);
        assert_eq!(a.convolve(&kernel), Matrix::new([37, 47, 67, 77]));
        assert_eq!(
            a.convolve_with_pad(&kernel),
            Matrix::new([4, 11, 18, 18, 37, 47, 36, 67, 77])
        );
        assert_eq!(
            a.convolve_with_pad_to(&kernel),
            Matrix::<_, 4, 2>::new([18, 37, 47, 21, 36, 67, 77, 33])
        );

        #[rustfmt::skip]
        let b = Matrix::<_, 3, 4>::new([
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
        ]);
        #[rustfmt::skip]
        let kernel = Matrix::<_, 2, 3>::new([
            1, 2,
            3, 4,
            5, 6,
        ]);
        assert_eq!(b.convolve(&kernel), Matrix::new([120, 141, 183, 204]));
        #[rustfmt::skip]
        assert_eq!(b.convolve_with_pad(&kernel), Matrix::new([
            28, 61, 79,
            60, 120, 141,
            96, 183, 204,
            54, 97, 107,
        ]));
        assert_eq!(
            b.convolve_with_pad_to(&kernel),
            Matrix::<_, 3, 2>::new([120, 141, 66, 183, 204, 93])
        );
    }

    #[test]
    fn convolution_tensor3() {
        #[rustfmt::skip]
        let a = Tensor3::<_, 3, 2, 2>::new([
            1, 2, 3,
            4, 5, 6,

            7, 8, 9,
            10, 11, 12,
        ]);
        #[rustfmt::skip]
        let kernel = Tensor3::<_, 2, 2, 2>::new([
            1, 2,
            3, 4,

            5, 6,
            7, 8,
        ]);
        assert_eq!(a.convolve(&kernel), Tensor3::new([278, 314]));
        #[rustfmt::skip]
        assert_eq!(a.convolve_with_pad(&kernel), Tensor3::new([
            8, 23, 38,
            38, 85, 111,

            60, 124, 146,
            140, 278, 314
        ]));
        assert_eq!(
            a.convolve_with_pad_to(&kernel),
            Tensor3::<_, 2, 1, 2>::new([278, 314, 97, 107])
        )
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn convolve(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        let b = Tensor3::<f64, 3, 3, 3>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.convolve(&b));
    }

    #[bench]
    fn convolve_with_pad(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        let b = Tensor3::<f64, 3, 3, 3>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.convolve_with_pad(&b));
    }
}
