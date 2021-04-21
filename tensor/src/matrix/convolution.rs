use super::*;

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    /// Perform naive convolution without padding.
    pub fn convolve<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Matrix<T, KERN_R, KERN_C>,
    ) -> Matrix<T, { R + 1 - KERN_R }, { C + 1 - KERN_C }>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + 1 - KERN_R) * (C + 1 - KERN_C)]: ,
    {
        let mut res = Matrix::default();
        for (i, &elem) in kernel.0.iter().enumerate() {
            let r = i / KERN_C;
            let c = i % KERN_C;
            res += self.slice(r, c).scale(elem)
        }
        res
    }

    /// Perform naive convolution with padding.
    /// Padded with zeros.
    pub fn convolve_with_pad<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Matrix<T, KERN_R, KERN_C>,
    ) -> Matrix<T, R, C>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + KERN_R - 1) * (C + KERN_C - 1)]: ,
    {
        let mut res = Matrix::default();
        let kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &elem) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let r = i / kern_c - kern_r / 2;
            let c = i % kern_c - kern_c / 2;
            res += self.slice_with_pad(r, c).scale(elem)
        }
        res
    }
}
