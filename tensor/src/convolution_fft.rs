use std::array::IntoIter;

use rustfft::{num_complex::Complex64, FftPlanner};

use super::*;
use crate::array_init::array_init;

// Macro to simplify writing the type of ConvolutionIntermediate
macro_rules! conv_inter {
    ($size:expr, Tensor1) => {
        ConvolutionIntermediate<{$size}, Tensor1<Complex64, {$size}>>
    };
    ($size:expr, $tensor:ident, $($len:expr)*) => {
        ConvolutionIntermediate<{$size}, $tensor<Complex64, $({$len},)*>>
    };
}

pub struct ConvolutionIntermediate<const X: usize, T: Tensor<Complex64, X>> {
    tensor: T,
}

#[allow(non_snake_case)]
pub const fn fft_l(L: usize, KERN_L: usize) -> usize {
    L + KERN_L - 1
}

fn to_complex_1<const L: usize, const X: usize>(tensor: &Tensor1<f64, L>) -> [Complex64; X] {
    let mut iter = tensor.iter().map(|x| Complex64::new(x, 0.));
    [(); X].map(|()| iter.next().unwrap_or_default())
}

fn to_complex_2<const C: usize, const R: usize, const X1: usize, const X2: usize>(
    tensor: &Tensor2<f64, C, R>,
) -> [Complex64; X1 * X2]
where
    [(); C * R]: ,
{
    let mut iter = tensor.iter().map(|x| Complex64::new(x, 0.));
    array_init(|i| {
        if i % X1 < C {
            iter.next().unwrap_or_default()
        } else {
            Complex64::default()
        }
    })
}

fn apply_fft_2<const X1: usize, const X2: usize>(
    data: &mut [Complex64; X1 * X2],
    fft_planner: &mut FftPlanner<f64>,
) {
    let row_fft = fft_planner.plan_fft_forward(X1);
    let col_fft = fft_planner.plan_fft_forward(X2);

    for row in 0..X2 {
        row_fft.process(&mut data[(row * X1)..((row + 1) * X1)]);
    }
    for col in 0..X1 {
        let mut column: [_; X2] = array_init(|i| data[i * X1 + col]);
        col_fft.process(&mut column);
        for (i, val) in IntoIter::new(column).enumerate() {
            data[i * X1 + col] = val;
        }
    }
}

impl<const L: usize> Tensor1<f64, L> {
    pub fn convolve_fft<const KERN_L: usize>(
        &self,
        kernel: &Tensor1<f64, KERN_L>,
        fft_planner: &mut FftPlanner<f64>,
    ) -> conv_inter!(fft_l(L, KERN_L), Tensor1) {
        let mut tensor_data = to_complex_1(self);
        let mut kernel_data = to_complex_1(&kernel.rev());
        let fft = fft_planner.plan_fft_forward(fft_l(L, KERN_L));
        fft.process(&mut tensor_data);
        fft.process(&mut kernel_data);
        let tensor = Tensor1(tensor_data) * &Tensor1(kernel_data);
        ConvolutionIntermediate { tensor }
    }
}

impl<const C: usize, const R: usize> Tensor2<f64, C, R>
where
    [(); C * R]: ,
{
    pub fn convolve_fft<const KERN_C: usize, const KERN_R: usize>(
        &self,
        kernel: &Tensor2<f64, KERN_C, KERN_R>,
        fft_planner: &mut FftPlanner<f64>,
    ) -> conv_inter!(fft_l(C, KERN_C) * fft_l(R, KERN_R), Tensor2, fft_l(C, KERN_C) fft_l(R, KERN_R))
    where
        [(); KERN_C * KERN_R]: ,
    {
        let mut tensor_data = to_complex_2::<C, R, { fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(self);
        let mut kernel_data =
            to_complex_2::<KERN_C, KERN_R, { fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(&kernel.rev());
        apply_fft_2::<{ fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(&mut tensor_data, fft_planner);
        apply_fft_2::<{ fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(&mut kernel_data, fft_planner);
        let tensor = Tensor2(tensor_data) * &Tensor2(kernel_data);
        ConvolutionIntermediate { tensor }
    }
}

impl<const L: usize> conv_inter!(L, Tensor1) {
    pub fn finish(self, fft_planner: &mut FftPlanner<f64>) -> Tensor1<f64, L> {
        let mut data = self.tensor.get_data();
        let fft = fft_planner.plan_fft_inverse(L);
        fft.process(&mut data);
        Tensor1(data.map(|z| z.norm() / L as f64))
    }
}

impl<const C: usize, const R: usize> conv_inter!(C * R, Tensor2, C R) {
    pub fn finish(self, fft_planner: &mut FftPlanner<f64>) -> Tensor2<f64, C, R>
    where
        [(); C * R]: ,
    {
        let mut data = self.tensor.get_data();
        let fft_row = fft_planner.plan_fft_inverse(C);
        let fft_col = fft_planner.plan_fft_inverse(R);
        for r in 0..R {
            fft_row.process(&mut data[(r * C)..((r + 1) * C)])
        }
        for c in 0..C {
            let mut range = 0..R;
            let mut column = [(); R].map(|()| {
                let r = range.next().unwrap();
                data[r * C + c]
            });
            fft_col.process(&mut column);
            for (r, val) in IntoIter::new(column).enumerate() {
                data[r * C + c] = val;
            }
        }
        Tensor2(data.map(|z| z.norm() / (C * R) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolve_fft_tensor1() {
        let distr = rand_distr::Uniform::new(-10., 10.);
        let a = Tensor1::<_, 100>::rand(distr);
        let kernel = Tensor1::<_, 5>::rand(distr);
        let mut fft_planner = FftPlanner::new();
        let naive = a.convolve_with_pad_to::<5, 104>(&kernel);
        let fft = a.convolve_fft(&kernel, &mut fft_planner).finish(&mut fft_planner);
        assert!((naive - &fft).sum() < 1e-9);
    }

    #[test]
    fn convolve_fft_tensor2() {
        let distr = rand_distr::Uniform::new(-10., 10.);
        let a = Tensor2::<_, 20, 20>::rand(distr);
        let kernel = Tensor2::<_, 3, 3>::rand(distr);
        let mut fft_planner = FftPlanner::new();
        let naive = a.convolve_with_pad_to::<3, 3, 22, 22>(&kernel);
        let fft = a.convolve_fft(&kernel, &mut fft_planner).finish(&mut fft_planner);
        assert!((naive - &fft).sum() < 1e-9);
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn before(ben: &mut Bencher) {
        let a = Tensor2::<f64, 50, 50>::rand(rand_distr::Uniform::new(-1., 1.));
        let b = Tensor2::<f64, 20, 10>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.convolve_with_pad_to::<20, 10, 54, 54>(&b));
    }

    #[bench]
    fn after(ben: &mut Bencher) {
        let a = Tensor2::<f64, 50, 50>::rand(rand_distr::Uniform::new(-1., 1.));
        let b = Tensor2::<f64, 20, 10>::rand(rand_distr::Uniform::new(-1., 1.));
        let mut fft_planner = FftPlanner::new();
        a.convolve_fft(&b, &mut fft_planner).finish(&mut fft_planner);
        ben.iter(|| a.convolve_fft(&b, &mut fft_planner).finish(&mut fft_planner));
    }
}
