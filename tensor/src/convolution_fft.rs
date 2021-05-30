use std::array::IntoIter;

use rustfft::{num_complex::Complex64, FftDirection, FftPlanner};

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

fn to_complex_3<
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const X1: usize,
    const X2: usize,
    const X3: usize,
>(
    tensor: &Tensor3<f64, D1, D2, D3>,
) -> [Complex64; X1 * X2 * X3]
where
    [(); D1 * D2 * D3]: ,
{
    let mut iter = tensor.iter().map(|x| Complex64::new(x, 0.));
    array_init(|i| {
        if i % X1 < D1 && (i / X1) % X2 < D2 {
            iter.next().unwrap_or_default()
        } else {
            Complex64::default()
        }
    })
}

fn apply_fft_2<const X1: usize, const X2: usize>(
    data: &mut [Complex64; X1 * X2],
    fft_planner: &mut FftPlanner<f64>,
    direction: FftDirection,
) {
    let row_fft = fft_planner.plan_fft(X1, direction);
    let col_fft = fft_planner.plan_fft(X2, direction);

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

fn apply_fft_3<const X1: usize, const X2: usize, const X3: usize>(
    data: &mut [Complex64; X1 * X2 * X3],
    fft_planner: &mut FftPlanner<f64>,
    direction: FftDirection,
) {
    let row_fft = fft_planner.plan_fft(X1, direction);
    let col_fft = fft_planner.plan_fft(X2, direction);
    let ax3_fft = fft_planner.plan_fft(X3, direction);

    for depth in 0..X3 {
        for row in 0..X2 {
            row_fft.process(&mut data[(depth * (X1 * X2) + row * X1)..(depth * (X1 * X2) + (row + 1) * X1)]);
        }
        for col in 0..X1 {
            let mut column: [_; X2] = array_init(|i| data[depth * (X1 * X2) + i * X1 + col]);
            col_fft.process(&mut column);
            for (i, val) in IntoIter::new(column).enumerate() {
                data[depth * (X1 * X2) + i * X1 + col] = val;
            }
        }
    }

    for col in 0..X1 {
        for row in 0..X2 {
            let mut ax3: [_; X3] = array_init(|i| data[i * (X1 * X2) + row * X1 + col]);
            ax3_fft.process(&mut ax3);
            for (i, val) in IntoIter::new(ax3).enumerate() {
                data[i * (X1 * X2) + row * X1 + col] = val;
            }
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
        apply_fft_2::<{ fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(
            &mut tensor_data,
            fft_planner,
            FftDirection::Forward,
        );
        apply_fft_2::<{ fft_l(C, KERN_C) }, { fft_l(R, KERN_R) }>(
            &mut kernel_data,
            fft_planner,
            FftDirection::Forward,
        );
        let tensor = Tensor2(tensor_data) * &Tensor2(kernel_data);
        ConvolutionIntermediate { tensor }
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> Tensor3<f64, D1, D2, D3>
where
    [(); D1 * D2 * D3]: ,
{
    pub fn convolve_fft<const KERN_D1: usize, const KERN_D2: usize, const KERN_D3: usize>(
        &self,
        kernel: &Tensor3<f64, KERN_D1, KERN_D2, KERN_D3>,
        fft_planner: &mut FftPlanner<f64>,
    ) -> conv_inter!(
           fft_l(D1, KERN_D1) * fft_l(D2, KERN_D2) * fft_l(D3, KERN_D3),
           Tensor3,
           fft_l(D1, KERN_D1) fft_l(D2, KERN_D2) fft_l(D3, KERN_D3)
       )
    where
        [(); KERN_D1 * KERN_D2 * KERN_D3]: ,
    {
        let mut tensor_data = to_complex_3::<
            D1,
            D2,
            D3,
            { fft_l(D1, KERN_D1) },
            { fft_l(D2, KERN_D2) },
            { fft_l(D3, KERN_D3) },
        >(self);
        let mut kernel_data = to_complex_3::<
            KERN_D1,
            KERN_D2,
            KERN_D3,
            { fft_l(D1, KERN_D1) },
            { fft_l(D2, KERN_D2) },
            { fft_l(D3, KERN_D3) },
        >(&kernel.rev());
        apply_fft_3::<{ fft_l(D1, KERN_D1) }, { fft_l(D2, KERN_D2) }, { fft_l(D3, KERN_D3) }>(
            &mut tensor_data,
            fft_planner,
            FftDirection::Forward,
        );
        apply_fft_3::<{ fft_l(D1, KERN_D1) }, { fft_l(D2, KERN_D2) }, { fft_l(D3, KERN_D3) }>(
            &mut kernel_data,
            fft_planner,
            FftDirection::Forward,
        );
        let tensor = Tensor3(tensor_data) * &Tensor3(kernel_data);
        ConvolutionIntermediate { tensor }
    }
}

impl<const L: usize> conv_inter!(L, Tensor1) {
    pub fn finish(self, fft_planner: &mut FftPlanner<f64>) -> Tensor1<f64, L> {
        let mut data = self.tensor.get_data();
        let fft = fft_planner.plan_fft_inverse(L);
        fft.process(&mut data);
        Tensor1(data.map(|z| z.re / L as f64))
    }
}

impl<const C: usize, const R: usize> conv_inter!(C * R, Tensor2, C R) {
    pub fn finish(self, fft_planner: &mut FftPlanner<f64>) -> Tensor2<f64, C, R>
    where
        [(); C * R]: ,
    {
        let mut data = self.tensor.get_data();
        apply_fft_2::<C, R>(&mut data, fft_planner, FftDirection::Inverse);
        Tensor2(data.map(|z| z.re / (C * R) as f64))
    }
}

impl<const D1: usize, const D2: usize, const D3: usize> conv_inter!(D1 * D2 * D3, Tensor3, D1 D2 D3) {
    pub fn finish(self, fft_planner: &mut FftPlanner<f64>) -> Tensor3<f64, D1, D2, D3>
    where
        [(); D1 * D2 * D3]: ,
    {
        let mut data = self.tensor.get_data();
        apply_fft_3::<D1, D2, D3>(&mut data, fft_planner, FftDirection::Inverse);
        Tensor3(data.map(|z| z.re / (D1 * D2 * D3) as f64))
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
        assert!((naive - &fft).map(f64::abs).sum() < 1e-9);
    }

    #[test]
    fn convolve_fft_tensor2() {
        let distr = rand_distr::Uniform::new(-10., 10.);
        let a = Tensor2::<_, 20, 20>::rand(distr);
        let kernel = Tensor2::<_, 3, 3>::rand(distr);
        let mut fft_planner = FftPlanner::new();
        let naive = a.convolve_with_pad_to::<3, 3, 22, 22>(&kernel);
        let fft = a.convolve_fft(&kernel, &mut fft_planner).finish(&mut fft_planner);
        assert!((naive - &fft).map(f64::abs).sum() < 1e-9);
    }

    #[test]
    /// Sometimes works, sometimes not???
    fn convolve_fft_tensor3() {
        let distr = rand_distr::Uniform::new(-10., 10.);
        let a = Tensor3::<_, 8, 8, 8>::rand(distr);
        let kernel = Tensor3::<_, 3, 3, 3>::rand(distr);
        let mut fft_planner = FftPlanner::new();
        let naive = a.convolve_with_pad_to::<3, 3, 3, 10, 10, 10>(&kernel);
        let fft = a.convolve_fft(&kernel, &mut fft_planner).finish(&mut fft_planner);
        assert!((naive - &fft).map(f64::abs).sum() < 1e-9);
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn before(ben: &mut Bencher) {
        with_larger_stack(|| {
            let a = Tensor3::<f64, 5, 5, 64>::rand(rand_distr::Uniform::new(-1., 1.));
            let b = Tensor3::<f64, 3, 3, 64>::rand(rand_distr::Uniform::new(-1., 1.));
            ben.iter(|| a.convolve_with_pad_to::<3, 3, 64, 7, 7, 127>(&b));
        });
    }

    #[bench]
    fn after(ben: &mut Bencher) {
        with_larger_stack(|| {
            let a = Tensor3::<f64, 5, 5, 64>::rand(rand_distr::Uniform::new(-1., 1.));
            let b = Tensor3::<f64, 3, 3, 64>::rand(rand_distr::Uniform::new(-1., 1.));
            let mut fft_planner = FftPlanner::new();
            a.convolve_fft(&b, &mut fft_planner).finish(&mut fft_planner);
            ben.iter(|| a.convolve_fft(&b, &mut fft_planner).finish(&mut fft_planner));
        });
    }
}
