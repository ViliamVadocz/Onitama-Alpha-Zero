use std::{
    array::IntoIter,
    cmp::PartialOrd,
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign},
};

use super::*;

/// Rectilinear unit.
pub fn relu(x: f64) -> f64 {
    f64::max(0., x)
}

/// Derivative of relu
pub fn d_relu(x: f64) -> f64 {
    if x >= 0. {
        1.
    } else {
        0.
    }
}

/// Numerically stable sigmoid function.
pub fn sig(x: f64) -> f64 {
    if x >= 0. {
        let z = (-x).exp();
        1. / (1. + z)
    } else {
        let z = x.exp();
        z / (1. + z)
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor3<T, D1, D2, D3>
where
    T: Copy + Default + Debug,
    [(); D1 * D2 * D3]: ,
{
    pub fn convolution_pass<const N: usize, const K_D1: usize, const K_D2: usize>(
        self,
        kernels: &[Tensor3<T, K_D1, K_D2, D3>; N],
        biases: &Tensor3<T, D1, D2, N>,
    ) -> Tensor3<T, D1, D2, N>
    where
        T: AddAssign + MulAssign + Add<Output = T> + PartialOrd,
        [(); D1 * D2 * D3]: ,
        [(); K_D1 * K_D2 * D3]: ,
        [(); D1 * D2 * N]: ,
    {
        let conv_out = kernels.map(|kernel| self.convolve_with_pad(kernel).get_data());
        let mut data = [T::default(); D1 * D2 * N];
        for (elem, var) in data.iter_mut().zip(IntoIter::new(conv_out).flatten()) {
            *elem = var;
        }
        Tensor3::new(data) + biases
    }
}

impl<T, const L: usize> Tensor1<T, L>
where
    T: Copy + Default + Debug,
{
    /// Essentially matrix multiplication.
    pub fn fully_connected_pass<const N: usize>(
        self,
        weights: &[Tensor1<T, L>; N],
        biases: &Tensor1<T, N>,
    ) -> Tensor1<T, N>
    where
        T: Add<Output = T> + Mul<Output = T> + Sum<T> + PartialOrd,
        [(); L * N]: ,
    {
        let mut data = [T::default(); N];
        for (elem, row) in data.iter_mut().zip(weights.iter()) {
            *elem = (self * row).sum();
        }
        Tensor1(data) + biases
    }

    /// Split the tensor by separating the last elem.
    pub fn split_last(&self) -> (Tensor1<T, { L - 1 }>, T)
    where
        [(); L - 1]: ,
    {
        let mut data = [T::default(); L - 1];
        data.iter_mut()
            .zip(self.0[0..(L - 1)].iter())
            .for_each(|(elem, &val)| *elem = val);
        (Tensor1(data), self.0[L - 1])
    }
}

impl<const L: usize> Tensor1<f64, L> {
    /// Softmax which takes into account numerical stability.
    pub fn softmax(self) -> Self {
        let b = self.0.iter().cloned().fold(f64::NAN, f64::max);
        let exp = self.map(|x| f64::exp(x - b));
        exp.scale(1. / exp.sum())
    }
}

use std::thread;
pub fn with_larger_stack<'a, T, F>(f: F)
where
    T: 'a + Send,
    F: FnOnce() -> T,
    F: 'a + Send,
{
    unsafe {
        thread::Builder::new()
            .stack_size(1024 * 1024 * 1024 * 32)
            .spawn_unchecked(f)
            .unwrap()
            .join()
            .unwrap();
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn conv_pass(ben: &mut Bencher) {
        let distr = rand_distr::Uniform::<f64>::new(-1., 1.);
        with_larger_stack(move || {
            let a = Tensor3::<_, 5, 5, 64>::rand(distr);
            let kernels = [(); 64].map(|()| Tensor3::<_, 3, 3, 64>::rand(distr));
            let biases = Tensor3::rand(distr);
            ben.iter(|| a.convolution_pass(&kernels, &biases));
        })
    }

    #[bench]
    fn full_matmul_pass(ben: &mut Bencher) {
        let distr = rand_distr::Uniform::<f64>::new(-1., 1.);
        with_larger_stack(move || {
            let a = Tensor1::<_, 2000>::rand(distr);
            let weights = [(); 1000].map(|()| Tensor1::rand(distr));
            let biases = Tensor1::rand(distr);
            ben.iter(|| a.fully_connected_pass(&weights, &biases));
        });
    }
}
