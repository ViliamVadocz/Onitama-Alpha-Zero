use rand::{distributions::Distribution, thread_rng};

use super::*;

pub trait RandomTensor<T, const X: usize>: Tensor<T, X>
where
    T: Default + Copy + Debug,
    [(); X]: ,
{
    fn rand<D: Distribution<T>>(distr: D) -> Self {
        let mut rng = thread_rng();
        Self::new([(); X].map(|()| distr.sample(&mut rng)))
    }
}

impl<T, const L: usize> RandomTensor<T, L> for Tensor1<T, L> where T: Default + Copy + Debug {}

impl<T, const C: usize, const R: usize> RandomTensor<T, { C * R }> for Tensor2<T, C, R> where
    T: Default + Copy + Debug
{
}

impl<T, const D1: usize, const D2: usize, const D3: usize> RandomTensor<T, { D1 * D2 * D3 }>
    for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
{
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn random(ben: &mut Bencher) {
        ben.iter(|| Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.)));
    }
}
