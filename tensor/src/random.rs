use super::*;
use rand::{distributions::Distribution, thread_rng};

pub trait RandomTensor<T, const X: usize>: Tensor<T, X>
where
    T: Default + Copy + Debug,
    [(); X]: ,
{
    fn rand<D: Distribution<T>>(distr: D) -> Self {
        let mut rng = thread_rng();
        let mut data = [T::default(); X];
        for elem in data.iter_mut() {
            *elem = distr.sample(&mut rng);
        }
        Self::new(data)
    }
}

impl<T, const L: usize> RandomTensor<T, L> for Tensor1<T, L> where T: Default + Copy + Debug {}

impl<T, const R: usize, const C: usize> RandomTensor<T, { R * C }> for Tensor2<T, R, C> where
    T: Default + Copy + Debug
{
}

impl<T, const D1: usize, const D2: usize, const D3: usize> RandomTensor<T, { D1 * D2 * D3 }>
    for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
{
}
