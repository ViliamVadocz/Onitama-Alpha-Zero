use super::*;
use rand::{distributions::Distribution, thread_rng};

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    /// Initialize a random matrix.
    pub fn rand<D: Distribution<T>>(distr: D) -> Self {
        let mut rng = thread_rng();
        let mut data = [T::default(); R * C];
        for elem in data.iter_mut() {
            *elem = distr.sample(&mut rng);
        }
        Matrix(data)
    }
}
