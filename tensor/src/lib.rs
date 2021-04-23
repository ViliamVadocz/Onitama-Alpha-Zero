#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

mod tensor;

pub use tensor::{Tensor, Tensor1, Tensor2, Tensor3, reshape};
pub use tensor::elementwise::ElementWiseTensor;
#[cfg(feature = "rand")]
pub use rand::distributions as rand_distr;
#[cfg(feature = "rand")]
pub use tensor::random::RandomTensor;
