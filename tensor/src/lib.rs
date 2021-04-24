#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

mod tensor;

#[cfg(feature = "rand")]
pub use rand::distributions as rand_distr;
pub use tensor::elementwise::ElementWiseTensor;
#[cfg(feature = "rand")]
pub use tensor::random::RandomTensor;
pub use tensor::{reshape, Tensor, Tensor1, Tensor2, Tensor3};
