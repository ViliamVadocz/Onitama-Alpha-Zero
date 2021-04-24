#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use std::fmt::{Debug, Display};

mod convolution;
mod default;
mod display;
mod elementwise;
mod slice;
mod tensor;

#[cfg(feature = "rand")]
mod random;

pub use crate::{
    elementwise::ElementWiseTensor,
    tensor::{reshape, Matrix, Tensor, Tensor1, Tensor2, Tensor3, Vector},
};

#[cfg(feature = "rand")]
pub use {crate::random::RandomTensor, rand::distributions as rand_distr};
