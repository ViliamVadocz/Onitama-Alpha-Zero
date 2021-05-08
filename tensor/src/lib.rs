#![feature(const_generics, const_evaluatable_checked, array_map)]
#![allow(incomplete_features)]
#![feature(test)]
extern crate test;

use std::fmt::{Debug, Display};

mod convolution;
mod default;
mod display;
mod elementwise;
mod ml;
mod slice;
mod tensor;

#[cfg(feature = "rand")]
mod random;

pub use crate::{
    elementwise::ElementWiseTensor,
    ml::{relu, d_relu, sig},
    tensor::{Matrix, Tensor, Tensor1, Tensor2, Tensor3, Vector},
};

#[cfg(feature = "rand")]
pub use {crate::random::RandomTensor, rand::distributions as rand_distr};
