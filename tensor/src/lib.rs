#![feature(const_generics, const_evaluatable_checked, array_map, thread_spawn_unchecked)]
#![allow(incomplete_features)]
#![feature(test)]
extern crate test;

use std::fmt::{Debug, Display};

mod array_init;
mod convolution;
mod default;
mod display;
mod elementwise;
mod ml;
mod slice;
mod tensor;

#[cfg(feature = "rand")]
mod random;

#[cfg(feature = "rand")]
pub use {crate::random::RandomTensor, rand::distributions as rand_distr};

pub use crate::{
    elementwise::ElementWiseTensor,
    ml::{d_relu, relu, sig, with_larger_stack},
    tensor::{Matrix, Tensor, Tensor1, Tensor2, Tensor3, Vector},
};
