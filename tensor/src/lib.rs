#![feature(const_generics, const_evaluatable_checked, thread_spawn_unchecked)]
#![allow(incomplete_features)]
#![feature(test)]
extern crate test;

use std::fmt::{Debug, Display};

mod array_init;
mod convolution;
mod convolution_fft;
mod default;
mod display;
mod elementwise;
mod ml;
mod shape;
mod slice;
mod tensor;

#[cfg(feature = "rand")]
mod random;

pub use rustfft::FftPlanner;
#[cfg(feature = "rand")]
pub use {crate::random::RandomTensor, rand::distributions as rand_distr};

pub use crate::{
    elementwise::ElementWiseTensor,
    ml::{d_relu, relu, sig, with_larger_stack},
    tensor::{Matrix, Tensor, Tensor1, Tensor2, Tensor3, Vector},
};
