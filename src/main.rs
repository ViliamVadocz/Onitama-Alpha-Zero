#![feature(const_generics, const_evaluatable_checked, array_map)]
#![feature(thread_spawn_unchecked)]
#![allow(incomplete_features)]
#![feature(test)]
extern crate test;

mod network;
mod alpha_zero;
mod convert;

use network::Network;
use std::thread;
use tensor::*;

fn run() {
    let test_net = Network::init();
    let (vec, eval) = test_net.feed_forward(Tensor3::rand(rand_distr::Uniform::new(0., 1.)));
    println!("{} {}", vec, eval)
}

fn main() {
    thread::Builder::new()
        .stack_size(1024 * 1024 * 1024 * 32)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
