#![feature(const_generics, const_evaluatable_checked, array_map)]
#![allow(incomplete_features)]

mod network;

use tensor::*;
use network::Network;

fn main() {
    let test_net = Network::init();
    // THIS OVERFLOWS THE STACK!
    let out = test_net.feed_forward(Tensor3::rand(rand_distr::Uniform::new(0., 1.)));
    println!("{}", out);
}
