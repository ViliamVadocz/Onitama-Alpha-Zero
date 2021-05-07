#![feature(
    const_generics,
    const_evaluatable_checked,
    array_map,
    thread_spawn_unchecked,
    entry_insert
)]
#![allow(incomplete_features)]
#![feature(test)]
extern crate test;

mod alpha_zero;
mod convert;
mod mcts;
mod network;
mod rand_game;

use alpha_zero::train_network;
use network::Network;
use std::thread;
use std::env;


fn run() {
    // Look at the second argument to see if we should load.
    let mut args = env::args();
    let _ = args.next(); // First arg will just be the name of the binary.
    let second_arg = args.next()
        .map(|x| x.parse::<u32>());

    let mut i = 0;
    let mut network = match second_arg {
        Some(Ok(load)) if load > 0 => {
            i = load + 1;
            Network::load(&format!("iters/alphazero_{:>0}.data", load))
        },
        _ => Network::init()
    };

    // Main training loop.
    // We save after each improvement.
    loop {
        train_network(&mut network);
        network.save(&format!("iters/alphazero_{:0>8}.data", i));
        i += 1;
    }
}

// TODO
// - Back prop
// - Logging
// - Testing
// - Optimizing (single thread, no gpu)
// - Litama support
// - Optimizing (multithreading)
// - Optimizing (gpu)
// - Playing around with hyper-parameters
// - README

fn main() {
    thread::Builder::new()
        .stack_size(1024 * 1024 * 1024 * 32)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
