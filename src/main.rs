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

fn run() {
    let mut network = Network::init();
    loop {
        train_network(&mut network);
        // TODO save network to file.
    }
}

// TODO
// - Saving to file
// - Back prop
// - Logging
// - Testing
// - Optimizing (single thread, no gpu)
// - Loading from file
// - Litama support
// - Optimizing (multithreading)
// - Optimizing (gpu)
// - Playing around with hyperparameters
// - README

fn main() {
    thread::Builder::new()
        .stack_size(1024 * 1024 * 1024 * 32)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
