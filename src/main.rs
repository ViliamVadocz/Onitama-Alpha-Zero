#![feature(const_generics, const_evaluatable_checked, array_map)]
#![allow(incomplete_features)]

use tensor::*;

struct Network {
    // Padded convolution layers.
    l1_kernels: [Tensor3<f64, 3, 3, 8>; 64],
    l1_biases: Tensor3<f64, 5, 5, 64>,
    l2_kernels: [Tensor3<f64, 3, 3, 64>; 64],
    l2_biases: Tensor3<f64, 5, 5, 64>,
    l3_kernels: [Tensor3<f64, 3, 3, 64>; 64],
    l3_biases: Tensor3<f64, 5, 5, 64>,
    l4_kernels: [Tensor3<f64, 3, 3, 64>; 64],
    l4_biases: Tensor3<f64, 5, 5, 64>,
    // Fully connected layers.
    l5_weights: Tensor2<f64, 1600, 800>,
    l5_biases: Tensor1<f64, 800>,
    l6_weights: Tensor2<f64, 800, 625>,
    l6_biases: Tensor1<f64, 626>,
}

impl Network {
    fn init() -> Network {
        let distr = rand_distr::Standard;
        Network {
            // Padded convolution layers.
            l1_kernels: [(); 64].map(|()| Tensor3::rand(distr)),
            l1_biases: Tensor3::rand(distr),
            l2_kernels: [(); 64].map(|()| Tensor3::rand(distr)),
            l2_biases: Tensor3::rand(distr),
            l3_kernels: [(); 64].map(|()| Tensor3::rand(distr)),
            l3_biases: Tensor3::rand(distr),
            l4_kernels: [(); 64].map(|()| Tensor3::rand(distr)),
            l4_biases: Tensor3::rand(distr),
            // Fully connected layers.
            l5_weights: Tensor2::rand(distr),
            l5_biases: Tensor1::rand(distr),
            l6_weights: Tensor2::rand(distr),
            l6_biases: Tensor1::rand(distr),
        }
    }

    fn feed_forward(&self, input: Tensor3<f64, 5, 5, 8>) -> Tensor1<f64, 626> {
        input
            .convolution_pass(self.l1_kernels, self.l1_biases) 
            .convolution_pass(self.l2_kernels, self.l2_biases)
            .convolution_pass(self.l3_kernels, self.l3_biases)
            .convolution_pass(self.l4_kernels, self.l4_biases)
            .reshape::<Tensor1<_, 1600>>()
            .fully_connected_pass(self.l5_weights, self.l5_biases)
            .fully_connected_pass(self.l6_weights, self.l6_biases)
    }
}

fn main() {}
