use tensor::*;

pub struct Network {
    // Padded convolution layers.
    l1_kernels: Box<[Tensor3<f64, 3, 3, 8>; 64]>,
    l1_biases: Box<Tensor3<f64, 5, 5, 64>>,
    l2_kernels: Box<[Tensor3<f64, 3, 3, 64>; 64]>,
    l2_biases: Box<Tensor3<f64, 5, 5, 64>>,
    l3_kernels: Box<[Tensor3<f64, 3, 3, 64>; 64]>,
    l3_biases: Box<Tensor3<f64, 5, 5, 64>>,
    l4_kernels: Box<[Tensor3<f64, 3, 3, 64>; 64]>,
    l4_biases: Box<Tensor3<f64, 5, 5, 64>>,
    // Fully connected layers.
    l5_weights: Box<[Tensor1<f64, 1600>; 800]>,
    l5_biases: Box<Tensor1<f64, 800>>,
    l6_weights: Box<[Tensor1<f64, 800>; 626]>,
    l6_biases: Box<Tensor1<f64, 626>>,
    // 626 outputs
    // 625 for each from-to move combination
    // 1 output for prediction
}

impl Network {
    pub fn init() -> Network {
        let distr = rand_distr::Standard;
        Network {
            // Padded convolution layers.
            l1_kernels: Box::new([(); 64].map(|()| Tensor3::rand(distr))),
            l1_biases: Box::new(Tensor3::rand(distr)),
            l2_kernels: Box::new([(); 64].map(|()| Tensor3::rand(distr))),
            l2_biases: Box::new(Tensor3::rand(distr)),
            l3_kernels: Box::new([(); 64].map(|()| Tensor3::rand(distr))),
            l3_biases: Box::new(Tensor3::rand(distr)),
            l4_kernels: Box::new([(); 64].map(|()| Tensor3::rand(distr))),
            l4_biases: Box::new(Tensor3::rand(distr)),
            // Fully connected layers.
            l5_weights: Box::new([(); 800].map(|()| Tensor1::rand(distr))),
            l5_biases: Box::new(Tensor1::rand(distr)),
            l6_weights: Box::new([(); 626].map(|()| Tensor1::rand(distr))),
            l6_biases: Box::new(Tensor1::rand(distr)),
        }
    }

    pub fn feed_forward(&self, input: Tensor3<f64, 5, 5, 8>) -> Tensor1<f64, 626> {
        input
            .convolution_pass(&self.l1_kernels, &self.l1_biases)
            .convolution_pass(&self.l2_kernels, &self.l2_biases)
            .convolution_pass(&self.l3_kernels, &self.l3_biases)
            .convolution_pass(&self.l4_kernels, &self.l4_biases)
            .reshape::<Tensor1<_, 1600>>()
            .fully_connected_pass(&self.l5_weights, &self.l5_biases)
            .fully_connected_pass(&self.l6_weights, &self.l6_biases)
    }

    pub fn back_prop(&mut self, training: Tensor1<f64, 626>) {
        unimplemented!()
    }
}
