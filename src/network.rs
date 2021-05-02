use tensor::*;

const LEARNING_RATE: f64 = 0.001;

pub struct Network {
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
    l5_weights: [Tensor1<f64, 1600>; 800],
    l5_biases: Tensor1<f64, 800>,
    l6_weights: [Tensor1<f64, 800>; 626],
    l6_biases: Tensor1<f64, 626>,
    // 626 outputs
    // 625 for each from-to move combination
    // 1 output for board evaluation
}

impl Network {
    pub fn init() -> Network {
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
            l5_weights: [(); 800].map(|()| Tensor1::rand(distr)),
            l5_biases: Tensor1::rand(distr),
            l6_weights: [(); 626].map(|()| Tensor1::rand(distr)),
            l6_biases: Tensor1::rand(distr),
        }
    }

    pub fn feed_forward(&self, input: Tensor3<f64, 5, 5, 8>) -> (Tensor1<f64, 625>, f64) {
        let (vec, board_eval) = input
            .convolution_pass(&self.l1_kernels, &self.l1_biases)
            .map(relu)
            .convolution_pass(&self.l2_kernels, &self.l2_biases)
            .map(relu)
            .convolution_pass(&self.l3_kernels, &self.l3_biases)
            .map(relu)
            .convolution_pass(&self.l4_kernels, &self.l4_biases)
            .map(relu)
            .reshape::<Tensor1<_, 1600>>()
            .fully_connected_pass(&self.l5_weights, &self.l5_biases)
            .map(relu)
            .fully_connected_pass(&self.l6_weights, &self.l6_biases)
            .split_last();
        (vec.softmax(), 2. * sig(board_eval) - 1.)
    }

    pub fn back_prop(
        &mut self,
        input: Tensor3<f64, 5, 5, 8>,
        training_vec: Tensor1<f64, 625>,
        training_eval: f64,
    ) {
        let (probability_vec, board_eval) = self.feed_forward(input);
        let cost = (training_eval - board_eval).powi(2) - (probability_vec.map(f64::ln) * &training_vec).sum();

        // TODO
        // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        // https://github.com/frandelgado/mytorch/blob/master/nets/activations.py
    }
}

#[cfg(test)]
mod benches {
    use super::*;
    use std::thread;
    use test::Bencher;

    fn with_larger_stack<'a, T, F>(f: F)
    where
        T: 'a + Send,
        F: FnOnce() -> T,
        F: 'a + Send,
    {
        unsafe {
            thread::Builder::new()
                .stack_size(1024 * 1024 * 1024 * 16)
                .spawn_unchecked(f)
                .unwrap()
                .join()
                .unwrap();
        }
    }

    #[bench]
    fn init(ben: &mut Bencher) {
        with_larger_stack(move || {
            ben.iter(|| Network::init());
        });
    }

    #[bench]
    fn forward_pass(ben: &mut Bencher) {
        with_larger_stack(move || {
            let network = Network::init();
            let input = Tensor3::rand(rand_distr::Uniform::new(0., 1.));
            ben.iter(|| network.feed_forward(input));
        })
    }
}
