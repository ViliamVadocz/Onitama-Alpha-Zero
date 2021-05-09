use std::fs;

use tensor::*;

const LEARNING_RATE: f64 = 0.001;
const NETWORK_SIZE: usize = 3 * 3 * 8 * 64
    + 5 * 5 * 64
    + 3 * 3 * 64 * 64
    + 5 * 5 * 64
    + 3 * 3 * 64 * 64
    + 5 * 5 * 64
    + 3 * 3 * 64 * 64
    + 5 * 5 * 64
    + 1600 * 800
    + 800
    + 800 * 626
    + 626;

#[derive(Debug, Clone, Copy, PartialEq)]
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
        let cost = (training_eval - board_eval).powi(2)
            - (probability_vec.map(f64::ln) * &training_vec).sum();

        // TODO
        // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        // https://github.com/frandelgado/mytorch/blob/master/nets/activations.py
        // https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
    }
}

impl Network {
    fn get_save_data(&self) -> Vec<f64> {
        let mut data = Vec::with_capacity(NETWORK_SIZE);

        // Convolutional layers.
        for tensor in self.l1_kernels.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l1_biases.iter());
        for tensor in self.l2_kernels.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l2_biases.iter());
        for tensor in self.l3_kernels.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l3_biases.iter());
        for tensor in self.l4_kernels.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l4_biases.iter());
        // Fully connected layers.
        for tensor in self.l5_weights.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l5_biases.iter());
        for tensor in self.l6_weights.iter() {
            data.extend(tensor.iter());
        }
        data.extend(self.l6_biases.iter());

        data
    }

    fn from_save_data(data: Vec<f64>) -> Network {
        assert_eq!(data.len(), NETWORK_SIZE);
        let mut iter = data.into_iter();
        Network {
            // Padded convolution layers.
            l1_kernels: [(); 64]
                .map(|()| Tensor3::new([(); 3 * 3 * 8].map(|()| iter.next().unwrap()))),
            l1_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l2_kernels: [(); 64]
                .map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
            l2_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l3_kernels: [(); 64]
                .map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
            l3_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l4_kernels: [(); 64]
                .map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
            l4_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            // Fully connected layers.
            l5_weights: [(); 800].map(|()| Tensor1::new([(); 1600].map(|()| iter.next().unwrap()))),
            l5_biases: Tensor1::new([(); 800].map(|()| iter.next().unwrap())),
            l6_weights: [(); 626].map(|()| Tensor1::new([(); 800].map(|()| iter.next().unwrap()))),
            l6_biases: Tensor1::new([(); 626].map(|()| iter.next().unwrap())),
        }
    }

    pub fn save(&self, path: &str) {
        let data = bincode::serialize(&self.get_save_data()).unwrap();
        fs::write(path, data).expect("couldn't save network to file");
    }

    pub fn load(path: &str) -> Network {
        let data = fs::read(path).expect("couldn't read file");
        Network::from_save_data(bincode::deserialize(&data).unwrap())
    }
}

use std::thread;
#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load() {
        with_larger_stack(|| {
            let orig = Network::init();
            orig.save("test.data");
            let network = Network::load("test.data");
            fs::remove_file("test.data").unwrap();
            assert_eq!(orig, network);
        })
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

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
