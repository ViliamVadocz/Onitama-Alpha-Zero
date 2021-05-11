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
    /* 626 outputs
     * 625 for each from-to move combination
     * 1 output for board evaluation */
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

    #[allow(non_snake_case, clippy::many_single_char_names)]
    pub fn back_prop(&mut self, input: Tensor3<f64, 5, 5, 8>, pi: Tensor1<f64, 625>, z: f64) -> f64 {
        // Some resources:
        // https://youtu.be/Ilg3gGewQ5U
        // http://neuralnetworksanddeeplearning.com/chap2.html
        // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        // https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c

        // We want to minimize the cost L.
        // L = (z - v)^2 - pi . log(p)
        // where
        //     z  = desired board eval
        //     v  = network board eval
        //     pi = improved policy
        //     p  = network policy

        // Feed-forward while keeping track of intermediate values.
        // x is pre-activation.
        // a is activation.
        let l1_x = input.convolution_pass(&self.l1_kernels, &self.l1_biases);
        let l1_a = l1_x.map(relu);
        let l2_x = l1_a.convolution_pass(&self.l2_kernels, &self.l2_biases);
        let l2_a = l2_x.map(relu);
        let l3_x = l2_a.convolution_pass(&self.l3_kernels, &self.l3_biases);
        let l3_a = l3_x.map(relu);
        let l4_x = l3_a.convolution_pass(&self.l4_kernels, &self.l4_biases);
        let l4_a = l4_x.map(relu).reshape::<Tensor1<_, 1600>>();
        let l5_x = l4_a.fully_connected_pass(&self.l5_weights, &self.l5_biases);
        let l5_a = l5_x.map(relu);
        let l6_x = l5_a.fully_connected_pass(&self.l6_weights, &self.l6_biases);
        let (o, b) = l6_x.split_last();
        let p = o.softmax();
        let v = b.tanh();
        let L = (z - v).powi(2) - (pi * &p.map(f64::ln)).sum();

        let dL_dv = 2. * (v - z);
        let dL_db = dL_dv * (1. / b.cosh().powi(2)); // tanh'(x) = sech^2(x)

        let dL_do = p - &pi; // Use log trick with derivative of softmax.

        // Combine dl_do and dl_db.
        let dL_dl6_x = {
            let mut data = [0.; 626];
            for (elem, val) in data.iter_mut().zip(dL_do.iter()) {
                *elem = val;
            }
            data[625] = dL_db;
            Tensor1::new(data)
        };

        // TODO Put this into functions!!!

        // Back-propagate layer 6.
        let dL_dl6_w: [Tensor1<f64, 800>; 626] = {
            let mut iter = dL_dl6_x.iter();
            [(); 626].map(|()| l5_a.scale(iter.next().unwrap() * LEARNING_RATE))
        };
        let dL_dl5_a = {
            let mut data = [0.; 800];
            let mut iter = self.l6_weights.iter();
            for (k, elem) in data.iter_mut().enumerate() {
                *elem = (dL_dl6_x * &Tensor1::new([(); 626].map(|()| iter.next().unwrap().nth(k)))).sum()
            }
            Tensor1::new(data)
        };
        self.l6_weights
            .iter_mut()
            .zip(dL_dl6_w.iter())
            .for_each(|(weights, adjustment)| *weights -= adjustment);
        self.l6_biases -= &dL_dl6_x.scale(LEARNING_RATE);

        // Back-propagate layer 5.
        let dL_dl5_x = dL_dl5_a.map(d_relu);
        let dL_dl5_w: [Tensor1<f64, 1600>; 800] = {
            let mut iter = dL_dl5_x.iter();
            [(); 800].map(|()| l4_a.scale(iter.next().unwrap() * LEARNING_RATE))
        };
        let dL_dl4_a = {
            let mut data = [0.; 1600];
            let mut iter = self.l5_weights.iter();
            for (k, elem) in data.iter_mut().enumerate() {
                *elem = (dL_dl5_x * &Tensor1::new([(); 800].map(|()| iter.next().unwrap().nth(k)))).sum()
            }
            Tensor1::new(data)
        };
        self.l5_weights
            .iter_mut()
            .zip(dL_dl5_w.iter())
            .for_each(|(weights, adjustment)| *weights -= adjustment);
        self.l5_biases -= &dL_dl5_x.scale(LEARNING_RATE);

        // Back-propagate layer 4.
        let dL_dl4_x = dL_dl4_a.map(d_relu).reshape::<Tensor3<f64, 5, 5, 64>>();
        let dL_dl4_k = {
            let mut iter = 0..64;
            [(); 64].map(|()| {
                let i = iter.next().unwrap();
                l3_a.convolve_with_pad_to(&dL_dl4_x.slice::<5, 5, 1>(0, 0, i))
            })
        };
        let dL_dl3_a = {
            let mut res = Tensor3::<_, 5, 5, 64>::new([0.; 1600]);
            for (i, &kernel) in self.l4_kernels.iter().enumerate() {
                res += &kernel
                    .rev()
                    .convolve_with_pad_to(&dL_dl4_x.slice::<5, 5, 1>(0, 0, i))
                    .rev();
            }
            res
        };
        self.l4_kernels
            .iter_mut()
            .zip(dL_dl4_k.iter())
            .for_each(|(kernel, adjustment)| *kernel -= adjustment);
        self.l4_biases -= &dL_dl4_x.scale(LEARNING_RATE);

        // Back-propagate layer 3.
        let dL_dl3_x = dL_dl3_a.map(d_relu);
        let dL_dl3_k = {
            let mut iter = 0..64;
            [(); 64].map(|()| {
                let i = iter.next().unwrap();
                l2_a.convolve_with_pad_to(&dL_dl3_x.slice::<5, 5, 1>(0, 0, i))
            })
        };
        let dL_dl2_a = {
            let mut res = Tensor3::<_, 5, 5, 64>::new([0.; 1600]);
            for (i, &kernel) in self.l3_kernels.iter().enumerate() {
                res += &kernel
                    .rev()
                    .convolve_with_pad_to(&dL_dl3_x.slice::<5, 5, 1>(0, 0, i))
                    .rev();
            }
            res
        };
        self.l3_kernels
            .iter_mut()
            .zip(dL_dl3_k.iter())
            .for_each(|(kernel, adjustment)| *kernel -= adjustment);
        self.l3_biases -= &dL_dl3_x.scale(LEARNING_RATE);

        // Back-propagate layer 2.
        let dL_dl2_x = dL_dl2_a.map(d_relu);
        let dL_dl2_k = {
            let mut iter = 0..64;
            [(); 64].map(|()| {
                let i = iter.next().unwrap();
                l1_a.convolve_with_pad_to(&dL_dl2_x.slice::<5, 5, 1>(0, 0, i))
            })
        };
        let dL_dl1_a = {
            let mut res = Tensor3::<_, 5, 5, 64>::new([0.; 1600]);
            for (i, &kernel) in self.l2_kernels.iter().enumerate() {
                res += &kernel
                    .rev()
                    .convolve_with_pad_to(&dL_dl2_x.slice::<5, 5, 1>(0, 0, i))
                    .rev();
            }
            res
        };
        self.l2_kernels
            .iter_mut()
            .zip(dL_dl2_k.iter())
            .for_each(|(kernel, adjustment)| *kernel -= adjustment);
        self.l2_biases -= &dL_dl2_x.scale(LEARNING_RATE);

        // Back-propagate layer 1.
        let dL_dl1_x = dL_dl1_a.map(d_relu);
        let dL_dl1_k = {
            let mut iter = 0..64;
            [(); 64].map(|()| {
                let i = iter.next().unwrap();
                l1_a.convolve_with_pad_to(&dL_dl1_x.slice::<5, 5, 1>(0, 0, i))
            })
        };
        self.l1_kernels
            .iter_mut()
            .zip(dL_dl1_k.iter())
            .for_each(|(kernel, adjustment)| *kernel -= adjustment);
        self.l1_biases -= &dL_dl1_x.scale(LEARNING_RATE);

        // Return loss just to track if it is going down.
        L
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
            l1_kernels: [(); 64].map(|()| Tensor3::new([(); 3 * 3 * 8].map(|()| iter.next().unwrap()))),
            l1_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l2_kernels: [(); 64].map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
            l2_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l3_kernels: [(); 64].map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
            l3_biases: Tensor3::new([(); 5 * 5 * 64].map(|()| iter.next().unwrap())),
            l4_kernels: [(); 64].map(|()| Tensor3::new([(); 3 * 3 * 64].map(|()| iter.next().unwrap()))),
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
