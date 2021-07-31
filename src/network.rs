use std::fs;
use std::sync::Arc;

use tensor::*;

const LEARNING_RATE: f64 = 0.0001;
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

#[derive(Clone)]
pub struct Network {
    fft_planner: Arc<FftPlanner<f64>>,
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
            fft_planner: Arc::new(FftPlanner::new()),
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
            .convolution_pass(&self.l1_kernels, &self.l1_biases, &mut self.fft_planner)
            .map(relu)
            .convolution_pass(&self.l2_kernels, &self.l2_biases, &mut self.fft_planner)
            .map(relu)
            .convolution_pass(&self.l3_kernels, &self.l3_biases, &mut self.fft_planner)
            .map(relu)
            .convolution_pass(&self.l4_kernels, &self.l4_biases, &mut self.fft_planner)
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
        let l1_x = input.convolution_pass(&self.l1_kernels, &self.l1_biases, &mut self.fft_planner);
        let l1_a = l1_x.map(relu);
        let l2_x = l1_a.convolution_pass(&self.l2_kernels, &self.l2_biases, &mut self.fft_planner);
        let l2_a = l2_x.map(relu);
        let l3_x = l2_a.convolution_pass(&self.l3_kernels, &self.l3_biases, &mut self.fft_planner);
        let l3_a = l3_x.map(relu);
        let l4_x = l3_a.convolution_pass(&self.l4_kernels, &self.l4_biases, &mut self.fft_planner);
        let l4_a = l4_x.map(relu).reshape::<Tensor1<_, 1600>>();
        let l5_x = l4_a.fully_connected_pass(&self.l5_weights, &self.l5_biases);
        let l5_a = l5_x.map(relu);
        let l6_x = l5_a.fully_connected_pass(&self.l6_weights, &self.l6_biases);
        let (o, b) = l6_x.split_last();
        let p = o.softmax();
        let v = b.tanh();
        // The cost function.
        let L = (z - v).powi(2) - (pi * &p.map(f64::ln)).sum();

        // Begin calculating partial derivatives.
        let dL_dv = 2. * (v - z);
        let dL_db = dL_dv * (1. / b.cosh().powi(2)); // tanh'(x) = sech^2(x)
        let dL_do = p - &pi; // Use log trick with derivative of softmax.

        // Combine dl_do and dl_db.
        let dL_dx = {
            let mut data = [0.; 626];
            for (elem, val) in data.iter_mut().zip(dL_do.into_iter()) {
                *elem = val;
            }
            data[625] = dL_db;
            Tensor1::new(data)
        };

        let dL_da = bp_fully_connected(&mut self.l6_weights, &mut self.l6_biases, l5_a, dL_dx);
        let dL_da = bp_fully_connected(&mut self.l5_weights, &mut self.l5_biases, l4_a, dL_da.map(d_relu));
        let dL_da = bp_convolution(
            &mut self.l4_kernels,
            &mut self.l4_biases,
            l3_a,
            dL_da.map(d_relu).reshape(),
        );
        let dL_da = bp_convolution(&mut self.l3_kernels, &mut self.l3_biases, l2_a, dL_da.map(d_relu));
        let dL_da = bp_convolution(&mut self.l2_kernels, &mut self.l2_biases, l1_a, dL_da.map(d_relu));
        let _ = bp_convolution(
            &mut self.l1_kernels,
            &mut self.l1_biases,
            input,
            dL_da.map(d_relu),
        );

        // Return loss just to track if it is going down.
        L
    }
}

/// Do back-propagation for a fully connected layer.
fn bp_fully_connected<const A: usize, const B: usize>(
    weights: &mut [Tensor1<f64, A>; B],
    biases: &mut Tensor1<f64, B>,
    prev_activations: Tensor1<f64, A>,
    next_layer_derivatives: Tensor1<f64, B>,
) -> Tensor1<f64, A> {
    let change_in_weights = {
        let mut iter = next_layer_derivatives.iter();
        [(); B].map(|()| prev_activations.scale(iter.next().unwrap() * LEARNING_RATE))
    };
    let prev_layer_derivatives = {
        let mut data = [0.; A];
        for (n, elem) in data.iter_mut().enumerate() {
            let mut iter = weights.iter();
            *elem =
                (next_layer_derivatives * &Tensor1::new([(); B].map(|()| iter.next().unwrap().nth(n)))).sum();
        }
        Tensor1::new(data)
    };

    weights
        .iter_mut()
        .zip(change_in_weights.iter())
        .for_each(|(weights, adjustment)| *weights -= adjustment);
    *biases -= &next_layer_derivatives.scale(LEARNING_RATE);

    prev_layer_derivatives
}

/// Do back-propagation for a convolution layer.
fn bp_convolution<const A: usize, const B: usize>(
    kernels: &mut [Tensor3<f64, 3, 3, A>; B],
    biases: &mut Tensor3<f64, 5, 5, B>,
    prev_activations: Tensor3<f64, 5, 5, A>,
    next_layer_derivatives: Tensor3<f64, 5, 5, B>,
) -> Tensor3<f64, 5, 5, A>
where
    [(); 3 * 3 * A]: ,
    [(); 5 * 5 * A]: ,
    [(); 5 * 5 * B]: ,
{
    let change_in_kernels = {
        let mut iter = 0..64;
        [(); 64].map(|()| {
            let i = iter.next().unwrap();
            prev_activations
                .convolve_with_pad_to(next_layer_derivatives.slice::<5, 5, 1>(0, 0, i))
                .scale(LEARNING_RATE)
        })
    };
    let prev_layer_derivatives = {
        let mut res = Tensor3::<_, 5, 5, A>::new([0.; 5 * 5 * A]);
        for (i, &kernel) in kernels.iter().enumerate() {
            res += &kernel
                .rev()
                .convolve_with_pad_to(next_layer_derivatives.slice::<5, 5, 1>(0, 0, i))
                .rev();
        }
        res
    };
    kernels
        .iter_mut()
        .zip(change_in_kernels.iter())
        .for_each(|(kernel, adjustment)| *kernel -= adjustment);
    *biases -= &next_layer_derivatives.scale(LEARNING_RATE);

    prev_layer_derivatives
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
            fft_planner: Arc::new(FftPlanner::new()),
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
            assert_eq!(orig.l1_kernels, network.l1_kernels);
            assert_eq!(orig.l1_biases, network.l1_biases);
            assert_eq!(orig.l2_kernels, network.l2_kernels);
            assert_eq!(orig.l2_biases, network.l2_biases);
            assert_eq!(orig.l3_kernels, network.l3_kernels);
            assert_eq!(orig.l3_biases, network.l3_biases);
            assert_eq!(orig.l4_kernels, network.l4_kernels);
            assert_eq!(orig.l4_biases, network.l4_biases);
            assert_eq!(orig.l5_weights, network.l5_weights);
            assert_eq!(orig.l5_biases, network.l5_biases);
            assert_eq!(orig.l6_weights, network.l6_weights);
            assert_eq!(orig.l6_biases, network.l6_biases);
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

    #[bench]
    fn back_prop(ben: &mut Bencher) {
        with_larger_stack(move || {
            let mut network = Network::init();
            let input = Tensor3::rand(rand_distr::Uniform::new(0., 1.));
            let pi = Tensor1::rand(rand_distr::Uniform::new(0., 1.));
            let z = 0.5;
            ben.iter(|| network.back_prop(input, pi, z));
        })
    }
}
