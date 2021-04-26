#![feature(const_generics, const_evaluatable_checked, array_map)]
#![allow(incomplete_features)]

use tensor::*;

fn ReLU(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

fn forward_pass<
    const IN_D1: usize,
    const IN_D2: usize,
    const IN_D3: usize,
    const K_D1: usize,
    const K_D2: usize,
    const N: usize,
>(
    input: Tensor3<f64, IN_D1, IN_D2, IN_D3>,
    kernels: [Tensor3<f64, K_D1, K_D2, IN_D3>; N],
    bias: Tensor3<f64, IN_D1, IN_D2, { IN_D3 * N }>,
) -> Tensor3<f64, IN_D1, IN_D2, { IN_D3 * N }>
where
    [(); IN_D1 * IN_D2 * IN_D3]: ,
    [(); K_D1 * K_D2 * IN_D3]: ,
    [(); IN_D1 * IN_D2 * (IN_D3 * N)]: ,
{
    let conv_out = kernels.map(|kernel| input.convolve_with_pad(kernel));
    let data = conv_out
        .map(|out| out.get_data())
        .iter()
        .flatten()
        .collect();
    Tensor3::new(data)
}

fn main() {}
