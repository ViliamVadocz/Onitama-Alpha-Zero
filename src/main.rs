#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

mod matrix;
use matrix::Matrix;

fn main() {
    #[rustfmt::skip]
    let a = Matrix::<5, 5>::new([
        1., 1., 1., 1., 1.,
        2., 2., 2., 2., 2.,
        3., 3., 3., 3., 3.,
        1., 0., 2., 0., 3.,
        0., 1., 0., 2., 0.,
    ]);
    let b = Matrix::<3, 3>::new([1., 2., 3., 1., 2., 3., 1., 2., 3.]);
    println!("{:.2}", a.convolve(b));
    println!("{:.2}", a.convolve_with_pad(b));
}
