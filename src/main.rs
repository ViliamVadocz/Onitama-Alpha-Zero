#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

use rand::distributions::Uniform;

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
    #[rustfmt::skip]
    let b = Matrix::<3, 3>::new([
        1., 2., 3.,
        1., 2., 3.,
        1., 2., 3.,
    ]);
    // println!("{:.2}", a.convolve(b));
    // println!("{:.2}", a.convolve_with_pad(b));

    let c = Matrix::<5, 4>::rand(Uniform::new(0., 5.));
    println!("{:.2}", c);
    // println!("{:.2}", c.reshape::<2, 10>());
    // println!("{:.2}", c.transpose());

    println!("{:.2}", c.convolve(b))
}
