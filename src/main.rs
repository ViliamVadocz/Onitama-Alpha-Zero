#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use rand::distributions::Uniform;
use tensor::matrix::Matrix;

fn main() {
    #[rustfmt::skip]
    let a = Matrix::<_, 5, 5>::new([
        1., 1., 1., 1., 1.,
        2., 2., 2., 2., 2.,
        3., 3., 3., 3., 3.,
        1., 0., 2., 0., 3.,
        0., 1., 0., 2., 0.,
    ]);
    #[rustfmt::skip]
    let b = Matrix::<_, 3, 3>::new([
        1., 2., 3.,
        1., 2., 3.,
        1., 2., 3.,
    ]);
    // println!("{:.2}", a.convolve(b));
    // println!("{:.2}", a.convolve_with_pad(b));

    let c = Matrix::<_, 5, 4>::rand(Uniform::new(0., 5.));
    println!("{:.2}", c);
    // println!("{:.2}", c.reshape::<2, 10>());
    // println!("{:.2}", c.transpose());
    // println!("{:.2}", c.convolve(b))

    println!(
        "{:.2}",
        a.matmul(c)
            .matmul(c.transpose())
            .matmul(a)
            .scale(0.0001)
            .convolve(b)
    );

    #[rustfmt::skip]
    let d = Matrix::<_, 3, 3>::new([
        1, 2, 3,
        3, 2, 1,
        0, 0, 1,
    ]);
    println!("{}", d);

    let e = Matrix::<_, 3, 3>::rand(Uniform::new(0, 5));
    println!("{}", e);

    println!("{}\n{}", d.matmul(e), e.matmul(d));
}
