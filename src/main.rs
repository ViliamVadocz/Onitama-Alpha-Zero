#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

use tensor::{Tensor, Tensor1, Tensor2, Tensor3, ElementWiseTensor, RandomTensor, rand_distr::Uniform, reshape};

fn main() {
    #[rustfmt::skip]
    let a = Tensor1::<i32, 10>::new([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    println!("{}", a);
    #[rustfmt::skip]
    let b = Tensor2::<i32, 3, 3>::new([
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]);
    println!("{}", b);
    #[rustfmt::skip]
    let c = Tensor3::<i32, 2, 2, 2>::new([
        0, 1,
        2, 3,

        4, 5,
        6, 7,
    ]);
    println!("{}", c);

    // println!("{}", a.slice::<3>(4));
    // println!("{}", b.slice::<2, 2>(1, 1));
    // println!("{}", c.slice::<2, 1, 1>(0, 1, 1));

    println!("{}", reshape::<_, _, Tensor2<_, 2, 2>, 4>(a.slice::<4>(3)));
}
