
pub mod elementwise;
#[cfg(feature = "rand")]
pub mod random;

use std::fmt::{self, Display, Debug};

pub type Vector<T, const L: usize> = Tensor1<T, L>;
pub type Matrix<T, const R: usize, const C: usize> = Tensor2<T, R, C>;

pub trait Tensor<T, const X: usize>:
    Default
    + Display
where
    T: Default + Copy + Debug,
{
    fn new(data: [T; X]) -> Self;
    fn get_data(self) -> [T; X];
    fn get_data_mut(&mut self) -> &mut [T; X];
    // fn reshape<G: Tensor<T, X>>(self) -> G;
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor1<T, const L: usize>([T; L])
where
    T: Default + Copy + Debug;

impl<T, const L: usize> Default for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn default() -> Self {
        Tensor1::new([T::default(); L])
    }
}

impl<T, const L: usize> Display for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<T, const L: usize> Tensor<T, L> for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn new(data: [T; L]) -> Self {
        Tensor1(data)
    }

    fn get_data(self) -> [T; L] {
        self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; L] {
        &mut self.0
    }

    // fn reshape<G: Tensor<T, L>>(self) -> G {
    //     G::new(self.0)
    // }
}
////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
pub struct Tensor2<T, const R: usize, const C: usize>([T; R * C])
where
    T: Default + Copy + Debug,
    [(); R * C]: ;

impl<T, const R: usize, const C: usize> Default for Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    fn default() -> Self {
        Tensor2::new([T::default(); R * C])
    }
}

impl<T, const R: usize, const C: usize> Display for Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..R {
            Debug::fmt(&self.0[(i * C)..((i + 1) * C)], f)?;
            writeln!(f,)?;
        }
        Ok(())
    }
}

impl<T, const R: usize, const C: usize> Tensor<T, { R * C }> for Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    fn new(data: [T; R * C]) -> Self {
        Tensor2(data)
    }

    fn get_data(self) -> [T; R * C] {
        self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; R * C] {
        &mut self.0
    }

    // fn reshape<G: Tensor<T, { R * C }>>(self) -> G {
    //     G::new(self.0)
    // }
}

////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Copy, Debug)]
pub struct Tensor3<T, const D1: usize, const D2: usize, const D3: usize>([T; D1 * D2 * D3])
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ;

impl<T, const D1: usize, const D2: usize, const D3: usize> Default for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn default() -> Self {
        Tensor3::new([T::default(); D1 * D2 * D3])
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Display for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for i in 0..D3 {
            for j in 0..D2 {
                let s = i * D2;
                Debug::fmt(&self.0[(s + j * D1)..(s + (j + 1) * D1)], f)?;
                writeln!(f,)?;
            }
            writeln!(f,)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor<T, { D1 * D2 * D3 }>
    for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn new(data: [T; D1 * D2 * D3]) -> Self {
        Tensor3(data)
    }

    fn get_data(self) -> [T; D1 * D2 * D3] {
        self.0
    }

    fn get_data_mut(&mut self) -> &mut [T; D1 * D2 * D3] {
        &mut self.0
    }

    // fn reshape<G: Tensor<T, { D1 * D2 * D3 }>>(self) -> G {
    //     G::new(self.0)
    // }
}
