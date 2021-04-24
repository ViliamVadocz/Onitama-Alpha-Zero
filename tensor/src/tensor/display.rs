use super::*;

impl<T, const L: usize> Display for Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)?;
        writeln!(f,)
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

impl<T, const D1: usize, const D2: usize, const D3: usize> Display for Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..D3 {
            for j in 0..D2 {
                writeln!(f,)?;
                Debug::fmt(
                    &self.0[((i * D1 * D2) + j * D1)..((i * D1 * D2) + (j + 1) * D1)],
                    f,
                )?;
            }
            writeln!(f,)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}
