use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use rand::{thread_rng, distributions::Distribution};

#[derive(Clone, Copy, Debug)]
pub struct Matrix<const R: usize, const C: usize>([f64; R * C])
where
    [(); R * C]: ;

impl<const R: usize, const C: usize> fmt::Display for Matrix<R, C>
where
    [(); R * C]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..R {
            fmt::Debug::fmt(&self.0[(i * C)..((i + 1) * C)], f)?;
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl<const R: usize, const C: usize> Matrix<R, C>
where
    [(); R * C]: ,
{
    pub fn new(data: [f64; R * C]) -> Self {
        Self(data)
    }

    /// Initialize a random matrix.
    pub fn rand<D: Distribution<f64>>(distr: D) -> Self {
        let mut rng = thread_rng();
        let mut data = [0.0; R * C];
        for elem in data.iter_mut() {
            *elem = distr.sample(&mut rng);
        }
        Matrix::new(data)
    }

    /// Multiply all elements in the matrix by a scalar.
    pub fn scale(mut self, scalar: f64) -> Self {
        for elem in self.0.iter_mut() {
            *elem *= scalar
        }
        self
    }

    /// Get a slice of the matrix.
    pub fn slice<const NEW_R: usize, const NEW_C: usize>(
        self,
        row_offset: usize,
        col_offset: usize,
    ) -> Matrix<NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        assert!(row_offset + NEW_R <= R);
        assert!(col_offset + NEW_C <= C);
        let mut data = [0.0; NEW_R * NEW_C];
        for (i, elem) in data.iter_mut().enumerate() {
            let r = i / NEW_R + row_offset;
            let c = i % NEW_C + col_offset;
            *elem = self.0[r * C + c];
        }
        Matrix::new(data)
    }

    /// Get a slice of the matrix with padding if needed.
    /// Allow slices which do not entirely fit inside the matrix.
    /// Padded with zeros.
    pub fn slice_with_pad<const NEW_R: usize, const NEW_C: usize>(
        self,
        row_offset: isize,
        col_offset: isize,
    ) -> Matrix<NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        let mut data = [0.0; NEW_R * NEW_C];
        let new_r = NEW_R as isize;
        let new_c = NEW_C as isize;
        for (i, elem) in data.iter_mut().enumerate() {
            let i = i as isize;
            let r = i / new_r + row_offset;
            let c = i % new_c + col_offset;
            if r < 0 || c < 0 {
                continue;
            }
            let r = r as usize;
            let c = c as usize;
            if r >= R || c >= C {
                continue;
            }
            *elem = self.0[r * C + c];
        }
        Matrix::new(data)
    }
        
    /// Perform naive convolution without padding.
    pub fn convolve<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Matrix<KERN_R, KERN_C>,
    ) -> Matrix<{ R + 1 - KERN_R }, { C + 1 - KERN_C }>
    where
        [(); KERN_R * KERN_C]: ,
        [(); (R + 1 - KERN_R) * (C + 1 - KERN_C)]: ,
    {
        let mut res = Matrix::new([0.0; (R + 1 - KERN_R) * (C + 1 - KERN_C)]);
        for (i, &elem) in kernel.0.iter().enumerate() {
            let r = i / KERN_R;
            let c = i % KERN_C;
            res += self.slice(r, c).scale(elem)
        }
        res
    }

    /// Perform naive convolution with padding.
    /// Padded with zeros.
    pub fn convolve_with_pad<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Matrix<KERN_R, KERN_C>,
    ) -> Matrix<R, C>
    where
        [(); KERN_R * KERN_C]: ,
        [(); (R + KERN_R - 1) * (C + KERN_C - 1)]: ,
    {
        let mut res = Matrix::new([0.0; R * C]);
        let kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &elem) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let r = i / kern_r - kern_r / 2;
            let c = i % kern_c - kern_c / 2;
            res += self.slice_with_pad(r, c).scale(elem)
        }
        res
    }
}

// Implement element-wise operations.
macro_rules! impl_op {
    ($bound:ident, $method:ident) => {
        impl<const R: usize, const C: usize> $bound for Matrix<R, C>
        where
            [(); R * C]: ,
        {
            type Output = Self;

            fn $method(mut self, other: Self) -> Self {
                for (elem, val) in self.0.iter_mut().zip(other.0.iter()) {
                    *elem = $bound::$method(*elem, val);
                }
                self
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

// Implement element-wise assign operations.
macro_rules! impl_op_assign {
    ($bound:ident, $method:ident) => {
        impl<const R: usize, const C: usize> $bound for Matrix<R, C>
        where
            [(); R * C]: ,
        {
            fn $method(&mut self, other: Self) {
                for (elem, val) in self.0.iter_mut().zip(other.0.iter()) {
                    $bound::$method(elem, val);
                }
            }
        }
    };
}

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);
impl_op_assign!(DivAssign, div_assign);
