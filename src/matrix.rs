use rand::{distributions::Distribution, thread_rng};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::{convert::TryInto, fmt, iter::Sum};

#[derive(Clone, Copy, Debug)]
pub struct Matrix<T, const R: usize, const C: usize>([T; R * C])
where
    T: Default + Copy,
    [(); R * C]: ;

impl<T, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C>
where
    T: Default + Copy + fmt::Debug,
    [(); R * C]: ,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..R {
            fmt::Debug::fmt(&self.0[(i * C)..((i + 1) * C)], f)?;
            writeln!(f, )?;
        }
        Ok(())
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    pub fn new(data: [T; R * C]) -> Self {
        Self(data)
    }

    /// Initialize a random matrix.
    pub fn rand<D: Distribution<T>>(distr: D) -> Self {
        let mut rng = thread_rng();
        let mut data = [T::default(); R * C];
        for elem in data.iter_mut() {
            *elem = distr.sample(&mut rng);
        }
        Matrix::new(data)
    }

    /// Change the shape of the matrix, keeping the elements in the same order.
    pub fn reshape<const NEW_R: usize, const NEW_C: usize>(self) -> Matrix<T, NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        assert_eq!(NEW_R * NEW_C, R * C);
        let mut data = [T::default(); NEW_R * NEW_C];
        for (elem, &val) in data.iter_mut().zip(self.0.iter()) {
            *elem = val;
        }
        Matrix::new(data)
    }

    /// Get the transpose of a matrix.
    pub fn transpose(self) -> Matrix<T, C, R>
    where
        [(); C * R]: ,
    {
        let mut data = [T::default(); C * R];
        for (i, elem) in data.iter_mut().enumerate() {
            let r = i % R;
            let c = i / R;
            *elem = self.0[r * C + c];
        }
        Matrix::new(data)
    }

    /// Multiply all elements in the matrix by a scalar.
    pub fn scale(mut self, scalar: T) -> Self
    where
        T: MulAssign,
    {
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
    ) -> Matrix<T, NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        assert!(row_offset + NEW_R <= R);
        assert!(col_offset + NEW_C <= C);
        let mut data = [T::default(); NEW_R * NEW_C];
        for (i, elem) in data.iter_mut().enumerate() {
            let r = i / NEW_C + row_offset;
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
    ) -> Matrix<T, NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        let mut data = [T::default(); NEW_R * NEW_C];
        // let new_r = NEW_R as isize;
        let new_c = NEW_C as isize;
        for (i, elem) in data.iter_mut().enumerate() {
            let i = i as isize;
            let r = i / new_c + row_offset;
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
        kernel: Matrix<T, KERN_R, KERN_C>,
    ) -> Matrix<T, { R + 1 - KERN_R }, { C + 1 - KERN_C }>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + 1 - KERN_R) * (C + 1 - KERN_C)]: ,
    {
        let mut res = Matrix::new([T::default(); (R + 1 - KERN_R) * (C + 1 - KERN_C)]);
        for (i, &elem) in kernel.0.iter().enumerate() {
            let r = i / KERN_C;
            let c = i % KERN_C;
            res += self.slice(r, c).scale(elem)
        }
        res
    }

    /// Perform naive convolution with padding.
    /// Padded with zeros.
    pub fn convolve_with_pad<const KERN_R: usize, const KERN_C: usize>(
        self,
        kernel: Matrix<T, KERN_R, KERN_C>,
    ) -> Matrix<T, R, C>
    where
        T: AddAssign + MulAssign,
        [(); KERN_R * KERN_C]: ,
        [(); (R + KERN_R - 1) * (C + KERN_C - 1)]: ,
    {
        let mut res = Matrix::new([T::default(); R * C]);
        let kern_r = KERN_R as isize;
        let kern_c = KERN_C as isize;
        for (i, &elem) in kernel.0.iter().enumerate() {
            let i = i as isize;
            let r = i / kern_c - kern_r / 2;
            let c = i % kern_c - kern_c / 2;
            res += self.slice_with_pad(r, c).scale(elem)
        }
        res
    }

    /// Naive matrix multiplication.
    pub fn matmul<const K: usize>(self, other: Matrix<T, C, K>) -> Matrix<T, R, K>
    where
        T: Add<Output = T> + Mul<Output = T> + Sum + Copy,
        [(); C * K]: ,
        [(); R * K]: ,
    {
        let mut data = [T::default(); R * K];
        for (i, row) in self.rows().enumerate() {
            for (j, col) in other.cols().enumerate() {
                data[i * K + j] = row.iter().zip(col.iter()).map(|(&a, &b)| a * b).sum();
            }
        }
        Matrix::new(data)
    }
}

// Iterators for rows and columns.
pub struct Rows<'a, T, const R: usize, const C: usize> {
    data: &'a [T],
    row: usize,
}

impl<'a, T, const R: usize, const C: usize> Iterator for Rows<'a, T, R, C>
where
    T: Copy,
{
    type Item = [T; C];

    fn next(&mut self) -> Option<[T; C]> {
        if self.row >= R {
            return None;
        }
        let next = self.data[(self.row * C)..((self.row + 1) * C)]
            .try_into()
            .unwrap();
        self.row += 1;
        Some(next)
    }
}

pub struct Cols<'a, T, const R: usize, const C: usize> {
    data: &'a [T],
    col: usize,
}

impl<'a, T, const R: usize, const C: usize> Iterator for Cols<'a, T, R, C>
where
    T: Default + Copy,
{
    type Item = [T; R];

    fn next(&mut self) -> Option<[T; R]> {
        if self.col >= C {
            return None;
        }
        let mut next = [T::default(); R];
        for (row, elem) in next.iter_mut().enumerate() {
            *elem = self.data[row * C + self.col]
        }
        self.col += 1;
        Some(next)
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
    pub fn rows(&self) -> Rows<'_, T, R, C> {
        Rows {
            data: &self.0,
            row: 0,
        }
    }

    pub fn cols(&self) -> Cols<'_, T, R, C> {
        Cols {
            data: &self.0,
            col: 0,
        }
    }
}

// Implement element-wise operations.
macro_rules! impl_op {
    ($bound:ident, $method:ident) => {
        impl<T, const R: usize, const C: usize> $bound for Matrix<T, R, C>
        where
            T: Default + Copy + $bound<Output = T>,
            [(); R * C]: ,
        {
            type Output = Self;

            fn $method(mut self, other: Self) -> Self {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
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
        impl<T, const R: usize, const C: usize> $bound for Matrix<T, R, C>
        where
            T: Default + Copy + $bound,
            [(); R * C]: ,
        {
            fn $method(&mut self, other: Self) {
                for (elem, &val) in self.0.iter_mut().zip(other.0.iter()) {
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
