use super::*;
use std::convert::TryInto;

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
    // Get an iterator over the rows of a Matrix.
    pub fn rows(&self) -> Rows<'_, T, R, C> {
        Rows {
            data: &self.0,
            row: 0,
        }
    }

    // Get an iterator over the columns of a Matrix.
    pub fn cols(&self) -> Cols<'_, T, R, C> {
        Cols {
            data: &self.0,
            col: 0,
        }
    }
}
