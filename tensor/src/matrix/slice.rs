use super::*;

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
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
        Matrix(data)
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
        Matrix(data)
    }
}
