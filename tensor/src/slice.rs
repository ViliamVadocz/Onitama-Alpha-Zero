use super::*;
use std::cmp::{max, min};

impl<T, const L: usize> Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    /// Get a slice of the tensor.
    pub fn slice<const NEW_L: usize>(self, offset: usize) -> Tensor1<T, NEW_L> {
        assert!(offset + NEW_L <= L);
        let mut data = [T::default(); NEW_L];
        for (elem, &val) in data.iter_mut().zip(self.0[offset..(offset + NEW_L)].iter()) {
            *elem = val;
        }
        Tensor1(data)
    }

    /// Get a slice with padding if needed.
    /// Allow slices which do not entirely fit inside the tensor.
    /// Padded with the default value.
    pub fn slice_with_pad<const NEW_L: usize>(self, offset: isize) -> Tensor1<T, NEW_L> {
        let mut data = [T::default(); NEW_L];
        let start = max(offset, 0) as usize;
        let end = min(offset + NEW_L as isize, L as isize) as usize;
        for (elem, &val) in data.iter_mut().zip(self.0[start..end].iter()) {
            *elem = val;
        }
        Tensor1(data)
    }
}

impl<T, const R: usize, const C: usize> Tensor2<T, R, C>
where
    T: Default + Copy + Debug,
    [(); R * C]: ,
{
    /// Get a slice of the tensor.
    pub fn slice<const NEW_R: usize, const NEW_C: usize>(
        self,
        row_offset: usize,
        col_offset: usize,
    ) -> Tensor2<T, NEW_R, NEW_C>
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
        Tensor2(data)
    }

    /// Get a slice of the matrix with padding if needed.
    /// Allow slices which do not entirely fit inside the tensor.
    /// Padded with the default value.
    pub fn slice_with_pad<const NEW_R: usize, const NEW_C: usize>(
        self,
        row_offset: isize,
        col_offset: isize,
    ) -> Tensor2<T, NEW_R, NEW_C>
    where
        [(); NEW_R * NEW_C]: ,
    {
        let mut data = [T::default(); NEW_R * NEW_C];
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
        Tensor2(data)
    }
}

impl<T, const D1: usize, const D2: usize, const D3: usize> Tensor3<T, D1, D2, D3>
where
    T: Default + Copy + Debug,
    [(); D1 * D2 * D3]: ,
{
    /// Get a slice of the tensor.
    pub fn slice<const NEW_D1: usize, const NEW_D2: usize, const NEW_D3: usize>(
        self,
        d1_offset: usize,
        d2_offset: usize,
        d3_offset: usize,
    ) -> Tensor3<T, NEW_D1, NEW_D2, NEW_D3>
    where
        [(); NEW_D1 * NEW_D2 * NEW_D3]: ,
    {
        assert!(d1_offset + NEW_D1 <= D1);
        assert!(d2_offset + NEW_D2 <= D2);
        assert!(d3_offset + NEW_D3 <= D3);
        let mut data = [T::default(); NEW_D1 * NEW_D2 * NEW_D3];
        for (i, elem) in data.iter_mut().enumerate() {
            let d1 = i % NEW_D1 + d1_offset;
            let d2 = (i / NEW_D1) % NEW_D2 + d2_offset;
            let d3 = (i / (NEW_D1 * NEW_D2)) + d3_offset;
            *elem = self.0[d1 + d2 * D1 + d3 * (D1 * D2)];
        }
        Tensor3(data)
    }

    /// Get a slice of the matrix with padding if needed.
    /// Allow slices which do not entirely fit inside the tensor.
    /// Padded with the default value.
    pub fn slice_with_pad<const NEW_D1: usize, const NEW_D2: usize, const NEW_D3: usize>(
        self,
        d1_offset: isize,
        d2_offset: isize,
        d3_offset: isize,
    ) -> Tensor3<T, NEW_D1, NEW_D2, NEW_D3>
    where
        [(); NEW_D1 * NEW_D2 * NEW_D3]: ,
    {
        let mut data = [T::default(); NEW_D1 * NEW_D2 * NEW_D3];
        let new_d1 = NEW_D1 as isize;
        let new_d2 = NEW_D2 as isize;
        for (i, elem) in data.iter_mut().enumerate() {
            let i = i as isize;
            let d1 = i % new_d1 + d1_offset;
            let d2 = (i / new_d1) % new_d2 + d2_offset;
            let d3 = (i / (new_d1 * new_d2)) + d3_offset;
            if d1 < 0 || d2 < 0 || d3 < 0 {
                continue;
            }
            let d1 = d1 as usize;
            let d2 = d2 as usize;
            let d3 = d3 as usize;
            if d1 >= D1 || d2 >= D2 || d3 >= D3 {
                continue;
            }
            *elem = self.0[d1 + d2 * D1 + d3 * (D1 * D2)];
        }
        Tensor3(data)
    }
}
