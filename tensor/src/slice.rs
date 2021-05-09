use super::*;

impl<T, const L: usize> Tensor1<T, L>
where
    T: Default + Copy + Debug,
{
    /// Get a slice of the tensor.
    pub fn slice<const NEW_L: usize>(&self, offset: usize) -> Tensor1<T, NEW_L> {
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
    pub fn slice_with_pad<const NEW_L: usize>(&self, offset: isize) -> Tensor1<T, NEW_L> {
        let mut data = [T::default(); NEW_L];
        for (i, elem) in data.iter_mut().enumerate() {
            let i = i as isize + offset;
            if i < 0 {
                continue;
            }
            let i = i as usize;
            if i >= L {
                break;
            }
            *elem = self.0[i];
        }
        Tensor1(data)
    }
}

impl<T, const C: usize, const R: usize> Tensor2<T, C, R>
where
    T: Default + Copy + Debug,
    [(); C * R]: ,
{
    /// Get a slice of the tensor.
    pub fn slice<const NEW_C: usize, const NEW_R: usize>(
        &self,
        col_offset: usize,
        row_offset: usize,
    ) -> Tensor2<T, NEW_C, NEW_R>
    where
        [(); NEW_C * NEW_R]: ,
    {
        assert!(col_offset + NEW_C <= C);
        assert!(row_offset + NEW_R <= R);
        let mut data = [T::default(); NEW_C * NEW_R];
        for (i, elem) in data.iter_mut().enumerate() {
            let c = i % NEW_C + col_offset;
            let r = i / NEW_C + row_offset;
            *elem = self.0[r * C + c];
        }
        Tensor2(data)
    }

    /// Get a slice of the matrix with padding if needed.
    /// Allow slices which do not entirely fit inside the tensor.
    /// Padded with the default value.
    pub fn slice_with_pad<const NEW_C: usize, const NEW_R: usize>(
        &self,
        col_offset: isize,
        row_offset: isize,
    ) -> Tensor2<T, NEW_C, NEW_R>
    where
        [(); NEW_C * NEW_R]: ,
    {
        let mut data = [T::default(); NEW_C * NEW_R];
        let new_c = NEW_C as isize;
        for (i, elem) in data.iter_mut().enumerate() {
            let i = i as isize;
            let c = i % new_c + col_offset;
            let r = i / new_c + row_offset;
            if c < 0 || r < 0 {
                continue;
            }
            let c = c as usize;
            let r = r as usize;
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
        &self,
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
        &self,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_tensor1() {
        let a = Vector::new([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(a.slice::<4>(3), Vector::new([3, 4, 5, 6]));
        assert_eq!(
            a.slice_with_pad::<10>(-1),
            Vector::new([0, 0, 1, 2, 3, 4, 5, 6, 7, 0])
        );
    }

    #[test]
    fn slice_tensor2() {
        let a: Matrix<_, 3, 3> = Matrix::new([0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(a.slice::<2, 2>(1, 1), Matrix::new([4, 5, 7, 8]));
        assert_eq!(a.slice::<3, 1>(0, 1), Matrix::new([3, 4, 5]));
        assert_eq!(a.slice::<1, 3>(2, 0), Matrix::new([2, 5, 8]));
        assert_eq!(a.slice_with_pad::<2, 2>(2, -1), Matrix::new([0, 0, 2, 0]));
        assert_eq!(
            a.slice_with_pad::<2, 4>(-1, 0),
            Matrix::new([0, 0, 0, 3, 0, 6, 0, 0])
        );
        assert_eq!(
            a.slice_with_pad::<3, 3>(1, 1),
            Matrix::new([4, 5, 0, 7, 8, 0, 0, 0, 0])
        );
    }

    #[test]
    fn slice_tensor3() {
        #[rustfmt::skip]
        let a: Tensor3<_, 2, 2, 2> = Tensor3::new([
            0, 1,
            2, 3,
            
            4, 5,
            6, 7  
        ]);
        assert_eq!(a.slice::<2, 2, 1>(0, 0, 0), Tensor3::new([0, 1, 2, 3]));
        assert_eq!(a.slice::<2, 1, 2>(0, 0, 0), Tensor3::new([0, 1, 4, 5]));
        assert_eq!(a.slice::<1, 2, 2>(0, 0, 0), Tensor3::new([0, 2, 4, 6]));
        assert_eq!(a.slice::<1, 1, 2>(1, 0, 0), Tensor3::new([1, 5]));
        assert_eq!(a.slice::<1, 2, 1>(0, 0, 1), Tensor3::new([4, 6]));
        assert_eq!(a.slice::<2, 1, 1>(0, 1, 0), Tensor3::new([2, 3]));
        assert_eq!(a.slice::<2, 1, 1>(0, 1, 0), Tensor3::new([2, 3]));
        assert_eq!(
            a.slice_with_pad::<2, 2, 2>(1, 0, 0),
            Tensor3::new([1, 0, 3, 0, 5, 0, 7, 0])
        );
        assert_eq!(
            a.slice_with_pad::<2, 2, 2>(0, 1, 0),
            Tensor3::new([2, 3, 0, 0, 6, 7, 0, 0])
        );
        assert_eq!(
            a.slice_with_pad::<2, 2, 2>(0, 0, 1),
            Tensor3::new([4, 5, 6, 7, 0, 0, 0, 0])
        );
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use super::*;

    #[bench]
    fn slice(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.slice::<18, 18, 18>(1, 1, 1));
    }

    #[bench]
    fn slice_with_pad(ben: &mut Bencher) {
        let a = Tensor3::<f64, 20, 20, 20>::rand(rand_distr::Uniform::new(-1., 1.));
        ben.iter(|| a.slice_with_pad::<22, 22, 22>(-1, -1, -1));
    }
}
