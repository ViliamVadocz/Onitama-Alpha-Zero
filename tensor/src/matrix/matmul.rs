use super::*;

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Copy,
    [(); R * C]: ,
{
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
        Matrix(data)
    }
}
