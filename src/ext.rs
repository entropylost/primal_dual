use std::ops::AddAssign;

use iter_fixed::IntoIteratorFixed;
use nalgebra::{SMatrix, SVector, Scalar};
use num_traits::Zero;

pub trait ArraySum {
    type Element;
    fn sum(self) -> Self::Element;
}

impl<const N: usize, const R: usize, const C: usize, T: Scalar + AddAssign<T> + Zero> ArraySum
    for [SMatrix<T, R, C>; N]
{
    type Element = SMatrix<T, R, C>;

    fn sum(self) -> Self::Element {
        let mut total = SMatrix::zeros();
        for item in self {
            total += item;
        }
        total
    }
}

pub trait ZipMap<const N: usize> {
    type Item1;
    type Item2;
    fn map<U>(self, f: impl FnMut(Self::Item1, Self::Item2) -> U) -> [U; N];
}

impl<const N: usize, T, S> ZipMap<N> for ([T; N], [S; N]) {
    type Item1 = T;
    type Item2 = S;

    fn map<U>(self, mut f: impl FnMut(Self::Item1, Self::Item2) -> U) -> [U; N] {
        self.0
            .into_iter_fixed()
            .zip(self.1.into_iter_fixed())
            .map(|(a, b)| f(a, b))
            .collect()
    }
}

pub fn diag<T: Scalar + Zero, const N: usize>(v: SVector<T, N>) -> SMatrix<T, N, N> {
    SMatrix::from_diagonal(&v)
}
