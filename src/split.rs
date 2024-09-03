use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use nalgebra::SMatrix;

use crate::Real;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Split<A, B> {
    pub(crate) linear: A,
    pub(crate) angular: B,
}
impl<A, B> Split<A, B> {
    pub(crate) fn new(linear: A, angular: B) -> Self {
        Self { linear, angular }
    }
    pub(crate) fn map_linear<C>(self, f: impl FnOnce(A) -> C) -> Split<C, B> {
        Split {
            linear: f(self.linear),
            angular: self.angular,
        }
    }
    pub(crate) fn map_angular<C>(self, f: impl FnOnce(B) -> C) -> Split<A, C> {
        Split {
            linear: self.linear,
            angular: f(self.angular),
        }
    }
}
impl<A, B: Default> Split<A, B> {
    pub(crate) fn from_linear(linear: A) -> Self {
        Self {
            linear,
            angular: Default::default(),
        }
    }
}
impl<A: Default, B> Split<A, B> {
    pub(crate) fn from_angular(angular: B) -> Self {
        Self {
            linear: Default::default(),
            angular,
        }
    }
}
impl<A, B> Neg for Split<A, B>
where
    A: Neg<Output = A>,
    B: Neg<Output = B>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            linear: -self.linear,
            angular: -self.angular,
        }
    }
}
impl<A, B> Add<Self> for Split<A, B>
where
    A: Add<A, Output = A>,
    B: Add<B, Output = B>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular,
        }
    }
}
impl<A, B> Sub<Self> for Split<A, B>
where
    A: Sub<A, Output = A>,
    B: Sub<B, Output = B>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular,
        }
    }
}
impl<A, B, C, D> Mul<Split<C, D>> for Split<A, B>
where
    A: Mul<C>,
    B: Mul<D>,
{
    type Output = Split<A::Output, B::Output>;
    fn mul(self, rhs: Split<C, D>) -> Self::Output {
        Split {
            linear: self.linear * rhs.linear,
            angular: self.angular * rhs.angular,
        }
    }
}
impl<A, B, C, D> Div<Split<C, D>> for Split<A, B>
where
    A: Div<C>,
    B: Div<D>,
{
    type Output = Split<A::Output, B::Output>;
    fn div(self, rhs: Split<C, D>) -> Self::Output {
        Split {
            linear: self.linear / rhs.linear,
            angular: self.angular / rhs.angular,
        }
    }
}

impl<A, B> AddAssign<Self> for Split<A, B>
where
    A: AddAssign<A>,
    B: AddAssign<B>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.linear += rhs.linear;
        self.angular += rhs.angular;
    }
}
impl<A, B> SubAssign<Self> for Split<A, B>
where
    A: SubAssign<A>,
    B: SubAssign<B>,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.linear -= rhs.linear;
        self.angular -= rhs.angular;
    }
}

impl<A, B> MulAssign<Self> for Split<A, B>
where
    A: MulAssign<A>,
    B: MulAssign<B>,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.linear *= rhs.linear;
        self.angular *= rhs.angular;
    }
}
impl<A, B> DivAssign<Self> for Split<A, B>
where
    A: DivAssign<A>,
    B: DivAssign<B>,
{
    fn div_assign(&mut self, rhs: Self) {
        self.linear /= rhs.linear;
        self.angular /= rhs.angular;
    }
}

impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize> Mul<Real>
    for Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>
{
    type Output = Self;
    fn mul(self, rhs: Real) -> Self {
        Self {
            linear: self.linear * rhs,
            angular: self.angular * rhs,
        }
    }
}
impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize> Div<Real>
    for Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>
{
    type Output = Self;
    fn div(self, rhs: Real) -> Self {
        Self {
            linear: self.linear / rhs,
            angular: self.angular / rhs,
        }
    }
}
impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize>
    Mul<Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>> for Real
{
    type Output = Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>;
    fn mul(self, rhs: Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>) -> Self::Output {
        Split {
            linear: self * rhs.linear,
            angular: self * rhs.angular,
        }
    }
}
impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize>
    Split<SMatrix<Real, R1, C1>, SMatrix<Real, R2, C2>>
{
    pub fn component_mul(self, rhs: Self) -> Self {
        Self {
            linear: self.linear.component_mul(&rhs.linear),
            angular: self.angular.component_mul(&rhs.angular),
        }
    }
    pub fn component_div(self, rhs: Self) -> Self {
        Self {
            linear: self.linear.component_div(&rhs.linear),
            angular: self.angular.component_div(&rhs.angular),
        }
    }
}
impl<A: Reciprocal, B: Reciprocal> Split<A, B> {
    pub fn recip(self) -> Self {
        Split {
            linear: self.linear.reciprocal(),
            angular: self.angular.reciprocal(),
        }
    }
}

pub trait Invertible {
    fn inverse(self) -> Self;
}
impl Invertible for Real {
    fn inverse(self) -> Self {
        self.recip()
    }
}
impl<const N: usize> Invertible for SMatrix<Real, N, N> {
    fn inverse(self) -> Self {
        self.try_inverse().unwrap()
    }
}
impl<A: Invertible, B: Invertible> Invertible for Split<A, B> {
    fn inverse(self) -> Self {
        Split {
            linear: self.linear.inverse(),
            angular: self.angular.inverse(),
        }
    }
}

pub trait Reciprocal {
    fn reciprocal(self) -> Self;
}
impl Reciprocal for Real {
    fn reciprocal(self) -> Self {
        self.recip()
    }
}
impl<const R: usize, const C: usize> Reciprocal for SMatrix<Real, R, C> {
    fn reciprocal(self) -> Self {
        self.map(Real::recip)
    }
}
