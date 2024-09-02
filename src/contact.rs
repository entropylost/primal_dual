use super::*;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rod {
    pub(crate) normal: RVector3,
    pub(crate) stiffness: Real,
    pub(crate) length: Real,
}
impl Constraint<2, 1> for Rod {
    fn value(&self, [pi, pj]: [Position; 2]) -> Scalar {
        let value = (self.normal * (pi.linear - pj.linear)).into_scalar() - self.length;
        Scalar::new(value)
    }
    fn gradient(&self, positions: [Position; 2]) -> [Gradient<1>; 2] {
        [
            Split::from_linear(self.normal),
            Split::from_linear(-self.normal),
        ]
    }
    fn stiffness(&self) -> Scalar {
        Scalar::new(self.stiffness)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Contact {
    pub(crate) normal: RVector3,
    pub(crate) stiffness: Real,
    pub(crate) length: Real,
}
impl Constraint<2, 1> for Contact {
    fn value(&self, [pi, pj]: [Position; 2]) -> Scalar {
        let value = (self.normal * (pi.linear - pj.linear)).into_scalar() - self.length;
        Scalar::new(value.min(0.0))
    }
    fn gradient(&self, positions: [Position; 2]) -> [Gradient<1>; 2] {
        [
            Split::from_linear(self.normal),
            Split::from_linear(-self.normal),
        ]
    }
    fn stiffness(&self) -> Scalar {
        Scalar::new(self.stiffness)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RadialContact {
    pub(crate) stiffness: Real,
    pub(crate) length: Real,
}
impl Constraint<2, 1> for RadialContact {
    fn value(&self, [pi, pj]: [Position; 2]) -> Scalar {
        let value = (pi.linear - pj.linear).norm() - self.length;
        Scalar::new(value.min(0.0))
    }
    fn gradient(&self, [pi, pj]: [Position; 2]) -> [Gradient<1>; 2] {
        let normal = (pi.linear - pj.linear).normalize().transpose();
        [Split::from_linear(normal), Split::from_linear(-normal)]
    }
    fn stiffness(&self) -> Scalar {
        Scalar::new(self.stiffness)
    }
}
