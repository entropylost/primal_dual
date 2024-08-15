use super::*;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rod {
    pub(crate) normal: Vector3,
    pub(crate) stiffness: f32,
    pub(crate) length: f32,
}
impl Constraint<2> for Rod {
    fn potential(&self, [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi.linear - pj.linear).dot(&self.normal) - self.length;
        value * self.stiffness
    }
    fn force(&self, p: [Position; 2]) -> [Force; 2] {
        let value = self.potential(p);
        [
            -Split::from_linear(value * self.normal),
            Split::from_linear(value * self.normal),
        ]
    }
    fn grad2_diag(&self, _p: [Position; 2]) -> [Split<Vector3, Vector3>; 2] {
        [Split::from_linear((self.normal * self.stiffness * self.normal.transpose()).diagonal()); 2]
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Contact {
    pub(crate) normal: Vector3,
    pub(crate) stiffness: f32,
    pub(crate) length: f32,
}
impl Constraint<2> for Contact {
    fn potential(&self, [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi.linear - pj.linear).dot(&self.normal) - self.length;
        (value * self.stiffness).min(0.0)
    }
    fn force(&self, p: [Position; 2]) -> [Force; 2] {
        let value = self.potential(p);
        [
            -Split::from_linear(value * self.normal),
            Split::from_linear(value * self.normal),
        ]
    }
    fn grad2_diag(&self, p: [Position; 2]) -> [Split<Vector3, Vector3>; 2] {
        if self.potential(p) < 0.0 {
            [Split::from_linear(
                (self.normal * self.stiffness * self.normal.transpose()).diagonal(),
            ); 2]
        } else {
            [Split::default(); 2]
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RadialContact {
    pub(crate) stiffness: f32,
    pub(crate) length: f32,
}
impl Constraint<2> for RadialContact {
    fn potential(&self, [pi, pj]: [Position; 2]) -> f32 {
        let value = (pi.linear - pj.linear).norm() - self.length;
        (value * self.stiffness).min(0.0)
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let normal = (pi.linear - pj.linear).normalize();
        let value = self.potential(p);
        [
            -Split::from_linear(value * normal),
            Split::from_linear(value * normal),
        ]
    }
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vector3, Vector3>; 2] {
        let normal = (pi.linear - pj.linear).normalize();

        if self.potential(p) < 0.0 {
            [Split::from_linear((normal * self.stiffness * normal.transpose()).diagonal()); 2]
        } else {
            [Split::default(); 2]
        }
    }
}
