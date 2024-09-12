use super::*;

#[derive(Debug, Clone, Copy)]
pub struct CosseratRod {
    pub radius: Real,
    pub young_modulus: Real,
    pub shear_modulus: Real,
    pub length: Real,
    pub rest_rotation: Rotation,
}
impl CosseratRod {
    pub fn resting_state(
        rod_radius: Real,
        young_modulus: Real,
        shear_modulus: Real,
        [pi, pj]: [Position; 2],
    ) -> Self {
        let length = (pj.linear - pi.linear).norm();
        let delta = pj.linear - pi.linear;
        let rest_rotation = delta.y.atan2(delta.x);
        Self {
            radius: rod_radius,
            young_modulus,
            shear_modulus,
            length,
            rest_rotation,
        }
    }
    fn stretch_shear(self) -> Vector {
        let s = PI * self.radius.powi(2);
        let a = 5.0 / 6.0 * s;
        Vector::new(self.young_modulus * s, self.shear_modulus * a)
    }
    fn bend_twist(self) -> Scalar {
        let i = PI * self.radius.powi(4) / 4.0;
        let j = PI * self.radius.powi(4) / 2.0;
        Scalar::new(self.young_modulus * i)
    }
    fn center_rotation(self, [pi, pj]: [Position; 2]) -> Rotation {
        (pi.angular + pj.angular) / 2.0
    }
    fn center_rotation_gradient(self, [pi, pj]: [Position; 2]) -> Real {
        0.5
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CosseratStretchShear {
    pub rod: CosseratRod,
}
impl Deref for CosseratStretchShear {
    type Target = CosseratRod;
    fn deref(&self) -> &Self::Target {
        &self.rod
    }
}

impl CosseratStretchShear {
    fn strain_measure(self, p @ [pi, pj]: [Position; 2]) -> Vector {
        1.0 / self.length
            * (rotation_matrix(self.center_rotation(p) + self.rest_rotation).transpose()
                * (pj.linear - pi.linear))
            - Vector::x()
    }
    // Wrt. the first position
    fn strain_gradient_lin(self, p: [Position; 2]) -> MatrixV {
        -1.0 / self.length
            * rotation_matrix(self.center_rotation(p) + self.rest_rotation).transpose()
    }
    fn strain_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Vector {
        let qij = self.center_rotation(p);
        let dqij = self.center_rotation_gradient(p);
        let rot_grad = dqij * rotation_matrix_gradient(qij + self.rest_rotation).transpose();
        1.0 / self.length * rot_grad * (pj.linear - pi.linear)
    }
}

impl Constraint<2, 2> for CosseratStretchShear {
    fn value(&self, p: [Position; 2]) -> Vector {
        self.strain_measure(p)
    }
    fn gradient(&self, p: [Position; 2]) -> [Gradient<2>; 2] {
        let grad_lin = self.strain_gradient_lin(p);
        let grad_ang = self.strain_gradient_ang(p);
        [
            Split::new(grad_lin, grad_ang),
            Split::new(-grad_lin, grad_ang),
        ]
    }
    fn stiffness(&self) -> Vector {
        self.stretch_shear() * self.length
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CosseratBendTwist {
    pub rod: CosseratRod,
}
impl Deref for CosseratBendTwist {
    type Target = CosseratRod;
    fn deref(&self) -> &Self::Target {
        &self.rod
    }
}

impl CosseratBendTwist {
    fn darboux_vector(self, p @ [pi, pj]: [Position; 2]) -> Real {
        // cos(th / 2.0) + k * sin(th / 2.0) - cos(phi / 2.0) - k * sin(phi / 2.0)
        //
        2.0 / self.length * ((pj.angular - pi.angular) / 2.0).sin()
    }
    fn darboux_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Real {
        -1.0 / self.length * ((pj.angular - pi.angular) / 2.0).cos()
    }
}

impl Constraint<2, 1> for CosseratBendTwist {
    fn value(&self, p: [Position; 2]) -> Scalar {
        Scalar::new(self.darboux_vector(p))
    }
    fn gradient(&self, p: [Position; 2]) -> [Gradient<1>; 2] {
        let grad_ang = Scalar::new(self.darboux_gradient_ang(p));
        [
            Split::from_angular(grad_ang),
            Split::from_angular(-grad_ang),
        ]
    }
    fn stiffness(&self) -> Scalar {
        self.bend_twist() * self.length
    }
}
