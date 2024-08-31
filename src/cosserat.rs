use super::*;

#[derive(Debug, Clone, Copy)]
pub struct CosseratRod {
    pub radius: f32,
    pub young_modulus: f32,
    pub shear_modulus: f32,
    pub length: f32,
    pub rest_rotation: UnitQuaternion,
}
impl CosseratRod {
    pub fn resting_state(
        rod_radius: f32,
        young_modulus: f32,
        shear_modulus: f32,
        [pi, pj]: [Position; 2],
    ) -> Self {
        let length = (pj.linear - pi.linear).norm();
        let rest_rotation =
            UnitQuaternion::rotation_between(&Vector3::z(), &(pj.linear - pi.linear)).unwrap();
        Self {
            radius: rod_radius,
            young_modulus,
            shear_modulus,
            length,
            rest_rotation,
        }
    }
    fn stretch_shear_diag(self) -> Vector3 {
        let s = PI * self.radius.powi(2);
        let a = 5.0 / 6.0 * s;
        Vector3::new(
            self.shear_modulus * a,
            self.shear_modulus * a,
            self.young_modulus * s,
        )
    }
    fn stretch_shear_a(self) -> f32 {
        let s = PI * self.radius.powi(2);
        let a = 5.0 / 6.0 * s;
        self.shear_modulus * a
    }
    fn stretch_shear_b(self) -> f32 {
        let s = PI * self.radius.powi(2);
        self.young_modulus * s - self.stretch_shear_a()
    }
    fn stretch_shear(self) -> Matrix3 {
        Matrix3::from_diagonal(&self.stretch_shear_diag())
    }
    fn bend_twist_diag(self) -> Vector3 {
        let i = PI * self.radius.powi(4) / 4.0;
        let j = PI * self.radius.powi(4) / 2.0;
        Vector3::new(
            self.young_modulus * i,
            self.young_modulus * i,
            self.shear_modulus * j,
        )
    }
    fn bend_twist(self) -> Matrix3 {
        Matrix3::from_diagonal(&self.bend_twist_diag())
    }
    fn center_rotation(self, [pi, pj]: [Position; 2]) -> UnitQuaternion {
        pi.angular.nlerp(&pj.angular, 0.5)
    }
    fn center_rotation_gradient(self, [pi, pj]: [Position; 2]) -> Matrix4 {
        let qm = pi.angular.lerp(&pj.angular, 0.5);
        let qij = UnitQuaternion::from_quaternion(qm);
        (Matrix4::identity() - qij.as_vector() * qij.as_vector().transpose()) / qm.norm()
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

// p * q = rmul_mat(q) * p
fn rmul_mat(q: Quaternion) -> Matrix4 {
    stack![
        Matrix3::from_diagonal_element(q.scalar()) - cross_matrix(q.vector().into()), q.vector();
        -q.vector().transpose(), na::Matrix1::<f32>::new(q.scalar())
    ]
}

// p * q = lmul_mat(p) * q
fn lmul_mat(p: Quaternion) -> Matrix4 {
    stack![
        Matrix3::from_diagonal_element(p.scalar()) + cross_matrix(p.vector().into()), p.vector();
         -p.vector().transpose(), na::Matrix1::<f32>::new(p.scalar())
    ]
}

#[test]
fn test_quat_mul() {
    let p = Quaternion::new(1.0, 6.0, -2.0, 3.0);
    let q = Quaternion::new(7.0, -3.0, 5.0, 10.0);
    assert_eq!((p * q).coords, lmul_mat(p) * q.coords);
    assert_eq!((p * q).coords, rmul_mat(q) * p.coords);
}

// v.cross(w) = cross_matrix(v) * w
fn cross_matrix(v: Vector3) -> Matrix3 {
    matrix![
        0.0, -v.z, v.y;
        v.z, 0.0, -v.x;
        -v.y, v.x, 0.0
    ]
}

impl CosseratStretchShear {
    fn strain_measure(self, p @ [pi, pj]: [Position; 2]) -> Vector3 {
        1.0 / self.length
            * ((self.center_rotation(p) * self.rest_rotation).inverse() * (pj.linear - pi.linear))
            - Vector3::z()
    }
    // Wrt. the first position
    fn strain_gradient_lin(self, p: [Position; 2]) -> Matrix3 {
        -1.0 / self.length
            * (self.center_rotation(p) * self.rest_rotation)
                .to_rotation_matrix()
                .matrix()
                .transpose()
    }
    fn strain_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Matrix3x4 {
        /*let qm = pi.angular.lerp(&pj.angular, 0.5);
        let qij = UnitQuaternion::from_quaternion(qm);

        let qpart = (Quaternion::from_imag(pj.linear - pi.linear) * *qij);
        let v = qpart.coords;

        let a = qij.to_rotation_matrix().matrix()
            * matrix![
                -v.w, -v.z, v.y, v.x;
                v.z, -v.w, -v.x, v.y;
                -v.y, v.x, -v.w, v.z;
            ];

        let b = (pj.linear - pi.linear) * qij.as_vector().transpose();

        1.0 / self.length / qm.norm()
            * (qij * self.rest_rotation)
                .to_rotation_matrix()
                .matrix()
                .transpose()
            * (a - b)
        */
        let qij = *self.center_rotation(p);
        let dqij = self.center_rotation_gradient(p);
        1.0 / self.length
            * self.rest_rotation.to_rotation_matrix().matrix().transpose()
            * rmul_mat(Quaternion::from_imag(pj.linear - pi.linear) * qij).fixed_view::<3, 4>(0, 0)
            * Matrix4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
            * dqij
    }
}

impl Constraint<2, 3> for CosseratStretchShear {
    fn value(&self, p: [Position; 2]) -> Vector3 {
        let strain = self.strain_measure(p);
        strain * (self.length / 2.0).sqrt()
    }
    fn gradient(&self, p @ [pi, pj]: [Position; 2]) -> [Gradient<3>; 2] {
        let strain = self.strain_measure(p);
        let grad_lin = self.strain_gradient_lin(p);
        let grad_ang = self.strain_gradient_ang(p);
        let grad = Split::new(grad_lin, grad_ang);
        [grad, -grad]
    }
    fn stiffness(&self) -> SVector<f32, 3> {
        self.stretch_shear_diag()
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
    fn darboux_vector(self, p @ [pi, pj]: [Position; 2]) -> Vector3 {
        2.0 / self.length
            * (*self.center_rotation(p).conjugate() * (*pj.angular - *pi.angular)).imag()
    }
    fn darboux_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Matrix3x4 {
        /*let qm = pi.angular.lerp(&pj.angular, 0.5);
        let qij = UnitQuaternion::from_quaternion(qm);
        let p = qij.coords;

        let qijc_mat = matrix![
            p.w, p.z, -p.y, -p.x;
            -p.z, p.w, p.x, -p.y;
            p.y, -p.x, p.w, -p.z;
        ];
        let qdiff = *pj.angular - *pi.angular;
        let v = qdiff.coords;
        let a = matrix![
            -v.w, -v.z, v.y, v.x;
            v.z, -v.w, -v.x, v.y;
            -v.y, v.x, -v.w, v.z
        ];
        let b = qijc_mat * v * p.transpose();
        let c = qijc_mat;
        ((a - b) / qm.norm() - 2.0 * c) / self.length*/
        let dqij = self.center_rotation_gradient(p);
        let gradient = 2.0 / self.length
            * (1.0 / 2.0
                * rmul_mat(*pj.angular - *pi.angular)
                * Matrix4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
                * dqij
                - lmul_mat(*self.center_rotation(p).conjugate()));

        gradient.fixed_view::<3, 4>(0, 0).into_owned()
    }
}

impl Constraint<2, 3> for CosseratBendTwist {
    fn value(&self, p: [Position; 2]) -> Vector3 {
        let darboux = self.darboux_vector(p);
        darboux * (self.length / 2.0).sqrt()
    }
    fn gradient(&self, p @ [pi, pj]: [Position; 2]) -> [Gradient<3>; 2] {
        let grad_lin = Matrix3::zeros();
        let grad_ang = self.darboux_gradient_ang(p);
        let grad = Split::new(grad_lin, grad_ang);
        [grad, -grad]
    }
    fn stiffness(&self) -> SVector<f32, 3> {
        self.bend_twist_diag()
    }
}
