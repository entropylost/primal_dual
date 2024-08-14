use nalgebra::{self as na, matrix, stack, vector};
use std::{
    f32::consts::PI,
    ops::{Add, Deref, Mul, Neg, Sub},
};

type Vec3 = na::Vector3<f32>;
type RVec3 = na::RowVector3<f32>;
type Mat3 = na::Matrix3<f32>;
type Mat3x4 = na::Matrix3x4<f32>;
type Mat4x3 = na::Matrix4x3<f32>;
type Mat4 = na::Matrix4<f32>;
type Quat = na::Quaternion<f32>;
type UQuat = na::UnitQuaternion<f32>;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Split<A, B> {
    linear: A,
    angular: B,
}
impl<A, B> Split<A, B> {
    fn new(linear: A, angular: B) -> Self {
        Self { linear, angular }
    }
    fn map_linear<C>(self, f: impl FnOnce(A) -> C) -> Split<C, B> {
        Split {
            linear: f(self.linear),
            angular: self.angular,
        }
    }
    fn map_angular<C>(self, f: impl FnOnce(B) -> C) -> Split<A, C> {
        Split {
            linear: self.linear,
            angular: f(self.angular),
        }
    }
}
impl<A, B: Default> Split<A, B> {
    fn from_linear(linear: A) -> Self {
        Self {
            linear,
            angular: Default::default(),
        }
    }
}
impl<A: Default, B> Split<A, B> {
    fn from_angular(angular: B) -> Self {
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

type Position = Split<Vec3, UQuat>;
type Displacement = Split<Vec3, Quat>;
type Velocity = Split<Vec3, Vec3>;
type Force = Split<Vec3, Vec3>;
type Mass = Split<f32, Mat3>;

impl Position {
    fn rotation_map(self) -> Mat4x3 {
        let q = self.angular.quaternion().as_vector();
        1.0 / 2.0
            * matrix![
                -q.x, -q.y, -q.z;
                q.w, q.z, -q.y;
                -q.z, q.w, q.x;
                q.y, -q.x, q.w
            ]
    }
    fn kinematic_map(self) -> Split<Mat3, Mat4x3> {
        Split::new(Mat3::identity(), self.rotation_map())
    }
    fn map_velocity(self, velocity: Velocity) -> Displacement {
        Displacement {
            linear: velocity.linear,
            // Should be equal to the kinematic map times the velocity.
            angular: (Quat::from_imag(velocity.angular) * *self.angular) / 2.0,
        }
    }
}

trait Constraint<const N: usize> {
    fn potential(&self, positions: [Position; N]) -> f32;
    fn force(&self, positions: [Position; N]) -> [Force; N];
    fn diag_grad2(&self, positions: [Position; N]) -> [Split<Vec3, Vec3>; N];
}

#[derive(Debug, Clone, Copy)]
struct Rod {
    normal: Vec3,
    stiffness: f32,
    length: f32,
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
    fn diag_grad2(&self, _p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        [Split::from_linear((self.normal * self.stiffness * self.normal.transpose()).diagonal()); 2]
    }
}

#[derive(Debug, Clone, Copy)]
struct Contact {
    normal: Vec3,
    stiffness: f32,
    length: f32,
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
    fn diag_grad2(&self, p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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
struct RadialContact {
    stiffness: f32,
    length: f32,
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
    fn diag_grad2(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        let normal = (pi.linear - pj.linear).normalize();

        if self.potential(p) < 0.0 {
            [Split::from_linear((normal * self.stiffness * normal.transpose()).diagonal()); 2]
        } else {
            [Split::default(); 2]
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratParameters {
    rod_radius: f32,
    young_modulus: f32,
    shear_modulus: f32,
    length: f32,
    rest_rotation: UQuat,
}
impl CosseratParameters {
    fn stretch_shear_diag(self) -> Vec3 {
        let s = PI * self.rod_radius.powi(2);
        let a = 5.0 / 6.0 * s;
        Vec3::new(
            self.shear_modulus * a,
            self.shear_modulus * a,
            self.young_modulus * s,
        )
    }
    fn stretch_shear(self) -> Mat3 {
        Mat3::from_diagonal(&self.stretch_shear_diag())
    }
    fn bend_twist_diag(self) -> Vec3 {
        let i = PI * self.rod_radius.powi(4) / 4.0;
        let j = PI * self.rod_radius.powi(4) / 2.0;
        Vec3::new(
            self.young_modulus * i,
            self.young_modulus * i,
            self.shear_modulus * j,
        )
    }
    fn bend_twist(self) -> Mat3 {
        Mat3::from_diagonal(&self.bend_twist_diag())
    }
    fn center_rotation(self, [pi, pj]: [Position; 2]) -> UQuat {
        pi.angular.nlerp(&pj.angular, 0.5)
    }
    fn center_rotation_gradient(self, [pi, pj]: [Position; 2]) -> Mat4 {
        let qm = pi.angular.lerp(&pj.angular, 0.5);
        let qij = UQuat::from_quaternion(qm);
        (Mat4::identity() - qij.as_vector() * qij.as_vector().transpose()) / qm.norm()
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratStretchShear {
    params: CosseratParameters,
}
impl Deref for CosseratStretchShear {
    type Target = CosseratParameters;
    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

// p * q = rmul_mat(q) * p
fn rmul_mat(q: Quat) -> Mat4 {
    stack![
        na::Matrix1::<f32>::new(q.scalar()), -q.vector().transpose();
        q.vector(), Mat3::from_diagonal_element(q.scalar()) - cross_matrix(q.vector().into())
    ]
}

// p * q = lmul_mat(p) * q
fn lmul_mat(p: Quat) -> Mat4 {
    stack![
        na::Matrix1::<f32>::new(p.scalar()), -p.vector().transpose();
        p.vector(), Mat3::from_diagonal_element(p.scalar()) + cross_matrix(p.vector().into())
    ]
}

// v.cross(w) = cross_matrix(v) * w
fn cross_matrix(v: Vec3) -> Mat3 {
    matrix![
        0.0, -v.z, v.y;
        v.z, 0.0, -v.x;
        -v.y, v.x, 0.0
    ]
}

impl CosseratStretchShear {
    fn strain_measure(self, p @ [pi, pj]: [Position; 2]) -> Vec3 {
        1.0 / self.length
            * ((self.center_rotation(p) * self.rest_rotation).inverse() * (pj.linear - pi.linear))
            - Vec3::z()
    }
    // Wrt. the first position
    // Also, this is the jacobian actually, so no transpose is needed in the later part.
    fn strain_gradient_lin(self, p: [Position; 2]) -> Mat3 {
        -1.0 / self.length
            * (self.center_rotation(p) * self.rest_rotation)
                .to_rotation_matrix()
                .matrix()
                .transpose()
    }
    fn strain_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Mat3x4 {
        let qij = *self.center_rotation(p);
        let dqij = self.center_rotation_gradient(p);
        1.0 / self.length
            * self.rest_rotation.to_rotation_matrix().matrix().transpose()
            * rmul_mat(Quat::from_imag(pj.linear - pi.linear) * qij).fixed_view::<3, 4>(1, 0)
            * Mat4::from_diagonal(&vector![1.0, -1.0, -1.0, -1.0])
            * dqij
    }
}

impl Constraint<2> for CosseratStretchShear {
    fn potential(&self, p: [Position; 2]) -> f32 {
        let strain = self.strain_measure(p);
        (self.length / 2.0 * strain.transpose() * self.stretch_shear() * strain).to_scalar()
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let strain = self.strain_measure(p);
        let force =
            -self.length * self.strain_gradient_lin(p).transpose() * self.stretch_shear() * strain;
        let torque =
            -self.length * self.strain_gradient_ang(p).transpose() * self.stretch_shear() * strain; // TODO: Finish.
        [
            Split::new(force, pi.rotation_map().transpose() * torque),
            -Split::new(force, pj.rotation_map().transpose() * torque),
        ]
    }
    fn diag_grad2(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        let lin = self.strain_gradient_lin(p);
        let ang = self.strain_gradient_ang(p);
        let jacobian_i = Split::new(lin, ang * pi.rotation_map());
        let diag_i = Split::new(
            jacobian_i.linear.transpose() * self.stretch_shear() * jacobian_i.linear,
            jacobian_i.angular.transpose() * self.stretch_shear() * jacobian_i.angular,
        );
        let diag_i = Split::new(diag_i.linear.diagonal(), diag_i.angular.diagonal());
        let jacobian_j = -Split::new(lin, ang * pj.rotation_map());
        let diag_j = Split::new(
            jacobian_j.linear.transpose() * self.stretch_shear() * jacobian_j.linear,
            jacobian_j.angular.transpose() * self.stretch_shear() * jacobian_j.angular,
        );
        let diag_j = Split::new(diag_j.linear.diagonal(), diag_j.angular.diagonal());
        [diag_i, diag_j]
    }
}

#[derive(Debug, Clone, Copy)]
struct CosseratBendTwist {
    params: CosseratParameters,
}
impl Deref for CosseratBendTwist {
    type Target = CosseratParameters;
    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl CosseratBendTwist {
    fn darboux_vector(self, p @ [pi, pj]: [Position; 2]) -> Vec3 {
        2.0 / self.length
            * (*self.center_rotation(p).conjugate() * (*pj.angular - *pi.angular)).imag()
    }
    fn darboux_gradient_ang(self, p @ [pi, pj]: [Position; 2]) -> Mat3x4 {
        let dqij = self.center_rotation_gradient(p);
        let gradient = -2.0 / self.length
            * (1.0 / 2.0
                * rmul_mat(*pj.angular - *pi.angular)
                * Mat4::from_diagonal(&vector![1.0, -1.0, -1.0, -1.0])
                * dqij
                - lmul_mat(*self.center_rotation(p).conjugate()));
        gradient.fixed_view::<3, 4>(1, 0).into_owned()
    }
}

impl Constraint<2> for CosseratBendTwist {
    fn potential(&self, p: [Position; 2]) -> f32 {
        let darboux = self.darboux_vector(p);
        (self.length / 2.0 * darboux.transpose() * self.bend_twist() * darboux).to_scalar()
    }
    fn force(&self, p @ [pi, pj]: [Position; 2]) -> [Force; 2] {
        let darboux = self.darboux_vector(p);
        let torque =
            -self.length * self.darboux_gradient_ang(p).transpose() * self.bend_twist() * darboux; // TODO: Finish.
        [
            Split::from_angular(pi.rotation_map().transpose() * torque),
            -Split::from_angular(pj.rotation_map().transpose() * torque),
        ]
    }
    fn diag_grad2(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
        let ang = self.darboux_gradient_ang(p);
        let jacobian_i = ang * pi.rotation_map();
        let diag_i = jacobian_i.transpose() * self.bend_twist() * jacobian_i;
        let diag_i = Split::from_angular(diag_i.diagonal());
        let jacobian_j = -ang * pj.rotation_map();
        let diag_j = jacobian_j.transpose() * self.bend_twist() * jacobian_j;
        let diag_j = Split::from_angular(diag_j.diagonal());
        [diag_i, diag_j]
    }
}

#[derive(Debug, Clone)]
struct Particles {
    position: Vec<Position>,
    last_position: Vec<Position>,
    velocity: Vec<Velocity>,
    last_velocity: Vec<Velocity>,
    mass: Vec<Mass>,
}
impl Particles {
    fn len(&self) -> usize {
        self.position.len()
    }
}

#[macroquad::main("main")]
async fn main() {
    use macroquad::prelude::*;
    let scaling = 50.0;

    let mut particles = Particles {
        position: vec![
            vec2(0.0, 0.0),
            vec2(4.0, 0.0),
            vec2(6.0, 0.0),
            vec2(7.0, 0.0),
        ],
        last_position: vec![vec2(0.0, 0.0); 4],
        velocity: vec![
            vec2(0.1, 0.0),
            vec2(0.0, 0.0),
            vec2(0.0, 0.0),
            vec2(0.0, 0.0),
        ],
        last_velocity: vec![vec2(0.0, 0.0); 4],
        mass: vec![1.0, 1.0, 1.0, 1.0],
    };

    let constraint_step = 0.1;

    loop {
        let offset = vec2(100.0, 100.0);
        clear_background(BLACK);

        if is_key_pressed(KeyCode::Period) || is_key_down(KeyCode::Space) {
            particles.last_position = particles.position.clone();
            particles.last_velocity = particles.velocity.clone();
            for i in 0..particles.len() {
                particles.position[i] += particles.velocity[i];
            }

            let mut contacts = vec![];

            for i in 0..particles.len() {
                for j in i + 1..particles.len() {
                    let pi = particles.position[i];
                    let pj = particles.position[j];
                    if pi.distance(pj) <= 1.0 {
                        contacts.push(Contact {
                            colliders: (i, j),
                            normal: (pi - pj).normalize(),
                            stiffness: 10000.0,
                        });
                    }
                }
            }
            for _iter in 0..100 {
                let mut forces = vec![Vec2::ZERO; particles.len()];
                let mut jacobian_diag = vec![Vec2::ZERO; particles.len()];

                for &Contact {
                    colliders: (i, j),
                    normal,
                    stiffness,
                } in &contacts
                {
                    let pi = particles.position[i];
                    let pj = particles.position[j];
                    let constraint = (pi - pj).dot(normal) - 1.0;
                    let force = normal * stiffness * constraint.min(0.0);
                    forces[i] -= force;
                    forces[j] += force;
                    jacobian_diag[i] += normal * normal * stiffness;
                    jacobian_diag[j] += normal * normal * stiffness;
                }
                let mut preconditioner_diag = vec![Vec2::ZERO; particles.len()];
                for i in 0..particles.len() {
                    preconditioner_diag[i] = 1.0 / (particles.mass[i] + jacobian_diag[i])
                }
                let gradient = (0..particles.len())
                    .map(|i| {
                        particles.mass[i] * (particles.velocity[i] - particles.last_velocity[i])
                            - forces[i]
                    })
                    .collect::<Vec<_>>();
                for i in 0..particles.len() {
                    particles.velocity[i] -= constraint_step * preconditioner_diag[i] * gradient[i];
                }
                for i in 0..particles.len() {
                    particles.position[i] = particles.velocity[i] + particles.last_position[i];
                }
            }
        }
        for i in 0..particles.len() {
            draw_circle(
                particles.position[i].x * scaling + offset.x,
                particles.position[i].y * scaling + offset.x,
                0.5 * scaling,
                RED,
            );
        }
        next_frame().await
    }
}
