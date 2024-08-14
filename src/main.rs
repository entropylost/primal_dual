// Hack to deal with nalgebra stack being slightly broken.
// TODO: File report.
#![allow(clippy::toplevel_ref_arg)]

use macroquad::input::KeyCode;
use nalgebra::{self as na, matrix, stack, vector, SMatrix};
use std::fmt::Debug;
use std::ops::{AddAssign, Div, DivAssign, MulAssign, SubAssign};
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

impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize> Mul<f32>
    for Split<SMatrix<f32, R1, C1>, SMatrix<f32, R2, C2>>
{
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self {
            linear: self.linear * rhs,
            angular: self.angular * rhs,
        }
    }
}
impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize> Div<f32>
    for Split<SMatrix<f32, R1, C1>, SMatrix<f32, R2, C2>>
{
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self {
            linear: self.linear / rhs,
            angular: self.angular / rhs,
        }
    }
}
impl<const R1: usize, const R2: usize, const C1: usize, const C2: usize>
    Mul<Split<SMatrix<f32, R1, C1>, SMatrix<f32, R2, C2>>> for f32
{
    type Output = Split<SMatrix<f32, R1, C1>, SMatrix<f32, R2, C2>>;
    fn mul(self, rhs: Split<SMatrix<f32, R1, C1>, SMatrix<f32, R2, C2>>) -> Self::Output {
        Split {
            linear: self * rhs.linear,
            angular: self * rhs.angular,
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
                q.w, q.z, -q.y;
                -q.z, q.w, q.x;
                q.y, -q.x, q.w;
                -q.x, -q.y, -q.z;
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
    fn unconstrain(self) -> Displacement {
        Displacement {
            linear: self.linear,
            angular: *self.angular,
        }
    }
    fn step(self, velocity: Velocity) -> Self {
        (self.map_velocity(velocity) + self.unconstrain()).normalize()
    }
}
impl Displacement {
    fn normalize(self) -> Position {
        Position {
            linear: self.linear,
            angular: UQuat::from_quaternion(self.angular),
        }
    }
}

#[test]
fn test_rotation_map() {
    let pos = Position::from_angular(UQuat::from_quaternion(Quat::new(0.7, 3.0, 2.0, -1.0)));
    let rot_map = pos.rotation_map();
    let vel = Vec3::new(7.0, -10.0, 2.6);
    assert_eq!(
        Quat::from_vector(rot_map * vel),
        pos.map_velocity(Velocity::from_angular(vel)).angular
    );
}

trait Constraint<const N: usize>: Debug {
    fn potential(&self, positions: [Position; N]) -> f32;
    fn force(&self, positions: [Position; N]) -> [Force; N];
    fn grad2_diag(&self, positions: [Position; N]) -> [Split<Vec3, Vec3>; N];
    // Add dual properties. Require indexing into a set of dual vectors or something.
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
    fn grad2_diag(&self, _p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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
    fn grad2_diag(&self, p: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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
    fn resting_state(
        rod_radius: f32,
        young_modulus: f32,
        shear_modulus: f32,
        [pi, pj]: [Position; 2],
    ) -> Self {
        let length = (pj.linear - pi.linear).norm();
        let rest_rotation = UQuat::rotation_between(&Vec3::z(), &(pj.linear - pi.linear)).unwrap();
        Self {
            rod_radius,
            young_modulus,
            shear_modulus,
            length,
            rest_rotation,
        }
    }
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
        Mat3::from_diagonal_element(q.scalar()) - cross_matrix(q.vector().into()), q.vector();
        -q.vector().transpose(), na::Matrix1::<f32>::new(q.scalar())
    ]
}

// p * q = lmul_mat(p) * q
fn lmul_mat(p: Quat) -> Mat4 {
    stack![
        Mat3::from_diagonal_element(p.scalar()) + cross_matrix(p.vector().into()), p.vector();
         -p.vector().transpose(), na::Matrix1::<f32>::new(p.scalar())
    ]
}

#[test]
fn test_quat_mul() {
    let p = Quat::new(1.0, 6.0, -2.0, 3.0);
    let q = Quat::new(7.0, -3.0, 5.0, 10.0);
    assert_eq!((p * q).coords, lmul_mat(p) * q.coords);
    assert_eq!((p * q).coords, rmul_mat(q) * p.coords);
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
            * rmul_mat(Quat::from_imag(pj.linear - pi.linear) * qij).fixed_view::<3, 4>(0, 0)
            * Mat4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
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
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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
                * Mat4::from_diagonal(&vector![-1.0, -1.0, -1.0, 1.0])
                * dqij
                - lmul_mat(*self.center_rotation(p).conjugate()));
        gradient.fixed_view::<3, 4>(0, 0).into_owned()
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
    fn grad2_diag(&self, p @ [pi, pj]: [Position; 2]) -> [Split<Vec3, Vec3>; 2] {
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

#[derive(Debug)]
struct Constraint2 {
    targets: [usize; 2],
    constraint: Box<dyn Constraint<2>>,
}

#[macroquad::main("main")]
async fn main() {
    let mut position: Vec<Position> = vec![vector![0.0, 0.0, 0.0], vector![2.0, 0.0, 0.0]]
        .into_iter()
        .map(Split::from_linear)
        .collect();
    let mut velocity: Vec<Velocity> = vec![vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]]
        .into_iter()
        .map(Split::from_linear)
        .collect();
    let mass: Vec<Mass> = vec![5.0, 1.0]
        .into_iter()
        .map(|x| Split::new(x, Mat3::from_diagonal_element(2.0 / 5.0 * x * 0.5 * 0.5)))
        .collect();
    let particles = position.len();
    assert_eq!(particles, velocity.len());
    assert_eq!(particles, mass.len());

    let bt = CosseratBendTwist {
        params: CosseratParameters::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };
    let se = CosseratStretchShear {
        params: CosseratParameters::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };
    println!(
        "Measures: {:?}, {:?}",
        se.strain_measure([position[0], position[1]]),
        bt.darboux_vector([position[0], position[1]])
    );

    let constraints = [
        // Constraint2 {
        //     targets: [0, 1],
        //     constraint: Box::new(bt),
        // },
        Constraint2 {
            targets: [0, 1],
            constraint: Box::new(se),
        },
    ];

    let constraint_step = 0.1;

    loop {
        let running = macroquad::input::is_key_pressed(KeyCode::Period)
            || macroquad::input::is_key_down(KeyCode::Space);

        if running {
            let last_position = position.clone();
            let last_velocity = velocity.clone();
            for i in 0..particles {
                position[i] = position[i].step(velocity[i]);
            }

            let mut contacts: Vec<Constraint2> = vec![];

            // for i in 0..particles {
            //     for j in i + 1..particles {
            //         let pi = position[i];
            //         let pj = position[j];
            //         if (pi.linear - pj.linear).norm() <= 1.0 {
            //             contacts.push(Constraint2 {
            //                 targets: [i, j],
            //                 constraint: Box::new(Contact {
            //                     normal: (pi.linear - pj.linear).normalize(),
            //                     stiffness: 10000.0,
            //                     length: 1.0,
            //                 }),
            //             });
            //         }
            //     }
            // }
            for _iter in 0..1 {
                let mut forces = vec![Force::default(); particles];
                let mut grad2_diag = vec![Split::<Vec3, Vec3>::default(); particles];

                for Constraint2 {
                    targets,
                    constraint,
                } in constraints.iter().chain(&contacts)
                {
                    let [i, j] = *targets;
                    let pi = position[i];
                    let pj = position[j];
                    let p = [pi, pj];
                    let force = constraint.force(p);
                    println!("Force: {:?}", force);
                    let grad2 = constraint.grad2_diag(p);
                    forces[i] += force[0];
                    forces[j] += force[1];
                    grad2_diag[i] += grad2[0];
                    grad2_diag[j] += grad2[1];
                }
                let mut preconditioner_diag = vec![Split::<Vec3, Vec3>::default(); particles];
                for i in 0..particles {
                    preconditioner_diag[i].linear =
                        (Vec3::repeat(mass[i].linear) + grad2_diag[i].linear).map(|x| 1.0 / x);
                    preconditioner_diag[i].angular =
                        (mass[i].angular.diagonal() + grad2_diag[i].angular).map(|x| 1.0 / x);
                }
                let gradient = (0..particles)
                    .map(|i| mass[i] * (velocity[i] - last_velocity[i]) - forces[i])
                    .collect::<Vec<_>>();
                for i in 0..particles {
                    let step = Split {
                        linear: preconditioner_diag[i]
                            .linear
                            .component_mul(&gradient[i].linear),
                        angular: preconditioner_diag[i]
                            .angular
                            .component_mul(&gradient[i].angular),
                    };
                    velocity[i] -= constraint_step * step;
                }
                for i in 0..particles {
                    position[i] = last_position[i].step(velocity[i]);
                }
            }
        }
        {
            use macroquad::prelude::*;
            let scaling = 50.0;
            let offset = vector![100.0, 100.0];

            clear_background(BLACK);
            for i in 0..particles {
                let pos = position[i].linear.xy() * scaling + offset;
                draw_circle(pos.x, pos.y, 0.5 * scaling, RED);
                let rot = (position[i].angular * vector![0.5, 0.0, 0.0]).xy() * scaling;
                draw_line(pos.x, pos.y, pos.x + rot.x, pos.y + rot.y, 3.0, WHITE);
            }
            macroquad::window::next_frame().await
        }
    }
}
