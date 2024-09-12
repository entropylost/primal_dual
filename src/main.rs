// Hack to deal with nalgebra stack being slightly broken.
#![allow(clippy::toplevel_ref_arg)]
#![allow(unused)]

use contact::Contact;
use cosserat::{CosseratBendTwist, CosseratRod, CosseratStretchShear};
use iter_fixed::IntoIteratorFixed;
use macroquad::input::KeyCode;
use macroquad::window::request_new_screen_size;
use nalgebra::{
    self as na, matrix, stack, vector, DMatrixView, Matrix, MatrixXx3, MatrixXx4, SMatrix, SVector,
};
use std::fmt::Debug;
use std::{f32::consts::PI, ops::Deref};

mod split;
use split::{Invertible, Split};
mod contact;
mod cosserat;

type Real = f32;
type Scalar = na::Matrix1<Real>;
type DVector = na::DVector<Real>;
type DMatrix = na::DMatrix<Real>;
type Vector = na::Vector3<Real>;
type RVector = na::RowVector3<Real>;
type MatrixV = na::Matrix3<Real>;
type MatrixVR = na::Matrix3x4<Real>;
type MatrixRV = na::Matrix4x3<Real>;
type MatrixR = na::Matrix4<Real>;
type PartialRotation = na::Quaternion<Real>;
type Rotation = na::UnitQuaternion<Real>;

type Position = Split<Vector, Rotation>;
type Displacement = Split<Vector, PartialRotation>;
type Velocity = Split<Vector, Vector>;
type Force = Split<Vector, Vector>;
type Mass = Split<Real, MatrixV>;
type Gradient<const V: usize> = Split<SMatrix<Real, V, 3>, SMatrix<Real, V, 4>>;
type DGradient = Split<MatrixXx3<Real>, MatrixXx4<Real>>;
type Jacobian<const V: usize> = Split<SMatrix<Real, V, 3>, SMatrix<Real, V, 3>>;
type DJacobian = Split<MatrixXx3<Real>, MatrixXx3<Real>>;

impl<const V: usize> Gradient<V> {
    fn dynamic(self) -> DGradient {
        let linear_rows = self
            .linear
            .row_iter()
            .map(|x| x.clone_owned())
            .collect::<Vec<_>>();
        let angular_rows = self
            .angular
            .row_iter()
            .map(|x| x.clone_owned())
            .collect::<Vec<_>>();
        Split::new(
            MatrixXx3::from_rows(&linear_rows),
            MatrixXx4::from_rows(&angular_rows),
        )
    }
}

impl<const V: usize> Jacobian<V> {
    fn dynamic(self) -> DJacobian {
        let linear_rows = self
            .linear
            .row_iter()
            .map(|x| x.clone_owned())
            .collect::<Vec<_>>();
        let angular_rows = self
            .angular
            .row_iter()
            .map(|x| x.clone_owned())
            .collect::<Vec<_>>();
        Split::new(
            MatrixXx3::from_rows(&linear_rows),
            MatrixXx3::from_rows(&angular_rows),
        )
    }
}

impl Position {
    fn rotation_map(self) -> MatrixRV {
        let q = self.angular.quaternion().as_vector() / 2.0;
        matrix![
            q.w, q.z, -q.y;
            -q.z, q.w, q.x;
            q.y, -q.x, q.w;
            -q.x, -q.y, -q.z;
        ]
    }
    fn kinematic_map(self) -> Split<MatrixV, MatrixRV> {
        Split::new(MatrixV::identity(), self.rotation_map())
    }
    fn map_velocity(self, velocity: Velocity) -> Displacement {
        Displacement {
            linear: velocity.linear,
            // Should be equal to the kinematic map times the velocity.
            angular: (PartialRotation::from_imag(velocity.angular) * *self.angular) / 2.0,
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
            angular: Rotation::from_quaternion(self.angular),
        }
    }
}

#[test]
fn test_rotation_map() {
    let pos = Position::from_angular(Rotation::from_quaternion(PartialRotation::new(
        0.7, 3.0, 2.0, -1.0,
    )));
    let rot_map = pos.rotation_map();
    let vel = Vector::new(7.0, -10.0, 2.6);
    assert_eq!(
        PartialRotation::from_vector(rot_map * vel),
        pos.map_velocity(Velocity::from_angular(vel)).angular
    );
}

trait Constraint<const N: usize, const V: usize>: Debug {
    fn value(&self, positions: [Position; N]) -> SVector<Real, V>;
    fn gradient(&self, positions: [Position; N]) -> [Gradient<V>; N];
    fn stiffness(&self) -> SVector<Real, V>;
    fn jacobian(&self, positions: [Position; N]) -> [Jacobian<V>; N] {
        let gradient = self.gradient(positions);
        gradient
            .into_iter_fixed()
            .zip(positions)
            .map(|(grad, pos)| grad * pos.kinematic_map())
            .collect()
    }
    fn potential(&self, positions: [Position; N]) -> Real {
        let value = self.value(positions);
        *(value.transpose() * Matrix::from_diagonal(&self.stiffness()) * value).as_scalar()
    }
    fn force(&self, positions: [Position; N]) -> [Force; N] {
        let gradient = self.gradient(positions);
        let value = self.value(positions);
        gradient
            .into_iter_fixed()
            .zip(positions)
            .map(|(grad, pos)| {
                let map = pos.kinematic_map();
                let jc = grad * map;
                Split::new(
                    -jc.linear.transpose() * Matrix::from_diagonal(&self.stiffness()) * value,
                    -jc.angular.transpose() * Matrix::from_diagonal(&self.stiffness()) * value,
                )
            })
            .collect()
    }
    fn grad2_diag(&self, positions: [Position; N]) -> [Split; N] {
        self.jacobian(positions)
            .into_iter_fixed()
            .map(|jc| {
                Split::new(
                    (jc.linear.transpose() * Matrix::from_diagonal(&self.stiffness()) * jc.linear)
                        .diagonal(),
                    (jc.angular.transpose()
                        * Matrix::from_diagonal(&self.stiffness())
                        * jc.angular)
                        .diagonal(),
                )
            })
            .collect()
    }
    fn dual_preconditioner(
        &self,
        positions: [Position; N],
        masses: [Mass; N],
    ) -> SMatrix<Real, V, V> {
        let denom = Matrix::from_diagonal(&self.stiffness().map(Real::recip))
            + self
                .jacobian(positions)
                .into_iter_fixed()
                .zip(masses)
                .map(|(jc, mass)| {
                    jc.linear * mass.linear.inverse() * jc.linear.transpose()
                        + jc.angular * mass.angular.inverse() * jc.angular.transpose()
                })
                .into_iter()
                .fold(SMatrix::zeros(), |acc, x| acc + x);
        denom.try_inverse().unwrap()
    }
    fn dual_cheap_preconditioner_diag(
        &self,
        positions: [Position; N],
        masses: [Mass; N],
    ) -> SVector<Real, V> {
        let denom = self.stiffness().map(Real::recip)
            + self
                .jacobian(positions)
                .into_iter_fixed()
                .zip(masses)
                .map(|(jc, mass)| {
                    (jc.linear * mass.linear.inverse() * jc.linear.transpose()).diagonal()
                        + (jc.angular * mass.angular.inverse() * jc.angular.transpose()).diagonal()
                })
                .into_iter()
                .fold(SVector::zeros(), |acc, x| acc + x);
        denom.map(Real::recip)
    }
}

#[derive(Debug)]
struct ConstraintWrapper<const N: usize, const V: usize, X: Constraint<N, V>>(X);

trait DynConstraint: Debug {
    fn dim_n(&self) -> usize;
    fn dim_v(&self) -> usize;
    fn value(&self, positions: &[Position]) -> DVector;
    fn gradient(&self, positions: &[Position]) -> Vec<DGradient>;
    fn jacobian(&self, positions: &[Position]) -> Vec<DJacobian>;
    fn stiffness(&self) -> DVector;
    fn potential(&self, positions: &[Position]) -> Real;
    fn force(&self, positions: &[Position]) -> Vec<Force>;
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split>;

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix;
    fn dual_cheap_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector;
}

impl<const N: usize, const V: usize, X> DynConstraint for ConstraintWrapper<N, V, X>
where
    X: Constraint<N, V>,
{
    fn dim_n(&self) -> usize {
        N
    }
    fn dim_v(&self) -> usize {
        V
    }
    fn value(&self, positions: &[Position]) -> DVector {
        DVector::from_column_slice(self.0.value(positions.try_into().unwrap()).as_slice())
    }
    fn gradient(&self, positions: &[Position]) -> Vec<DGradient> {
        self.0
            .gradient(positions.try_into().unwrap())
            .map(|x| x.dynamic())
            .into()
    }
    fn jacobian(&self, positions: &[Position]) -> Vec<DJacobian> {
        self.0
            .jacobian(positions.try_into().unwrap())
            .map(|x| x.dynamic())
            .into()
    }
    fn stiffness(&self) -> DVector {
        DVector::from_column_slice(self.0.stiffness().as_slice())
    }
    fn potential(&self, positions: &[Position]) -> Real {
        self.0.potential(positions.try_into().unwrap())
    }
    fn force(&self, positions: &[Position]) -> Vec<Force> {
        self.0.force(positions.try_into().unwrap()).into()
    }
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split> {
        self.0.grad2_diag(positions.try_into().unwrap()).into()
    }

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix {
        let pc = self
            .0
            .dual_preconditioner(positions.try_into().unwrap(), mass.try_into().unwrap());
        let v: DMatrixView<f32> = pc.as_view();
        v.clone_owned()
    }
    fn dual_cheap_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector {
        DVector::from_column_slice(
            self.0
                .dual_cheap_preconditioner_diag(
                    positions.try_into().unwrap(),
                    mass.try_into().unwrap(),
                )
                .as_slice(),
        )
    }
}

#[derive(Debug)]
struct ConstraintBox {
    targets: Vec<usize>,
    constraint: Box<dyn DynConstraint>,
}
impl ConstraintBox {
    fn new<const N: usize, const V: usize>(
        targets: [usize; N],
        constraint: impl Constraint<N, V> + 'static,
    ) -> Self {
        Self {
            targets: targets.to_vec(),
            constraint: Box::new(ConstraintWrapper(constraint)),
        }
    }
}

#[macroquad::main("Primal / Dual")]
async fn main() {
    request_new_screen_size(1000.0, 800.0);

    let mut position: Vec<Position> = vec![
        vector![0.0, 0.0, 0.0],
        vector![2.0, 0.0, 0.0],
        vector![4.0, 0.0, 0.0],
        vector![6.0, 0.0, 0.0],
        vector![8.0, 0.0, 0.0],
        vector![8.0, -3.0, 0.0],
    ]
    .into_iter()
    .map(Split::from_linear)
    .collect();
    let mut velocity: Vec<Velocity> = vec![
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 2.0, 0.0], vector![0.0, 0.0, 0.0]),
    ];
    // TODO: Setting it to infinity is not supported yet.
    let mass: Vec<Mass> = vec![9999999.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        .into_iter()
        .map(|x| Split::new(x, MatrixV::from_diagonal_element(2.0 / 5.0 * x * 0.5 * 0.5)))
        .collect();
    let particles = position.len();
    assert_eq!(particles, velocity.len());
    assert_eq!(particles, mass.len());

    let constraint_step = 0.5;
    let dt = 1.0 / 60.0;
    let substeps = 1;
    let num_iters = 1;
    let mut dual = true;
    let mut dual_cheap_precond = true;
    let mut warm_start = false;
    let warm_start_factor = 0.5;
    let mut running = false;

    let dt = dt / substeps as Real;
    for v in &mut velocity {
        v.linear *= dt;
        v.angular *= dt;
    }

    let rod = CosseratRod::resting_state(
        0.5,
        10000.0 * dt * dt, // This makes the rod stiffness independent of time.
        10000.0 * dt * dt,
        [position[0], position[1]],
    );

    let mut constraints = vec![
        ConstraintBox::new([0, 1], CosseratStretchShear { rod }),
        ConstraintBox::new([0, 1], CosseratBendTwist { rod }),
        ConstraintBox::new([1, 2], CosseratStretchShear { rod }),
        ConstraintBox::new([1, 2], CosseratBendTwist { rod }),
        ConstraintBox::new([2, 3], CosseratStretchShear { rod }),
        ConstraintBox::new([2, 3], CosseratBendTwist { rod }),
        ConstraintBox::new([3, 4], CosseratStretchShear { rod }),
        ConstraintBox::new([3, 4], CosseratBendTwist { rod }),
    ];

    let mut last_forces = vec![Force::default(); particles];
    let mut last_dual_vars = constraints
        .iter()
        .map(|x| DVector::zeros(x.constraint.dim_v()))
        .collect::<Vec<_>>();

    loop {
        if macroquad::input::is_key_pressed(KeyCode::Space) {
            running = !running;
        }
        if macroquad::input::is_key_pressed(KeyCode::Escape) {
            break;
        }
        if macroquad::input::is_key_pressed(KeyCode::P) {
            dual = !dual;
        }
        if macroquad::input::is_key_pressed(KeyCode::C) {
            dual_cheap_precond = !dual_cheap_precond;
        }
        if macroquad::input::is_key_pressed(KeyCode::W) {
            warm_start = !warm_start;
        }
        if macroquad::input::is_key_pressed(KeyCode::W)
            || macroquad::input::is_key_pressed(KeyCode::P)
        {
            last_forces = vec![Force::default(); particles];
            last_dual_vars = constraints
                .iter()
                .map(|x| DVector::zeros(x.constraint.dim_v()))
                .collect::<Vec<_>>();
        }

        if running || macroquad::input::is_key_pressed(KeyCode::Period) {
            for _step in 0..substeps {
                let last_position = position.clone();
                let last_velocity = velocity.clone();
                for i in 0..particles {
                    position[i] = position[i].step(velocity[i]);
                }

                let lasting_constraints = constraints.len();

                for i in 0..particles {
                    for j in i + 1..particles {
                        let pi = position[i];
                        let pj = position[j];
                        if (pi.linear - pj.linear).norm() <= 1.0 {
                            constraints.push(ConstraintBox::new(
                                [i, j],
                                Contact {
                                    normal: (pi.linear - pj.linear).normalize().transpose(),
                                    stiffness: 99999.0 * dt * dt,
                                    length: 1.0,
                                },
                            ));
                        }
                    }
                }
                if dual {
                    let mut dual_vars = constraints
                        .iter()
                        .map(|x| DVector::zeros(x.constraint.dim_v()))
                        .collect::<Vec<_>>();
                    for _iter in 0..num_iters {
                        let last_dual_vars = dual_vars.clone();
                        for (
                            i,
                            ConstraintBox {
                                targets,
                                constraint,
                            },
                        ) in constraints.iter().enumerate()
                        {
                            let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();
                            let m = targets.iter().map(|&i| mass[i]).collect::<Vec<_>>();
                            let dual_force = -constraint.value(&p)
                                - last_dual_vars[i].component_div(&constraint.stiffness());
                            let precond = if dual_cheap_precond {
                                DMatrix::from_diagonal(
                                    &constraint.dual_cheap_preconditioner_diag(&p, &m),
                                )
                            } else {
                                constraint.dual_preconditioner(&p, &m)
                            };
                            let delta = constraint_step * precond * dual_force;
                            dual_vars[i] += &delta;
                            let jacobian = constraint.jacobian(&p);
                            for (j, &k) in targets.iter().enumerate() {
                                velocity[k].linear += mass[k].linear.inverse()
                                    * jacobian[j].linear.transpose()
                                    * &delta;
                                velocity[k].angular += mass[k].angular.inverse()
                                    * jacobian[j].angular.transpose()
                                    * &delta;
                            }
                        }
                        for i in 0..particles {
                            position[i] = last_position[i].step(velocity[i]);
                        }
                    }
                } else {
                    if warm_start {
                        for i in 0..particles {
                            velocity[i] += mass[i].inverse() * last_forces[i] * warm_start_factor;
                        }
                        for i in 0..particles {
                            position[i] = last_position[i].step(velocity[i]);
                        }
                    }
                    for _iter in 0..num_iters {
                        let mut forces = vec![Force::default(); particles];
                        let mut grad2_diag = vec![Split::<Vector, Vector>::default(); particles];

                        for ConstraintBox {
                            targets,
                            constraint,
                        } in &constraints
                        {
                            let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();

                            let force = constraint.force(&p);
                            let grad2 = constraint.grad2_diag(&p);
                            for (i, &j) in targets.iter().enumerate() {
                                forces[j] += force[i];
                                grad2_diag[j] += grad2[i];
                            }
                        }
                        let step = (0..particles)
                            .map(|i| {
                                let precond = Split::new(
                                    Vector::repeat(mass[i].linear) + grad2_diag[i].linear,
                                    mass[i].angular.diagonal() + grad2_diag[i].angular,
                                )
                                .recip();
                                let grad = mass[i] * (velocity[i] - last_velocity[i]) - forces[i];
                                precond.component_mul(grad)
                            })
                            .collect::<Vec<_>>();
                        for (i, step) in step.into_iter().enumerate() {
                            velocity[i] -= constraint_step * step;
                        }
                        for i in 0..particles {
                            position[i] = last_position[i].step(velocity[i]);
                        }
                    }
                    if warm_start {
                        for i in 0..particles {
                            last_forces[i] = mass[i] * (velocity[i] - last_velocity[i]);
                        }
                    }
                }
                constraints.truncate(lasting_constraints);
            }
        }
        {
            use macroquad::prelude::*;
            let scaling = 50.0;
            let offset = vector![screen_width() / 2.0, screen_height() / 2.0];

            clear_background(BLACK);
            if !running {
                draw_text("Paused", screen_width() - 100.0, 30.0, 30.0, WHITE);
            }
            if dual {
                draw_text("Solver: Dual", 10.0, 30.0, 30.0, WHITE);
                draw_text(
                    if dual_cheap_precond {
                        "Preconditioner: Cheap"
                    } else {
                        "Preconditioner: Exact"
                    },
                    10.0,
                    60.0,
                    30.0,
                    WHITE,
                );
            } else {
                draw_text("Solver: Primal", 10.0, 30.0, 30.0, WHITE);
            }

            for p in &position {
                let pos = p.linear.xy() * scaling + offset;
                draw_circle(pos.x, pos.y, 0.5 * scaling, RED);
                let rot_x = (p.angular * vector![0.5, 0.0, 0.0]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_x.x, rot_x.y, 3.0, WHITE);
                let rot_y = (p.angular * vector![0.0, 0.5, 0.0]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_y.x, rot_y.y, 3.0, GREEN);
                let rot_z = (p.angular * vector![0.0, 0.0, 0.5]).xy() * scaling + pos;
                draw_line(pos.x, pos.y, rot_z.x, rot_z.y, 3.0, BLUE);
            }
            macroquad::window::next_frame().await
        }
    }
}
