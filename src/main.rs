// Hack to deal with nalgebra stack being slightly broken.
// TODO: File report.
#![allow(clippy::toplevel_ref_arg)]
#![allow(unused)]

use contact::Contact;
use cosserat::{CosseratBendTwist, CosseratRod, CosseratStretchShear};
use iter_fixed::IntoIteratorFixed;
use macroquad::input::KeyCode;
use nalgebra::{self as na, matrix, stack, vector, Matrix, MatrixXx3, MatrixXx4, SMatrix, SVector};
use std::fmt::Debug;
use std::{f32::consts::PI, ops::Deref};

mod opt;
mod split;
use split::Split;
mod contact;
mod cosserat;

type Real = f32;
type Scalar = na::Matrix1<Real>;
type DVector = na::DVector<Real>;
type Vector3 = na::Vector3<Real>;
type RVector3 = na::RowVector3<Real>;
type Matrix3 = na::Matrix3<Real>;
type Matrix3x4 = na::Matrix3x4<Real>;
type Matrix4x3 = na::Matrix4x3<Real>;
type Matrix4 = na::Matrix4<Real>;
type Quaternion = na::Quaternion<Real>;
type UnitQuaternion = na::UnitQuaternion<Real>;

type Split3 = Split<Vector3, Vector3>;
type Position = Split<Vector3, UnitQuaternion>;
type Displacement = Split<Vector3, Quaternion>;
type Velocity = Split<Vector3, Vector3>;
type Force = Split<Vector3, Vector3>;
type Mass = Split<Real, Matrix3>;
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
    fn rotation_map(self) -> Matrix4x3 {
        let q = self.angular.quaternion().as_vector() / 2.0;
        matrix![
            q.w, q.z, -q.y;
            -q.z, q.w, q.x;
            q.y, -q.x, q.w;
            -q.x, -q.y, -q.z;
        ]
    }
    fn kinematic_map(self) -> Split<Matrix3, Matrix4x3> {
        Split::new(Matrix3::identity(), self.rotation_map())
    }
    fn map_velocity(self, velocity: Velocity) -> Displacement {
        Displacement {
            linear: velocity.linear,
            // Should be equal to the kinematic map times the velocity.
            angular: (Quaternion::from_imag(velocity.angular) * *self.angular) / 2.0,
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
            angular: UnitQuaternion::from_quaternion(self.angular),
        }
    }
}

#[test]
fn test_rotation_map() {
    let pos = Position::from_angular(UnitQuaternion::from_quaternion(Quaternion::new(
        0.7, 3.0, 2.0, -1.0,
    )));
    let rot_map = pos.rotation_map();
    let vel = Vector3::new(7.0, -10.0, 2.6);
    assert_eq!(
        Quaternion::from_vector(rot_map * vel),
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
    fn grad2_diag(&self, positions: [Position; N]) -> [Split3; N] {
        self.jacobian(positions)
            .into_iter_fixed()
            .zip(positions)
            .map(|(jc, pos)| {
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
    fn preconditioner_diag(&self, positions: [Position; N], masses: [Mass; N]) -> SVector<Real, V> {
        let denom = self.stiffness().map(|x| 1.0 / x)
            + self
                .jacobian(positions)
                .into_iter_fixed()
                .zip(masses)
                .map(|(jc, mass)| {
                    (jc.linear * (1.0 / mass.linear) * jc.linear.transpose()).diagonal()
                        + (jc.angular
                            * mass.angular.try_inverse().unwrap()
                            * jc.angular.transpose())
                        .diagonal()
                })
                .into_iter()
                .fold(SVector::zeros(), |acc, x| acc + x);
        denom.map(|x| 1.0 / x)
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
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split3>;

    fn preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector;
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
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split3> {
        self.0.grad2_diag(positions.try_into().unwrap()).into()
    }

    fn preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector {
        DVector::from_column_slice(
            self.0
                .preconditioner_diag(positions.try_into().unwrap(), mass.try_into().unwrap())
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

#[macroquad::main("Pbd")]
async fn main() {
    let mut position: Vec<Position> = vec![vector![0.0, 0.0, 0.0], vector![2.0, 0.0, 0.0]]
        .into_iter()
        .map(Split::from_linear)
        .collect();
    let mut velocity: Vec<Velocity> = vec![
        Split::new(vector![0.1, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
    ];
    let mass: Vec<Mass> = vec![1.0, 1.0]
        .into_iter()
        .map(|x| Split::new(x, Matrix3::from_diagonal_element(2.0 / 5.0 * x * 0.5 * 0.5)))
        .collect();
    let particles = position.len();
    assert_eq!(particles, velocity.len());
    assert_eq!(particles, mass.len());

    let bt = CosseratBendTwist {
        rod: CosseratRod::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };
    let se = CosseratStretchShear {
        rod: CosseratRod::resting_state(0.5, 1.0, 1.0, [position[0], position[1]]),
    };

    let mut constraints = vec![
        ConstraintBox::new([0, 1], se),
        ConstraintBox::new([0, 1], bt),
    ];

    let constraint_step = 1.0;
    let num_iters = 10;

    let pbd = true;

    loop {
        let running = macroquad::input::is_key_pressed(KeyCode::Period)
            || macroquad::input::is_key_down(KeyCode::Space);

        if running {
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
                                stiffness: Real::INFINITY,
                                length: 1.0,
                            },
                        ));
                    }
                }
            }
            if pbd {
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
                        let precond = constraint.preconditioner_diag(&p, &m);
                        dual_vars[i] += constraint_step * dual_force.component_mul(&precond);
                    }
                    let delta = dual_vars
                        .iter()
                        .zip(last_dual_vars)
                        .map(|(x, y)| x - y)
                        .collect::<Vec<_>>();
                    for (
                        i,
                        ConstraintBox {
                            targets,
                            constraint,
                        },
                    ) in constraints.iter().enumerate()
                    {
                        let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();
                        let jacobian = constraint.jacobian(&p);
                        for (j, &k) in targets.iter().enumerate() {
                            velocity[k].linear +=
                                (1.0 / mass[k].linear) * jacobian[j].linear.transpose() * &delta[i];
                            velocity[k].angular += mass[k].angular.try_inverse().unwrap()
                                * jacobian[j].angular.transpose()
                                * &delta[i];
                        }
                    }
                    for i in 0..particles {
                        position[i] = last_position[i].step(velocity[i]);
                    }
                }
            } else {
                for _iter in 0..num_iters {
                    let mut forces = vec![Force::default(); particles];
                    let mut grad2_diag = vec![Split::<Vector3, Vector3>::default(); particles];

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
                    let mut preconditioner_diag =
                        vec![Split::<Vector3, Vector3>::default(); particles];
                    for i in 0..particles {
                        preconditioner_diag[i].linear = (Vector3::repeat(mass[i].linear)
                            + grad2_diag[i].linear)
                            .map(|x| 1.0 / x);
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
            constraints.truncate(lasting_constraints);
        }
        {
            use macroquad::prelude::*;
            let scaling = 50.0;
            let offset = vector![200.0, 200.0];

            clear_background(BLACK);
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
