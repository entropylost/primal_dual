// Hack to deal with nalgebra stack being slightly broken.
#![allow(clippy::toplevel_ref_arg)]
// #![allow(unused)]

use contact::Contact;
use cosserat::{CosseratBendTwist, CosseratRod, CosseratStretchShear};
use iter_fixed::IntoIteratorFixed;
use macroquad::input::KeyCode;
use macroquad::window::request_new_screen_size;
use nalgebra::{
    self as na, matrix, stack, vector, DMatrixView, Matrix, MatrixXx1, MatrixXx2, MatrixXx3,
    MatrixXx4, SMatrix, SVector,
};
use std::fmt::Debug;
use std::{f32::consts::PI, ops::Deref};

mod split;
use split::{Invertible, Split};

use crate::solver::{DualSolver, Solver, Solvers};
mod contact;
mod cosserat;
mod solver;

type Real = f32;
type Scalar = na::Matrix1<Real>;
type DVector = na::DVector<Real>;
type DMatrix = na::DMatrix<Real>;
type Vector = na::Vector2<Real>;
type RVector = na::RowVector2<Real>;
type MatrixV = na::Matrix2<Real>;
type MatrixVR = na::Matrix2<Real>;
type MatrixRV = na::Matrix2<Real>;
type MatrixR = na::Matrix2<Real>;

type PartialRotation = Vector;
type Rotation = Real;

type Position = Split<Vector, Rotation>;
type Displacement = Split<Vector, Rotation>;
type Velocity = Split<Vector, Real>;
type Force = Split<Vector, Real>;
type Mass = Split<Real, Real>;
type Gradient<const V: usize> = Split<SMatrix<Real, V, 2>, SMatrix<Real, V, 1>>;
type DGradient = Split<MatrixXx2<Real>, DVector>;
type Jacobian<const V: usize> = Split<SMatrix<Real, V, 2>, SMatrix<Real, V, 1>>;
type DJacobian = Split<MatrixXx2<Real>, DVector>;
type Hessian = Split<SMatrix<Real, 2, 2>, Real>;

fn rotation_matrix(q: Rotation) -> MatrixR {
    matrix![
        q.cos(), -q.sin();
        q.sin(), q.cos()
    ]
}
fn rotation_matrix_gradient(q: Rotation) -> MatrixR {
    matrix![
        -q.sin(), -q.cos();
        q.cos(), -q.sin()
    ]
}

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
            MatrixXx2::from_rows(&linear_rows),
            MatrixXx1::from_rows(&angular_rows),
        )
    }
}

// impl<const V: usize> Jacobian<V> {
//     fn dynamic(self) -> DJacobian {
//         let linear_rows = self
//             .linear
//             .row_iter()
//             .map(|x| x.clone_owned())
//             .collect::<Vec<_>>();
//         let angular_rows = self
//             .angular
//             .row_iter()
//             .map(|x| x.clone_owned())
//             .collect::<Vec<_>>();
//         Split::new(
//             MatrixXx2::from_rows(&linear_rows),
//             MatrixXx2::from_rows(&angular_rows),
//         )
//     }
// }

impl Position {
    fn normalize(mut self) -> Self {
        self.angular %= 4.0 * PI;
        self
    }

    fn step(self, velocity: Velocity) -> Self {
        (velocity + self).normalize()
    }
}

trait Constraint<const N: usize, const V: usize>: Debug {
    fn value(&self, positions: [Position; N]) -> SVector<Real, V>;
    fn gradient(&self, positions: [Position; N]) -> [Gradient<V>; N];
    fn stiffness(&self) -> SVector<Real, V>;
    fn jacobian(&self, positions: [Position; N]) -> [Jacobian<V>; N] {
        self.gradient(positions)
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
            .map(|jc| {
                Split::new(
                    -jc.linear.transpose() * Matrix::from_diagonal(&self.stiffness()) * value,
                    (-jc.angular.transpose() * Matrix::from_diagonal(&self.stiffness()) * value)
                        .into_scalar(),
                )
            })
            .collect()
    }
    fn hessian(&self, positions: [Position; N]) -> [Hessian; N] {
        self.jacobian(positions)
            .into_iter_fixed()
            .map(|jc| {
                Split::new(
                    jc.linear.transpose() * Matrix::from_diagonal(&self.stiffness()) * jc.linear,
                    (jc.angular.transpose()
                        * Matrix::from_diagonal(&self.stiffness())
                        * jc.angular)
                        .to_scalar(),
                )
            })
            .collect()
    }
    fn hessian_diag(&self, positions: [Position; N]) -> [Split; N] {
        self.jacobian(positions)
            .into_iter_fixed()
            .map(|jc| {
                Split::new(
                    (jc.linear.transpose() * Matrix::from_diagonal(&self.stiffness()) * jc.linear)
                        .diagonal(),
                    (jc.angular.transpose()
                        * Matrix::from_diagonal(&self.stiffness())
                        * jc.angular)
                        .into_scalar(),
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
    fn dual_preconditioner_diag(
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

    fn hessian(&self, positions: &[Position]) -> Vec<Hessian>;
    fn hessian_diag(&self, positions: &[Position]) -> Vec<Split>;

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix;
    fn dual_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector;
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

    fn hessian(&self, positions: &[Position]) -> Vec<Hessian> {
        self.0.hessian(positions.try_into().unwrap()).into()
    }
    fn hessian_diag(&self, positions: &[Position]) -> Vec<Split> {
        self.0.hessian_diag(positions.try_into().unwrap()).into()
    }

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix {
        let pc = self
            .0
            .dual_preconditioner(positions.try_into().unwrap(), mass.try_into().unwrap());
        let v: DMatrixView<f32> = pc.as_view();
        v.clone_owned()
    }
    fn dual_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector {
        DVector::from_column_slice(
            self.0
                .dual_preconditioner_diag(positions.try_into().unwrap(), mass.try_into().unwrap())
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
        vector![0.0, 0.0],
        vector![2.0, 0.0],
        vector![4.0, 0.0],
        vector![6.0, 0.0],
        vector![8.0, 0.0],
        vector![8.0, -3.0],
    ]
    .into_iter()
    .map(Split::from_linear)
    .collect();
    let mut velocity: Vec<Velocity> = vec![
        Split::new(vector![0.0, 0.0], 0.0),
        Split::new(vector![0.0, 0.0], 0.0),
        Split::new(vector![0.0, 0.0], 0.0),
        Split::new(vector![0.0, 0.0], 0.0),
        Split::new(vector![0.0, 0.0], 0.0),
        Split::new(vector![0.0, 2.0], 0.0),
    ];
    // TODO: Setting it to infinity is not supported yet.
    let mass: Vec<Mass> = vec![9999999.0, 1.0, 1.0, 1.0, 1.0, 5.0]
        .into_iter()
        .map(|x| Split::new(x, 1.0 / 2.0 * x * 0.5 * 0.5))
        .collect();
    let particles = position.len();
    assert_eq!(particles, velocity.len());
    assert_eq!(particles, mass.len());

    let dt = 1.0 / 60.0;
    let substeps = 1;

    let mut solver = Solvers::Dual(DualSolver {
        iterations: 1,
        constraint_step: 0.5,
        diag_precond: true,
    });

    let mut running = false;

    let dt = dt / substeps as Real;
    let dt2 = dt * dt;
    for v in &mut velocity {
        v.linear *= dt;
        v.angular *= dt;
    }

    let rod = CosseratRod::resting_state(
        0.5,
        10000.0 * dt2, // This makes the rod stiffness independent of time.
        10000.0 * dt2,
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

    loop {
        if macroquad::input::is_key_pressed(KeyCode::Space) {
            running = !running;
        }
        if macroquad::input::is_key_pressed(KeyCode::Escape) {
            break;
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
                                    stiffness: 99999.0 * dt2,
                                    length: 1.0,
                                },
                            ));
                        }
                    }
                }

                solver.solve(
                    &mass,
                    &constraints,
                    &last_position,
                    &last_velocity,
                    &mut position,
                    &mut velocity,
                );

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
            draw_text(
                &format!(
                    "Solver: {} ({})",
                    solver.name(),
                    if solver.diag_precond() {
                        "Diag"
                    } else {
                        "Full"
                    }
                ),
                10.0,
                30.0,
                30.0,
                WHITE,
            );

            for p in &position {
                let pos = p.linear * scaling + offset;
                draw_circle(pos.x, pos.y, 0.5 * scaling, RED);
                let rot_x = (rotation_matrix(p.angular) * vector![0.5, 0.0]) * scaling + pos;
                draw_line(pos.x, pos.y, rot_x.x, rot_x.y, 3.0, WHITE);
                let rot_y = (rotation_matrix(p.angular) * vector![0.0, 0.5]) * scaling + pos;
                draw_line(pos.x, pos.y, rot_y.x, rot_y.y, 3.0, GREEN);
            }
            macroquad::window::next_frame().await
        }
    }
}
