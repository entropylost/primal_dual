// Hack to deal with nalgebra stack being slightly broken.
// TODO: File report.
#![allow(clippy::toplevel_ref_arg)]
#![allow(unused)]

use contact::Contact;
use cosserat::{CosseratBendTwist, CosseratRod, CosseratStretchShear};
use macroquad::input::KeyCode;
use nalgebra::{
    self as na, matrix, stack, vector, DVector, Matrix, MatrixXx3, MatrixXx4, SMatrix, SVector,
};
use std::fmt::Debug;
use std::{f32::consts::PI, ops::Deref};

mod opt;
mod split;
use split::Split;
mod contact;
mod cosserat;

type Scalar = na::Matrix1<f32>;
type Vector3 = na::Vector3<f32>;
type RVector3 = na::RowVector3<f32>;
type Matrix3 = na::Matrix3<f32>;
type Matrix3x4 = na::Matrix3x4<f32>;
type Matrix4x3 = na::Matrix4x3<f32>;
type Matrix4 = na::Matrix4<f32>;
type Quaternion = na::Quaternion<f32>;
type UnitQuaternion = na::UnitQuaternion<f32>;

type Split3 = Split<Vector3, Vector3>;
type Position = Split<Vector3, UnitQuaternion>;
type Displacement = Split<Vector3, Quaternion>;
type Velocity = Split<Vector3, Vector3>;
type Force = Split<Vector3, Vector3>;
type Mass = Split<f32, Matrix3>;
type Gradient<const V: usize> = Split<SMatrix<f32, V, 3>, SMatrix<f32, V, 4>>;
type DGradient = Split<MatrixXx3<f32>, MatrixXx4<f32>>;

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
    fn value(&self, positions: [Position; N]) -> SVector<f32, V>;
    fn gradient(&self, positions: [Position; N]) -> [Gradient<V>; N];
    fn stiffness(&self) -> SVector<f32, V>;
}

#[derive(Debug)]
struct ConstraintWrapper<const N: usize, const V: usize, X: Constraint<N, V>>(X);

trait DynConstraint: Debug {
    fn dim_n(&self) -> usize;
    fn dim_v(&self) -> usize;
    fn value(&self, positions: &[Position]) -> DVector<f32>;
    fn gradient(&self, positions: &[Position]) -> Vec<DGradient>;
    fn stiffness(&self) -> DVector<f32>;
    fn potential(&self, positions: &[Position]) -> f32;
    fn force(&self, positions: &[Position]) -> Vec<Force>;
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split3>;
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
    fn value(&self, positions: &[Position]) -> DVector<f32> {
        DVector::from_column_slice(self.0.value(positions.try_into().unwrap()).as_slice())
    }
    fn gradient(&self, positions: &[Position]) -> Vec<DGradient> {
        Vec::from(
            self.0
                .gradient(positions.try_into().unwrap())
                .map(|x| x.dynamic()),
        )
    }
    fn stiffness(&self) -> DVector<f32> {
        DVector::from_column_slice(self.0.stiffness().as_slice())
    }
    fn potential(&self, positions: &[Position]) -> f32 {
        let positions: [Position; N] = positions.try_into().unwrap();
        let value = self.0.value(positions);
        *(value.transpose() * Matrix::from_diagonal(&self.0.stiffness()) * value).as_scalar()
    }
    fn force(&self, positions: &[Position]) -> Vec<Force> {
        let positions: [Position; N] = positions.try_into().unwrap();
        let gradient = self.0.gradient(positions);
        let value = self.0.value(positions);
        gradient
            .into_iter()
            .zip(positions)
            .map(|(grad, pos)| {
                let map = pos.kinematic_map();
                let jc = grad * map;
                Split::new(
                    -jc.linear.transpose() * Matrix::from_diagonal(&self.0.stiffness()) * value,
                    -jc.angular.transpose() * Matrix::from_diagonal(&self.0.stiffness()) * value,
                )
            })
            .collect::<Vec<_>>()
    }
    fn grad2_diag(&self, positions: &[Position]) -> Vec<Split3> {
        let positions: [Position; N] = positions.try_into().unwrap();
        let gradient = self.0.gradient(positions);
        gradient
            .into_iter()
            .zip(positions)
            .map(|(grad, pos)| {
                let map = pos.kinematic_map();
                let jc = grad * map;
                Split::new(
                    (jc.linear.transpose()
                        * Matrix::from_diagonal(&self.0.stiffness())
                        * jc.linear)
                        .diagonal(),
                    (jc.angular.transpose()
                        * Matrix::from_diagonal(&self.0.stiffness())
                        * jc.angular)
                        .diagonal(),
                )
            })
            .collect::<Vec<_>>()
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
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 1.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, -1.0]),
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

    let constraints = [
        ConstraintBox::new([0, 1], se),
        ConstraintBox::new([0, 1], bt),
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

            let mut contacts: Vec<ConstraintBox> = vec![];

            for i in 0..particles {
                for j in i + 1..particles {
                    let pi = position[i];
                    let pj = position[j];
                    if (pi.linear - pj.linear).norm() <= 1.0 {
                        contacts.push(ConstraintBox::new(
                            [i, j],
                            Contact {
                                normal: (pi.linear - pj.linear).normalize().transpose(),
                                stiffness: 10000.0,
                                length: 1.0,
                            },
                        ));
                    }
                }
            }
            for _iter in 0..10 {
                let mut forces = vec![Force::default(); particles];
                let mut grad2_diag = vec![Split::<Vector3, Vector3>::default(); particles];

                for ConstraintBox {
                    targets,
                    constraint,
                } in constraints.iter().chain(&contacts)
                {
                    let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();

                    let force = constraint.force(&p);
                    let grad2 = constraint.grad2_diag(&p);
                    for (i, &j) in targets.iter().enumerate() {
                        forces[j] += force[i];
                        grad2_diag[j] += grad2[i];
                    }
                }
                let mut preconditioner_diag = vec![Split::<Vector3, Vector3>::default(); particles];
                for i in 0..particles {
                    preconditioner_diag[i].linear =
                        (Vector3::repeat(mass[i].linear) + grad2_diag[i].linear).map(|x| 1.0 / x);
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
