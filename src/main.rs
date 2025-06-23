// Hack to deal with nalgebra stack being slightly broken.
#![allow(clippy::toplevel_ref_arg)]
// #[allow(unused)]

use std::f32::consts::PI;
use std::fmt::Debug;
use std::ops::Deref;

use dyn_clone::DynClone;
use macroquad::color;
use macroquad::input::KeyCode;
use macroquad::window::request_new_screen_size;
use nalgebra::{self as na, matrix, vector, DMatrixView, DVectorView, DVectorViewMut};

mod contact;
#[cfg(feature = "2d")]
mod cosserat2d;
#[cfg(feature = "2d")]
use cosserat2d as cosserat;
#[cfg(not(feature = "2d"))]
mod cosserat3d;
#[cfg(not(feature = "2d"))]
use cosserat3d as cosserat;
mod ext;
mod solver;

mod split;
use contact::Contact;
use cosserat::{CosseratBendTwist, CosseratRod, CosseratStiffness, CosseratStretchShear};
use ext::*;
use solver::{DualSolver, PrimalSolver, Solver, Solvers};
use split::{Invertible, Reciprocal, Split};

mod math {
    use nalgebra::{self as na, Const, Dyn};

    use crate::split::Split;

    #[cfg(feature = "2d")]
    pub const P: usize = 2;
    #[cfg(feature = "2d")]
    pub const Q: usize = 1;
    #[cfg(feature = "2d")]
    pub const W: usize = 1;
    #[cfg(not(feature = "2d"))]
    pub const P: usize = 3;
    #[cfg(not(feature = "2d"))]
    pub const Q: usize = 4;
    #[cfg(not(feature = "2d"))]
    pub const W: usize = 3;

    pub type Real = f32;
    pub type SMatrix<const R: usize, const C: usize> = na::SMatrix<Real, R, C>;
    pub type SVector<const N: usize> = na::SVector<Real, N>;
    pub type RowSVector<const N: usize> = na::RowSVector<Real, N>;

    pub type Scalar = na::Matrix1<Real>;
    pub type DVector = na::DVector<Real>;
    pub type DMatrix = na::DMatrix<Real>;
    pub type Vector = SVector<P>;
    pub type RVector = RowSVector<P>;
    pub type VectorW = SVector<W>;
    pub type MatrixP = SMatrix<P, P>;
    pub type MatrixQ = SMatrix<Q, Q>;
    pub type MatrixW = SMatrix<W, W>;
    pub type MatrixWQ = SMatrix<W, Q>;
    pub type MatrixQW = SMatrix<Q, W>;
    pub type SMatrixP<const X: usize> = SMatrix<X, P>;
    pub type SMatrixQ<const X: usize> = SMatrix<X, Q>;
    pub type SMatrixW<const X: usize> = SMatrix<X, W>;
    pub type DMatrixP = na::OMatrix<Real, Dyn, Const<P>>;
    pub type DMatrixQ = na::OMatrix<Real, Dyn, Const<Q>>;
    pub type DMatrixW = na::OMatrix<Real, Dyn, Const<W>>;

    #[cfg(feature = "2d")]
    pub type PartialRotation = Scalar;
    #[cfg(feature = "2d")]
    pub type Rotation = Scalar;
    #[cfg(not(feature = "2d"))]
    pub type PartialRotation = na::Quaternion<Real>;
    #[cfg(not(feature = "2d"))]
    pub type Rotation = na::UnitQuaternion<Real>;

    pub type Position = Split<Vector, Rotation>;
    pub type Displacement = Split<Vector, PartialRotation>;
    pub type Velocity = Split<Vector, VectorW>;
    pub type Force = Split<Vector, VectorW>;
    pub type Mass = Split<Real, MatrixW>;
    pub type Gradient<const V: usize> = Split<SMatrixP<V>, SMatrixQ<V>>;
    pub type DGradient = Split<DMatrixP, DMatrixQ>;
    pub type Jacobian<const V: usize> = Split<SMatrixP<V>, SMatrixW<V>>;
    pub type DJacobian = Split<DMatrixP, DMatrixW>;
    pub type Hessian = Split<MatrixP, MatrixW>;
    pub type HessDiag = Split<Vector, VectorW>;
}
use math::*;

#[cfg(feature = "2d")]
type Matrix2 = SMatrix<2, 2>;
#[cfg(feature = "2d")]
fn rotation_matrix(q: Scalar) -> Matrix2 {
    let q = q.into_scalar();
    matrix![
        q.cos(), -q.sin();
        q.sin(), q.cos()
    ]
}
#[cfg(feature = "2d")]
fn rotation_matrix_gradient(q: Scalar) -> Matrix2 {
    let q = q.into_scalar();
    matrix![
        -q.sin(), -q.cos();
        q.cos(), -q.sin()
    ]
}

#[expect(unused)]
#[cfg(not(feature = "2d"))]
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
            DMatrixP::from_rows(&linear_rows),
            DMatrixQ::from_rows(&angular_rows),
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
            DMatrixP::from_rows(&linear_rows),
            DMatrixW::from_rows(&angular_rows),
        )
    }
}

#[cfg(feature = "2d")]
impl Position {
    fn normalize(mut self) -> Self {
        *self.angular.as_scalar_mut() %= 4.0 * PI;
        self
    }
    fn kinematic_map(self) -> Split<MatrixP, MatrixQW> {
        Split::new(MatrixP::identity(), MatrixQW::identity())
    }
    fn step(self, velocity: Velocity) -> Self {
        (velocity + self).normalize()
    }
}
#[cfg(not(feature = "2d"))]
impl Position {
    fn rotation_map(self) -> MatrixQW {
        let q = self.angular.quaternion().as_vector() / 2.0;
        matrix![
            q.w, q.z, -q.y;
            -q.z, q.w, q.x;
            q.y, -q.x, q.w;
            -q.x, -q.y, -q.z;
        ]
    }
    fn kinematic_map(self) -> Split<MatrixP, MatrixQW> {
        Split::new(MatrixP::identity(), self.rotation_map())
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
#[cfg(not(feature = "2d"))]
impl Displacement {
    fn normalize(self) -> Position {
        Position {
            linear: self.linear,
            angular: Rotation::from_quaternion(self.angular),
        }
    }
}

trait Constraint<const N: usize, const V: usize>: Debug {
    fn value(&self, positions: [Position; N]) -> SVector<V>;
    fn gradient(&self, positions: [Position; N]) -> [Gradient<V>; N];

    fn jacobian(&self, positions: [Position; N]) -> [Jacobian<V>; N] {
        let gradient = self.gradient(positions);
        (gradient, positions).map(|grad, pos| grad * pos.kinematic_map())
    }
    #[expect(unused)]
    fn potential(&self, stiffness: SVector<V>, positions: [Position; N]) -> Real {
        let value = self.value(positions);
        (value.t() * diag(stiffness) * value).into_scalar()
    }
    fn force(&self, stiffness: SVector<V>, positions: [Position; N]) -> [Force; N] {
        let value = self.value(positions);
        self.jacobian(positions).map(|jc| {
            Split::new(
                -jc.linear.t() * diag(stiffness) * value,
                -jc.angular.t() * diag(stiffness) * value,
            )
        })
    }
    // TODO: This treats the linear and angular parts as independent.
    // Instead, it should return a 6x6 matrix with a jc.linear.t() * jc.angular coefficient.
    fn hessian(&self, stiffness: SVector<V>, positions: [Position; N]) -> [Hessian; N] {
        self.jacobian(positions).map(|jc| {
            Split::new(
                jc.linear.t() * diag(stiffness) * jc.linear,
                jc.angular.t() * diag(stiffness) * jc.angular,
            )
        })
    }
    fn hessian_diag(&self, stiffness: SVector<V>, positions: [Position; N]) -> [HessDiag; N] {
        self.jacobian(positions).map(|jc| {
            Split::new(
                (jc.linear.t() * diag(stiffness) * jc.linear).diagonal(),
                (jc.angular.t() * diag(stiffness) * jc.angular).diagonal(),
            )
        })
    }
    fn dual_preconditioner(
        &self,
        stiffness: SVector<V>,
        positions: [Position; N],
        masses: [Mass; N],
    ) -> SMatrix<V, V> {
        let denom = diag(stiffness.reciprocal())
            + (self.jacobian(positions), masses)
                .map(|jc, mass| {
                    jc.linear * mass.linear.inverse() * jc.linear.t()
                        + jc.angular * mass.angular.inverse() * jc.angular.t()
                })
                .sum();
        denom.try_inverse().unwrap()
    }
    fn dual_preconditioner_diag(
        &self,
        stiffness: SVector<V>,
        positions: [Position; N],
        masses: [Mass; N],
    ) -> SVector<V> {
        let denom = stiffness.reciprocal()
            + (self.jacobian(positions), masses)
                .map(|jc, mass| {
                    (jc.linear * mass.linear.inverse() * jc.linear.t()).diagonal()
                        + (jc.angular * mass.angular.inverse() * jc.angular.t()).diagonal()
                })
                .sum();
        denom.reciprocal()
    }
}

#[derive(Debug, Clone)]
struct ConstraintWrapper<const N: usize, const V: usize, X: Constraint<N, V>>(X, SVector<V>);

trait DynConstraint: Debug + DynClone {
    #[expect(unused)]
    fn dim_n(&self) -> usize;
    fn dim_v(&self) -> usize;
    fn value(&self, positions: &[Position]) -> DVector;
    #[expect(unused)]
    fn gradient(&self, positions: &[Position]) -> Vec<DGradient>;
    fn jacobian(&self, positions: &[Position]) -> Vec<DJacobian>;
    #[expect(unused)]
    fn potential(&self, positions: &[Position]) -> Real;
    fn force(&self, positions: &[Position]) -> Vec<Force>;

    fn hessian(&self, positions: &[Position]) -> Vec<Hessian>;
    fn hessian_diag(&self, positions: &[Position]) -> Vec<HessDiag>;

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix;
    fn dual_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector;

    fn stiffness(&self) -> DVectorView<Real>;
    fn stiffness_mut(&mut self) -> DVectorViewMut<Real>;
}

impl<const N: usize, const V: usize, X> DynConstraint for ConstraintWrapper<N, V, X>
where
    X: Constraint<N, V> + Clone,
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
    fn potential(&self, positions: &[Position]) -> Real {
        self.0.potential(self.1, positions.try_into().unwrap())
    }
    fn force(&self, positions: &[Position]) -> Vec<Force> {
        self.0.force(self.1, positions.try_into().unwrap()).into()
    }

    fn hessian(&self, positions: &[Position]) -> Vec<Hessian> {
        self.0.hessian(self.1, positions.try_into().unwrap()).into()
    }
    fn hessian_diag(&self, positions: &[Position]) -> Vec<HessDiag> {
        self.0
            .hessian_diag(self.1, positions.try_into().unwrap())
            .into()
    }

    fn dual_preconditioner(&self, positions: &[Position], mass: &[Mass]) -> DMatrix {
        let pc = self.0.dual_preconditioner(
            self.1,
            positions.try_into().unwrap(),
            mass.try_into().unwrap(),
        );
        let v: DMatrixView<f32> = pc.as_view();
        v.clone_owned()
    }
    fn dual_preconditioner_diag(&self, positions: &[Position], mass: &[Mass]) -> DVector {
        DVector::from_column_slice(
            self.0
                .dual_preconditioner_diag(
                    self.1,
                    positions.try_into().unwrap(),
                    mass.try_into().unwrap(),
                )
                .as_slice(),
        )
    }

    fn stiffness(&self) -> DVectorView<Real> {
        self.1.as_view()
    }
    fn stiffness_mut(&mut self) -> DVectorViewMut<Real> {
        self.1.as_view_mut()
    }
}
dyn_clone::clone_trait_object!(DynConstraint);

#[derive(Debug, Clone)]
struct ConstraintBox {
    targets: Vec<usize>,
    constraint: Box<dyn DynConstraint>,
}
impl ConstraintBox {
    fn new<const N: usize, const V: usize>(
        targets: [usize; N],
        constraint: impl Constraint<N, V> + Clone + 'static,
        stiffness: SVector<V>,
    ) -> Self {
        Self {
            targets: targets.to_vec(),
            constraint: Box::new(ConstraintWrapper(constraint, stiffness)),
        }
    }
}

struct World {
    mass: Vec<Mass>,
    position: Vec<Position>,
    velocity: Vec<Velocity>,
    constraints: Vec<ConstraintBox>,
    substeps: usize,
    contact_stiffness: Real,
}
impl World {
    fn new(
        mass: &[Mass],
        position: &[Position],
        velocity: &[Velocity],
        constraints: &[ConstraintBox],
        dt: Real,
        substeps: usize,
        contact_stiffness: Real,
    ) -> Self {
        let dt = dt / substeps as Real;
        let dt2 = dt * dt;
        let contact_stiffness = contact_stiffness * dt2;

        let mass = mass.to_vec();
        let position = position.to_vec();
        let mut velocity = velocity.to_vec();
        for v in &mut velocity {
            *v = *v * dt;
        }
        let mut constraints = constraints.to_vec();
        for constraint in &mut constraints {
            let mut stiffness = constraint.constraint.stiffness_mut();
            stiffness *= dt * dt;
        }
        Self {
            mass,
            position,
            velocity,
            constraints,
            substeps,
            contact_stiffness,
        }
    }
    fn update(&mut self, solver: &mut impl Solver) {
        let particles = self.mass.len();
        for _ in 0..self.substeps {
            let last_position = self.position.clone();
            let last_velocity = self.velocity.clone();
            for i in 0..particles {
                self.position[i] = self.position[i].step(self.velocity[i]);
            }

            let mut constraints = self.constraints.clone();

            for i in 0..particles {
                for j in i + 1..particles {
                    let pi = self.position[i];
                    let pj = self.position[j];
                    if (pi.linear - pj.linear).norm() <= 1.0 {
                        constraints.push(ConstraintBox::new(
                            [i, j],
                            Contact {
                                normal: (pi.linear - pj.linear).normalize().t(),
                                length: 1.0,
                            },
                            Scalar::new(self.contact_stiffness),
                        ));
                    }
                }
            }

            solver.solve(
                &self.mass,
                &constraints,
                &last_position,
                &last_velocity,
                &mut self.position,
                &mut self.velocity,
            );
        }
    }
}

#[macroquad::main("Primal / Dual")]
async fn main() {
    request_new_screen_size(1000.0, 800.0);

    let mass: Vec<Mass> = vec![f32::INFINITY, 1.0, 1.0, 1.0, 1.0, 5.0]
        .into_iter()
        .map(|x| Split::new(x, MatrixW::from_diagonal_element(2.0 / 5.0 * x * 0.5 * 0.5)))
        .collect();

    let position: Vec<Position> = vec![
        vector![0.0, 0.0, 0.0],
        vector![2.0, 0.0, 0.0],
        vector![4.0, 0.0, 0.0],
        vector![6.0, 0.0, 0.0],
        vector![8.0, 0.0, 0.0],
        vector![8.0, -3.0, 0.4],
    ]
    .into_iter()
    .map(Split::from_linear)
    .collect();
    let velocity: Vec<Velocity> = vec![
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0]),
        Split::new(vector![0.0, 2.0, 0.0], vector![0.0, 0.0, 0.0]),
    ];
    let particles = mass.len();
    assert_eq!(particles, position.len());
    assert_eq!(particles, velocity.len());

    let dt = 1.0 / 60.0;

    let rod = CosseratRod::resting_state([position[0], position[1]]);
    let stiffness = CosseratStiffness::new(0.5, 10000.0, 10000.0);
    let ss = stiffness.stretch_shear(rod.length);
    let bt = stiffness.bend_twist(rod.length);

    let constraints = vec![
        ConstraintBox::new([0, 1], CosseratStretchShear { rod }, ss),
        ConstraintBox::new([0, 1], CosseratBendTwist { rod }, bt),
        ConstraintBox::new([1, 2], CosseratStretchShear { rod }, ss),
        ConstraintBox::new([1, 2], CosseratBendTwist { rod }, bt),
        ConstraintBox::new([2, 3], CosseratStretchShear { rod }, ss),
        ConstraintBox::new([2, 3], CosseratBendTwist { rod }, bt),
        ConstraintBox::new([3, 4], CosseratStretchShear { rod }, ss),
        ConstraintBox::new([3, 4], CosseratBendTwist { rod }, bt),
    ];

    let mut worlds = [
        (
            Solvers::Primal(PrimalSolver {
                iterations: 10,
                constraint_step: 0.5,
                diag_precond: false,
            }),
            1,
            color::BLUE,
        ),
        (
            Solvers::Primal(PrimalSolver {
                iterations: 1,
                constraint_step: 0.5,
                diag_precond: true,
            }),
            10,
            color::RED,
        ),
        (
            Solvers::Primal(PrimalSolver {
                iterations: 3,
                constraint_step: 0.5,
                diag_precond: true,
            }),
            3,
            color::GREEN,
        ),
        (
            Solvers::Dual(DualSolver {
                iterations: 100,
                constraint_step: 0.5,
                diag_precond: false,
            }),
            10,
            color::WHITE,
        ),
    ]
    .map(|(solver, substeps, color)| {
        (
            World::new(
                &mass,
                &position,
                &velocity,
                &constraints,
                dt,
                substeps,
                99999.0,
            ),
            solver,
            color,
        )
    });

    let mut running = false;

    loop {
        if macroquad::input::is_key_pressed(KeyCode::Space) {
            running = !running;
        }
        if macroquad::input::is_key_pressed(KeyCode::Escape) {
            break;
        }

        if running || macroquad::input::is_key_pressed(KeyCode::Period) {
            for (world, solver, _) in &mut worlds {
                world.update(solver);
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
            for (i, (world, solver, color)) in worlds.iter().enumerate() {
                draw_text(
                    &format!(
                        "Solver: {} ({}) [{} / {}], step {}",
                        solver.name(),
                        if solver.diag_precond() {
                            "Diag"
                        } else {
                            "Full"
                        },
                        world.substeps,
                        solver.iterations(),
                        solver.constraint_step(),
                    ),
                    10.0,
                    20.0 * (i + 1) as f32,
                    20.0,
                    *color,
                );

                for p in &world.position {
                    let pos = p.linear.xy() * scaling + offset;
                    draw_circle(pos.x, pos.y, 0.5 * scaling, Color { a: 0.5, ..*color });
                    #[cfg(not(feature = "2d"))]
                    {
                        let rot_x = (p.angular * vector![0.5, 0.0, 0.0]).xy() * scaling + pos;
                        draw_line(pos.x, pos.y, rot_x.x, rot_x.y, 3.0, WHITE);
                        let rot_y = (p.angular * vector![0.0, 0.5, 0.0]).xy() * scaling + pos;
                        draw_line(pos.x, pos.y, rot_y.x, rot_y.y, 3.0, GREEN);
                        let rot_z = (p.angular * vector![0.0, 0.0, 0.5]).xy() * scaling + pos;
                        draw_line(pos.x, pos.y, rot_z.x, rot_z.y, 3.0, BLUE);
                    }
                    #[cfg(feature = "2d")]
                    {
                        let rot_x =
                            (rotation_matrix(p.angular) * vector![0.5, 0.0]) * scaling + pos;
                        draw_line(pos.x, pos.y, rot_x.x, rot_x.y, 3.0, WHITE);
                        let rot_y =
                            (rotation_matrix(p.angular) * vector![0.0, 0.5]) * scaling + pos;
                        draw_line(pos.x, pos.y, rot_y.x, rot_y.y, 3.0, GREEN);
                    }
                }
            }
            macroquad::window::next_frame().await
        }
    }
}
