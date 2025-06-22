use either::Either;

use super::*;

pub enum Solvers {
    Primal(PrimalSolver),
    Dual(DualSolver),
}
impl Solvers {
    pub fn name(&self) -> &'static str {
        match self {
            Solvers::Primal(_) => "Primal",
            Solvers::Dual(_) => "Dual",
        }
    }
    pub fn iterations(&self) -> usize {
        match self {
            Solvers::Primal(solver) => solver.iterations,
            Solvers::Dual(solver) => solver.iterations,
        }
    }
    pub fn constraint_step(&self) -> Real {
        match self {
            Solvers::Primal(solver) => solver.constraint_step,
            Solvers::Dual(solver) => solver.constraint_step,
        }
    }
    pub fn diag_precond(&self) -> bool {
        match self {
            Solvers::Primal(solver) => solver.diag_precond,
            Solvers::Dual(solver) => solver.diag_precond,
        }
    }
}
impl Solver for Solvers {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        match self {
            Solvers::Primal(solver) => solver.solve(
                mass,
                constraints,
                last_position,
                last_velocity,
                position,
                velocity,
            ),
            Solvers::Dual(solver) => solver.solve(
                mass,
                constraints,
                last_position,
                last_velocity,
                position,
                velocity,
            ),
        }
    }
}

pub trait Solver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    );
}

pub struct PrimalSolver {
    pub iterations: usize,
    pub constraint_step: Real,
    pub diag_precond: bool,
}

impl Solver for PrimalSolver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        let particles = mass.len();
        for _ in 0..self.iterations {
            let mut forces = vec![Force::default(); particles];
            let mut hessians = if self.diag_precond {
                Either::Left(
                    mass.iter()
                        .map(|m| Split::new(Vector::repeat(m.linear), m.angular))
                        .collect::<Vec<_>>(),
                )
            } else {
                Either::Right(
                    mass.iter()
                        .map(|m| Split::new(MatrixV::identity() * m.linear, m.angular))
                        .collect::<Vec<_>>(),
                )
            };

            for ConstraintBox {
                targets,
                constraint,
            } in constraints
            {
                let p = targets.iter().map(|&i| position[i]).collect::<Vec<_>>();

                let force = constraint.force(&p);
                for (i, &j) in targets.iter().enumerate() {
                    forces[j] += force[i];
                }

                match &mut hessians {
                    Either::Left(hessians) => {
                        let hessian = constraint.hessian_diag(&p);
                        for (i, &j) in targets.iter().enumerate() {
                            hessians[j] += hessian[i];
                        }
                    }
                    Either::Right(hessians) => {
                        let hessian = constraint.hessian(&p);
                        for (i, &j) in targets.iter().enumerate() {
                            hessians[j] += hessian[i];
                        }
                    }
                }
            }
            let step = (0..particles)
                .map(|i| {
                    let grad = mass[i] * (velocity[i] - last_velocity[i]) - forces[i];
                    match &hessians {
                        Either::Left(hessians) => {
                            let precond = hessians[i].recip();
                            precond.component_mul(grad)
                        }
                        Either::Right(hessians) => {
                            let precond = hessians[i].inverse();
                            precond * grad
                        }
                    }
                })
                .collect::<Vec<_>>();
            for (i, step) in step.into_iter().enumerate() {
                velocity[i] -= self.constraint_step * step;
            }
            for i in 0..particles {
                position[i] = last_position[i].step(velocity[i]);
            }
        }
    }
}

pub struct DualSolver {
    pub iterations: usize,
    pub constraint_step: Real,
    pub diag_precond: bool,
}
impl Solver for DualSolver {
    fn solve(
        &mut self,
        mass: &[Mass],
        constraints: &[ConstraintBox],
        last_position: &[Position],
        _last_velocity: &[Velocity],
        position: &mut [Position],
        velocity: &mut [Velocity],
    ) {
        let particles = mass.len();
        let mut dual_vars = constraints
            .iter()
            .map(|x| DVector::zeros(x.constraint.dim_v()))
            .collect::<Vec<_>>();
        for _ in 0..self.iterations {
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
                let precond = if self.diag_precond {
                    DMatrix::from_diagonal(&constraint.dual_preconditioner_diag(&p, &m))
                } else {
                    constraint.dual_preconditioner(&p, &m)
                };
                let delta = self.constraint_step * precond * dual_force;
                dual_vars[i] += &delta;
                let jacobian = constraint.jacobian(&p);
                for (j, &k) in targets.iter().enumerate() {
                    velocity[k].linear +=
                        mass[k].linear.inverse() * jacobian[j].linear.transpose() * &delta;
                    velocity[k].angular +=
                        (mass[k].angular.inverse() * jacobian[j].angular.transpose() * &delta)
                            .into_scalar();
                }
            }
            for i in 0..particles {
                position[i] = last_position[i].step(velocity[i]);
            }
        }
    }
}
